# 搜索策略改造：从"贪心单候选"到"可测量的方向路由"

现状问题（来自当前实现，不是猜测）：

- `search()` 每轮只产出一个候选，`select_proposal()` 用
  `priority + category_successes - category_failures` 排序
  （[search.py:294-305](../src/afd_rsi/search.py)）。这是一个**未做难度校正的原始计数**规则：
  难度更高的类别只要试过一次失败就会被永久压低，而恰好先试到的类别会因为早期运气获得长期优势。
- 账本只有"类别 → 成功/失败次数"，没有**算子/改动特征**，也没有**技能版本**。
  知识或技能更新后，历史记录仍然以全权重参与排序，无法表达"这条经验已经过期"。
- 每轮只测 1 个候选，没有并行臂，也没有 good/medium/poor 分层再利用；
  被拒绝的候选只在 `_lesson()` 里留下一条记录，不会被当作"下一轮的起点"。
- 停机条件是预算耗尽或模型自愿 `stop`，没有**收益上界**判据，
  因此无法回答"继续搜还有没有空间"。
- `deep/incremental` 是按轮次比例切的时间表（[search.py:139-150](../src/afd_rsi/search.py)），
  不是按证据决定策略，和"当前最佳离硬件上限还有多远"无关。

下面借鉴 CAKE（CANNBOT-CAKE.pdf）的五个机制给出改造方案，全部保持 afd-rsi 的既有不变量：
冻结 plan、候选只能改 `allowed_files`、双 PASS 才晋升、证据哈希复核、模型不参与验收。

---

## 1. 借鉴的机制与落点

| CAKE 机制 | 原文位置 | afd-rsi 落点 | 数据形态 |
| --- | --- | --- | --- |
| L0 可行性闸门 + abstain | §3 L0 | 类别 × 后端 × 算子族的静态可行性表；无可行臂时回退到今天的行为，不猜 | `strategy_feasibility.json`（随 foundation 冻结） |
| L1 八维特征 + 难度 b=w·φ | §3 Eq.(1) | 从 **profile + 补丁内容** 提取改动画像 φ，算难度 b | `arm_features.json`，来源可追溯 |
| L2 item-response 能力 + 每维交互 | §3 Eq.(2)(3) | 类别能力 a0 + 每维增量 δ，难度 b 在比较中抵消 | `search/ledger.jsonl` |
| 信息门控探索 | §3 "Selecting tracks for exploration" | 只在"可能反超"且"信息量足够"时花一轮探新臂 | 同上 |
| (track, skill_version) 半权 | §3 "Evaluation ledger" | 知识/技能更新后旧记录半权，并触发重测奖励 | 账本行带 `foundation_hash` |
| 每轮 N 变体 + 三层采样 | §4.2 | 一轮内并行生成 N 个候选，按结果分档，下一轮从各档采样 | `search/rounds/<NNNN>/candidates/` |
| roofline 停机判据 | §4.1 Eq.(4)(5) | 用 headroom 判断"还有没有空间"，替代"用完预算" | `headroom.json` |
| PMR 蒸馏 | §4.2 | 反复有效的优化模式沉淀为类别规则，供后续 category 选择 | `search/rules/*.json` |

---

## 2. 改动分层（按性价比排序）

### P0：可测量 + 难度校正（不改流程形态）

**目标**：让"该试哪个方向"从裸计数变成带难度校正的估计。

1. 账本追加 `search/ledger.jsonl`，一行一次已评估候选：

```json
{"schema_version": 1, "round": "0003", "category": "tiling",
 "features": {"T": 0, "S": 1, "B": 2, "R": 0, "D": 1, "M": 1, "F": 0, "C": 0},
 "difficulty": 1.42, "decision": "promoted", "stage": "confirmation",
 "score": 0.61, "gain_pct": 2.4, "foundation_hash": "…", "record_hash": "…"}
```

字段名与 `docs/prototypes/router_core.py` 的 `Ledger.add()` 一致，可直接对照。
追加写、不可改；`record_hash` 沿用现有 `identity()` 的规范化哈希做法
（[search.py:21](../src/afd_rsi/search.py)），`verify()` 逐行复核，篡改即抛错。

2. 难度 b 用固定权重，随 foundation 冻结：

```toml
[search.features]
weights = { T = 0.51, S = 0.78, B = 0.59, R = 0.70, D = 0.44, M = 0.61, F = 0.64, C = 0.98 }
```

权重不是随便给的：CAKE 的 Eq.(1) 用同一组量级。**必须写进 plan.lock 并纳入
`fingerprint()`**（[search.py:25-29](../src/afd_rsi/search.py)），否则跨 campaign 的经验会不可比。

3. 选择规则改为：

```
score(category) = a0[cat] + δ[cat]·φ − b
```

- 观测足够（有效权重和 ≥ 维数+1）才拟合 δ 与协方差，否则**回退到标量 a0**，并在
  `routing.json` 里记录 `kind: scalar`（CAKE 同样带 fallback，且写进决策记录）。
- **不引入第三方依赖**：8 维 ridge WLS 用手写高斯消元即可，`router_core.py` 已验证
  （README 明确控制程序无第三方运行时依赖）。

4. 停机不变：`min_gain_pct`、双 PASS、`max_rounds` 全部保留。这一层只换排序函数。

**风险**：8 维 + 小样本容易过拟合。缓解：ridge 惩罚 + 维数不足时回退。
配方预算已提到 `max_rounds = 9`（见下），但**账本只在某轮判定后才增加一行**，
所以第 k 轮开始时最多有 k−1 行；单个类别要在同一 campaign 内被判定 9 次才会
进入向量档。实践中大多停在标量档，这是设计预期，不是缺陷。

### P1：信息门控探索（让预算花在能反超的臂上）

```python
# 概念代码；插入 select_proposal 之前
best = max(categories, key=lambda c: score(c, phi))
for c in categories:
    if c is best: continue
    p, se = predict(c, phi)            # 能力估计 + 标准误
    if p * (1 - p) >= info_floor and p + 1.96 * se > score(best, phi) + margin:
        return c                        # 探索臂
return best
```

- `info_floor`、`margin` 冻结在配方里，不由模型给。
- 探索臂的结果**与利用臂同等记账**，但若 PASS 仍要走完整晋升流程（不降低门槛）。
- 探索预算上限：`explore_max_fraction`（建议 0.25），防止把轮次全花在探索上。

### P2：并行臂 + 分层采样（结构改动最大）

当前一轮一个候选。改成一轮 N 个候选（N=2~3，受设备与时间约束）：

- `search/rounds/<NNNN>/candidates/<a|b|c>/`，各自 source、patch、attempt。
- 判定后按 `gain_pct` 分 good / medium / poor 三档：
  - good：进入 confirm（今天的复测路径）
  - medium：保留源码，作为下一轮的**父本**之一
  - poor：只留记录与失败原因
- 下一轮的 request 里，"父本"不再只有 incumbent，还有 medium 档候选；
  但**best 仍然只能被双 PASS 更新**——采样策略只影响探索，不影响晋升。

代价：`phase_jobs` 与 attempt 目录要支持一轮多 attempt；`state.json` 的
`active` 从单值变成列表。这是对 `search()` 状态机的实质改动，建议放到 P0/P1 观测到
"经常连续 2 轮被拒"之后再上。

### P3：roofline headroom 停机判据

CAKE 的 `T_min = max(t_hbm_read, t_write, t_cube, t_vector)`（Eq.4）在 afd-rsi 里可以简化：

- 输入形状、dtype、设备频率、HBM/L2 带宽、cube/vector 峰值是**冻结的配方/硬件事实**，
  可以算出一个乐观下界 `t_floor`；
- `headroom = 实测中位数 / t_floor`；
- 连续 K 轮 headroom 下降 < ε 时标记 `converged`，把预算让给别的方向或直接停。

**必须写明边界**：这是"乐观分析下界"，不是实测可达延迟；
MegaMoe 是融合任务，通信与等待可能让真实值长期高于下界。
这与现有 `staged-validation.md` 里"不能凭任务总时长断言 Cube/MTE/通信 bound"的措辞一致，
方案要沿用同一口径，不得把 headroom 当作收益声明。

### P4：PMR 规则蒸馏 + 三层知识治理

对应 CAKE Book 的"快速记忆 / 稳定记忆"分离：

- 每次 `promoted` 的经验，若同一 `(category, feature 模式)` 在 ≥2 个不同 campaign 命中，
  写 `search/rules/<id>.json`（候选规则，带证据 run id 列表）；
- 规则**不自动改知识**：照现有 `AGENTS.md` "no automatic knowledge promotion" 的约定，
  只能由人把它搬进 `knowledge/` 并重冻结 manifest；
- 反之，若某条规则在后续 campaign 里连续失效，标记 `stale` 并保留，不覆盖历史。

---

## 3. 接口草案（最小改动面）

### 3.1 配方新增字段

```toml
[search]
router = "auto"                # auto | off
info_floor = 0.0               # 探索的信息量下限，0 = 关闭（默认）
margin = 0.02                  # 反超所需的估计优势
explore_max_fraction = 0.25    # 探索轮占比上限

[search.features]
weights = { T = 0.51, S = 0.78, B = 0.59, R = 0.70, D = 0.44, M = 0.61, F = 0.64, C = 0.98 }
```

由 `contracts.routing()` 校验并补齐默认值后写进 `plan.lock.json`：键集合严格、
权重必须恰好是这 8 个维度且为有限正数、三个比例必须落在 `[0,1]`。
`search.fingerprint()` 只有在配方**声明**了这些字段时才把它们并入指纹，
所以旧 campaign 的历史经验不会被静默判为不可比。

### 3.2 特征提取

φ 从补丁文本推导，不额外测量，也不读 profile（避免把测量噪音混进难度属性）：

| 维度 | 含义 | 判据（仅统计新增行） |
| --- | --- | --- |
| T | tiling 结构变化 | `tile`、`tiling`、`basem`、`basen` |
| S | 调度深度 | `pipeline`、`queue`、`setflag`、`waitflag`、`barrier` |
| B | 分块粒度 | `block`、`basek`、`blockidx` |
| R | 归约 | `reduce`、`atomic`、`accum`、`sum` |
| D | dtype/量化 | `w4a8`、`int8`、`fp16`、`bf16`、`quant`、`cast` |
| M | 数据搬移 | `datacopy`、`stride`、`offset`、`address`、`gm_to_ub`、`ub_to_gm` |
| F | 融合广度 | `dispatch`、`combine`、`ffn`、`attn`、`moe`、`gate` 的不同命中数 |
| C | Cube 强度 | `matmul`、`mmad`、`cube` |

每维饱和到 3。实现在 `src/afd_rsi/router.py`：`patch_features()` 与
`difficulty(features, weights)`。

### 3.3 选择函数替换点

`select_proposal()` 保持签名和返回单个 proposal 的语义；路由结果由
`route_proposals()` 提供，两者共用去重逻辑：

```python
def route_proposals(proposals, records, parent, router=None):
    fresh = {i: p for i, p in enumerate(proposals)
             if identity(p["patch"].strip()) not in seen_from_parent}
    if not fresh:
        return None, None
    if router is None or not router.enabled:
        return legacy_max_priority_plus_counts(fresh, records), None
    decision = router.rank([{"category": p["category"], "features": router.features(p["patch"]),
                             "priority": p["priority"], "index": i} for i, p in fresh.items()])
    return fresh[decision["index"]], decision
```

关键约束：**路由器只改"类别之间的排序"，不改"同一类别内 priority 的含义"**，
也不改补丁白名单、`apply_patch`、attempt 与晋升路径。`select_proposal()` 不接受
router 时逐字节等价于旧实现。

### 3.4 记录与回放

- `search/rounds/<NNNN>/routing.json` 保存当轮决策：各臂期望得分与标准误、
  难度、账本行数、是否探索、以及使用标量还是向量估计。
- `search/ledger.jsonl` 每行带 `record_hash`，重新载入时逐行复核，
  篡改或损坏直接抛错，不降级。
- 崩溃后重放同一 round 不会重复记账（按 round 名去重）。
- 行内 `stage` 记录该得分来自 `screen`（算子级）还是 `confirmation`（完整级）：
  `operator-first` 配方里多数候选止步于初筛，两者会混在同一回归中，
  这个字段让事后分析可以按层级过滤，而不是假装它们同质。

---

## 4. 原型验证（可复现）

三个原型都在 `docs/prototypes/`：

| 文件 | 作用 | 运行方式 |
| --- | --- | --- |
| `router_core.py` | **可直接合并的实现雏形**：账本 + 手写高斯消元 ridge WLS + 标量回退 + 逐行哈希校验，纯标准库 | `python docs/prototypes/router_core.py` → `router_core self-check: ok` |
| `router_sim.py` | 机制对照：固定排序 / 纯利用 / 信息门控探索，含中途漂移 | 需 numpy |
| `router_sensitivity.py` | 三场景 × 20 种子的均值与最坏值 | 需 numpy |

`router_core.py` 自检覆盖的关键性质：

- 8 维 + 小样本**必然回退到标量 a0**，不硬拟合（两行记录无法识别八维）。
- 记录全部来自其他 foundation 时，半权后有效样本低于识别阈值 → 回退标量；
  只有**混合版本**才拟合向量。这是 CAKE `(track, skill_version)` 半权的可执行语义。
- 账本行被篡改时 `verify()` 抛错，不降级。

模拟结论（5 条策略臂、8 维特征、噪声结果）：

| 场景 | 固定排序（今天） | 纯利用（能力 argmax） | 路由（能力+信息门控探索） |
| --- | --- | --- | --- |
| 可分离 | regret 0.2025 | 0.0640（−68.4%） | 0.0684（−66.2%） |
| 近似并列 | 0.0192 | 0.0103（−46.5%） | 0.0102（−47.2%） |
| 中途漂移 | 0.2521 | 0.1153（−54.2%） | **0.0951（−62.3%）** |

读法与限制：

- **难度校正是主要收益来源**（三行里纯利用都比固定排序好 46%~68%）；
- **信息门控探索只在非平稳时额外有价值**（漂移行 −62.3% vs −54.2%），
  在可分离场景里略差于纯利用（多花的探索轮）；
- 因此 P1 应当**默认关闭**，只在检测到"每轮都换臂"或"知识更新后历史失效"时打开；
- 这些数字来自合成模型，只证明机制方向，不构成 afd-rsi 在 NPU 上的收益预期。

---

## 5. 与现有不变量的兼容性检查

| 现有约束 | 是否受影响 | 处理 |
| --- | --- | --- |
| plan 冻结与 `plan.sha256` | 受影响（新增 `[search.features]`） | 纳入 `contracts.py` 校验与 `fingerprint()` |
| 候选只能改 `allowed_files` | 不受影响 | 选择策略不改补丁应用路径 |
| 双 PASS 才晋升 | 不受影响 | 采样/探索只影响提案，不影响晋升 |
| 证据哈希复核 | 受影响（新账本） | 账本行带 `record_hash`，`audit` 时复核 |
| 跨 campaign 经验可比 | 受影响 | 账本行带 `foundation_hash`，不同版本半权 |
| 无第三方依赖 | 受影响 | 手写 8 维 WLS，不引入 numpy |
| 模型不参与验收 | 不受影响 | 路由器是控制器代码，不是模型调用 |
| 旧 campaign 可恢复 | 受影响 | 新字段缺省 → 回退到今天的选择规则 |

**旧 campaign 兼容**：账本缺失或 `search.features` 缺失时，`select_proposal` 走原路径，
并在 `state.json` 记录 `router: disabled`。这与仓库既有做法一致
（`validation` 字段缺省 = full）。

---

## 6. 实施状态

| 阶段 | 状态 | 落点 |
| --- | --- | --- |
| S1 账本 + 难度 + 标量/向量估计 + 回退 | **已实现** | `src/afd_rsi/router.py`；`tests/test_router.py` |
| S2 接入提案选择（探索默认关闭） | **已实现** | `search.route_proposals()`；`select_proposal()` 签名与旧规则不变 |
| S3 信息门控探索（配方开关） | **已实现（默认关）** | `info_floor`/`margin`/`explore_max_fraction`，受 `Router._explore_allowed` 配额约束 |
| S4 并行臂 + 三档采样 | 未实施 | 需改 `phase_jobs` 与 attempt 目录语义 |
| S5 headroom 停机 | 未实施 | 先只记录，不下判定 |
| S6 PMR 规则候选 | 未实施 | 人工评审 |

已实现部分的实际行为：

- 配方通过 `contracts.routing()` 补齐默认值并冻结进 `plan.lock.json`；
  `search.fingerprint()` 只有在配方**声明**了 router 字段时才把这些值并入指纹，
  因此旧 campaign 的历史经验仍然可读、可复用。
- 账本 `search/ledger.jsonl` 逐行 `record_hash`；`_record_ledger()` 在
  `decided` 阶段写入，并在崩溃重放时按 round 名去重，保证追加语义。
- 每轮写 `search/rounds/<NNNN>/routing.json`；`search-report.md` 增加一行
  "Direction routing"，说明当前估计档位与账本行数，并声明它不能替代晋升。

---

## 7. 实施顺序与验收

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S1 | 账本 + 难度 + 标量/向量估计 + 回退 | `python -m pytest`；CPU 合成 campaign 跑通；账本哈希可复核 |
| S2 | 把选择规则接入 `select_proposal`（默认开启，`info_floor` 关闭） | 两个 CPU campaign 对比：被拒候选数、平均 gain |
| S3 | 信息门控探索（配方开关） | 只在漂移场景开启；对比探索轮占比 |
| S4 | 并行臂 + 三档采样 | 需要真机；先验证 attempt 目录与恢复语义 |
| S5 | headroom 停机 | 先只做"记录不下判定"，确认口径无误再接停机 |
| S6 | PMR 规则候选 | 人工评审流程，不自动改知识 |

每个阶段都必须满足：`python -m pytest` 通过、`PYTHONPATH=src python -m afd_rsi check` 通过、
失败尝试保留、不修改冻结的验证器与门槛。

注意：`tests/` 依赖 `fcntl`，**只能在 Linux 执行机跑**。在 Windows 上 `pytest` 会在
collection 阶段就因 `ModuleNotFoundError: No module named 'fcntl'` 失败
（已实测），所以 S1 的本机验证必须是 CPU 合成 campaign + 单元测试在 Linux 上跑。

---

## 8. 明确不做的事

- 不让模型决定 category 权重或难度权重（那等于让被测者定义评分）。
- 不把 headroom 当作收益声明，也不放宽 `min_gain_pct`。
- 不用探索臂替晋升：任何 PASS 都必须走两次独立 attempt 与最终服务验收。
- 不做跨 campaign 的知识自动晋升（与 `AGENTS.md` 一致）。
- 不靠提高 `max_rounds` 去假装向量能力模型已经生效：9 轮是识别八维的下界，
  多数 campaign 仍会停在标量 a0（这也是 CAKE 的 fallback 语义）。
