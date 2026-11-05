# 自动搜索、服务验收与续跑

框架在 Linux 执行机本地运行，Agent 仅提出补丁。新 Ascend 配方使用
`validation="operator-first"`：资格验证基线并收集一次 profile，候选先做原生
正确性与 AB/BA 计时，再以真实模型服务主指标筛选；新 attempt 对同一 incumbent
反转服务顺序并独立复测。两轮模型收益都通过且完整资格达标后才替换 best。
未设置该策略的 full 配方保留原顺序。报告不会把合成字段或 HTTP 指标当作算子计时。

## 选择提案模型

`[search].provider` 决定由哪个冻结适配器写补丁，`codex` 与 `deepseek` 现成可用：

```toml
[search]
provider = "deepseek"

[search.models]              # 可选；必须写在文件末尾的独立表
analysis = "DeepSeek-V4.1-Flash-1017"
code = "DeepSeek-V4.1-Flash-1017"
```

`[search.models]` 让两个阶段各自冻结：只写一个阶段时另一个取默认值，所以以后换了
更擅长编码的模型，只需改 `code` 一行。声明后这两项进入经验指纹，环境变量不再覆盖
已冻结的值（未声明的 campaign 仍可用环境变量临时试模型）。

声明后适配器命令自动补成 `{foundation}/benchmarks/agents/<provider>_agent.py`；
旧配方手写完整 `agent_command` 的写法继续有效，两者必须一致，否则 prepare 阶段
直接报错。请求会记录所选适配器的相对路径与哈希，`propose` 与控制器都会复核，
所以"换了模型却沿用旧请求"会失败而不是静默降级。声明 `provider` 的配方使用独立
的经验指纹——不同模型的方向排序历史不会互相复用。

### DeepSeek 需要和本机模型网关一致的三个输入

适配器不内置网关地址，也不假设凭据变量名：凭空给的默认值会让 campaign 去调用一个
根本不服务该模型的地址。三项都在提案机上配置：

```bash
export AFD_RSI_DEEPSEEK_BASE_URL='http://<gateway-host>:<port>/v1'   # 含 API 前缀
export AFD_RSI_DEEPSEEK_KEY_ENV=CSI_API_KEY   # 该网关实际使用的凭据变量名
export CSI_API_KEY=...                        # 只存在环境里，不落盘
```

模型标识默认 `DeepSeek-V4.1-Flash-1017`，可用 `[search.models]` 或
`AFD_RSI_DEEPSEEK_MODEL` 改。三项要和发起提案那台机器上已有的模型网关配置一致
（例如同一份 `providers` 条目里的地址、凭据变量与模型名）。

先验证再搜索：

```bash
python -m afd_rsi probe            # Linux 执行机
python benchmarks/agents/deepseek_agent.py --check   # 提案机（可为 Windows）
```

它逐项报告配置、连通性、网关模型清单、能否生成、JSON 输出是否可用，第一个失败项
直接指出该改哪个变量。`401` 是凭据问题，`404` 通常是地址缺 `/v1`，`503` 是网关
自身没有健康上游——属于可用性问题，不是配置问题，等网关恢复或换时间重试。
`--output` 可把报告存成 JSON 便于归档。

适配器只用标准库，不需要额外 SDK。提案机可以是 Windows：适配器不依赖 POSIX，
会话锁在两个平台都可用。换模型的唯一代价是 `max_calls` 语义不变、预算不变、
验收门槛不变。

思考等级也按阶段分开，默认分析方向用 `max`、写补丁用 `high`
（`AFD_RSI_DEEPSEEK_EFFORT_ANALYSIS` / `AFD_RSI_DEEPSEEK_EFFORT_CODE` 可按主机覆盖），
两者都记进 `usage.json` 与 `session.json`，便于事后核对用的是哪一档。
细则见 [提案模型提供方](../knowledge/measurement/agent-providers.md)。

## 单机 Codex

需要 Python 3.11+、Linux pidfd、已安装并登录的 Codex CLI。沿用 CLI 的 provider
配置和登录，不读取或复制认证文件；不需要额外模型 SDK。模型与推理强度显式固定：

- 实验操作：`gpt-6.1-sol` / `medium`。固定控制器直接执行实验，无需额外模型调用。
- 分析证据、profiling 和优化方向：`gpt-6-astra` / `xhigh`。
- 编写候选补丁：`gpt-6.1-sol` / `xhigh`。

这里将 ultra high 对应为 CLI 的 `xhigh`。提案先分析再写补丁；分析决定停止时不调用
编码模型。两步共享原有超时和 deadline，不静默降级；归档各自命令、输出、JSONL、
stderr 和带模型标签的 usage。`max_calls` 限制 Agent 命令派发次数，每次内含至多
两次模型调用。外部文件交换采用相同流程。模型规则固化在仓库
`knowledge/measurement/codex-models.md`。修改后的提案器用于新 campaign。
Ascend 样例给每次提案 1200 秒和 524288 字节上下文；两步模型调用共享超时。
priority=3 为优先、1 为备选，控制器按方向路由的估计能力排序类别，同类别内才比较
priority。

使用仓库固定的 JSON
schema 和只读模式；补丁由控制器进行 git-apply 和允许文件检查。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m afd_rsi prepare recipes/cpu_search.toml --output /tmp/rsi-search-demo
python3 -m afd_rsi search /tmp/rsi-search-demo
python3 -m afd_rsi report /tmp/rsi-search-demo
python3 -m afd_rsi export /tmp/rsi-search-demo --best --output /tmp/rsi-best
```

该 recipe 的耗时是合成字段，只验证 Agent/控制器的合同。实际 NPU 使用
`recipes/ascend_eager.toml` 或 `recipes/ascend_graph.toml`：先按现场修改 source、已分配设备、模型路径、网络和
预算，再 prepare。其原生驱动确实运行目标的 build_ext 和 torchrun A/F
fixture，比较两个 MegaMoe 版本，使用共同 layered 数值参考，校验实际加载库和
设备对象身份；主指标是主 Attention rank 的七个 case 事务样本均值（us）。
FFN rank 没有伪造计时，仍必须给出完整数值/身份/清理检查。graph 配方额外检查
实际 FULL 捕获/replay 和禁止 eager 回退；参考输出仍来自 layered eager。

## 长时间运行与预算

这是长期连续运行的项目，一场 campaign 正常就要跑数小时：构建、数值、AB/BA、
服务阶段串起来，单轮可能远超一次模型调用。因此：

- `budget_seconds` 是**安全上限**，不是排期表。冻结进 plan 后在 prepare 时算成
  `expires_at`，用于防止失控，而不是用来安排什么时候结束。样例给到 24 小时，
  合成样例 1 小时。
- 到点后控制器不再开启新一轮，并记为 `budget_exhausted`。若当时正处在一轮中间，
  那一轮已经写下的样本仍保留，判定为 `INCONCLUSIVE`——报告会写明缺什么，
  不会把半轮结果当成收益或当成失败。
- 运行期间用 `afd_rsi report RUN`、`afd_rsi optimize ... --follow` 或直接读
  `search/state.json` 的 `status` 看进展。**不要因为"看着像卡住"就杀掉再重来**：
  重新 prepare 会换掉 manifest/plan 身份，之前的 attempt 和证据就不再属于同一个
  campaign。要续跑就在原目录 `afd_rsi resume RUN`。
- 真要延长上限时，正确做法是**准备新 campaign 并复用 `experience_dirs`**，
  而不是改 `plan.lock.json`——那个文件被 sha256 冻结，改动会被拒绝。

## 服务是 campaign 的固定阶段

`[service]` 不再只启动外部独立作业。控制器指向本 attempt/side 的隔离原生构建，
调用仓库固定版本 AFD E2E 的 plan/run：标准 vllm serve、HTTP 就绪、真实请求、
正常关闭和有界清理。服务输出、实际加载库映射、源码/产物身份及哈希归档在该
side/service/trial 中。带服务主指标的初筛只跑 smoke 和主性能场景；复测两侧
运行完整 GSM8K 与所有冻结场景。每个来源重新加载、核对身份并清理。

服务默认要求完整生命周期通过。本轮按需求在样例中固定
`require_clean_shutdown = false`：只暂不考察 E2E 已归类为 shutdown 的错误，
原始 status/shutdown_status 仍保留失败，并记录 shutdown_errors_ignored。推理、
全部冻结测试、实际映射库身份和 cleanup 必须通过；报告/启动/推理错误不能
借此忽略。该策略只能在 prepare 前设置，Agent 无权修改。没有把服务结果伪装为
native 全 rank 指标。样例配置为真实 V4 W4A8、
16 设备 DP2TP4/EP8，eager/graph 按 recipe 分开冻结。完整复测执行 smoke、300 条 GSM8K
strict-match（V4 Flash 门槛 284/300，V2 Lite 默认门槛 0.27）以及九点 prefill：
128/512/2048 输入 token、1 输出 token、并发 1/4/8，每点 2 次 warmup、8 次测量。离线数据集须已缓存，或在 prepare 前配置 `--dataset-path`。
GSM8K 样本数/分数和 HTTP TTFT mean/p50/p90 进入观察、比较报告及 Agent 上下文。
eager/graph 配方使用 point-008（2048 输入、并发 8）的 TTFT p50 作主指标；
它包含服务/网络开销，合成原生耗时只供诊断。这组小样本的收益会有噪声，
需结合原始分布和独立复测判断，不代表正式生产性能资格。

## 自动 profiling 和瓶颈假设

样例 `[native].profile = true`：初始基线通过原生数值和计时后收集一次 profile，
生成首份 Agent 请求。普通初筛不做完整 profile；独立复测模型收益和最终服务门槛
全部通过后才收集候选 profile。初始基线未做服务，不作为模型精度资格。

复用已有 torch_npu profiler 子进程，保留每个 FFN rank 的原始 trace、
`kernel_details.csv`、加载身份与哈希；控制器汇总 top kernels 的次数、总/平均设备
任务时长及占比，不导入 torch。已完成 profile 可在续跑复核后复用，篡改或不完整
结果不能通过；异常使用现有健康检查屏障。profile 期间的耗时不进入主指标。

Agent 请求会包含当前最佳版本的 profile、最近候选的 profile 和历史判定，并加载
仓库内性能策略 skill。Agent 从设备耗时最大的任务出发，结合源码和 tiling 提出可
证伪的小补丁，后续仍走数值、GSM8K、prefill、独立计时和确认流程。MegaMoe 是
融合任务，仅凭任务总时长无法断言 Cube/MTE/通信 bound；原生合成 trace 也不能
单独解释模型 TTFT。具体证据边界已固化到仓库本地知识。

独立 profile 还会构建一份带算子内部打点的诊断副本，采集 72 个核的
GetSystemCycle 阶段、依赖等待和发布区间，并检查数值、记录上限和 scratch guard。
生产服务与普通计时继续使用未打点产物。当前 profile 摘要包含选定核前四个 tile，
覆盖双槽首次复用；历史 profile 保留阶段总量和原始归档，避免重复时间线占满
上下文。源码、测量事实和仓库知识优先于日志，预算按实际 JSON 文件大小计算。
每个核使用自己的时间原点，不能据此对齐跨核时钟或声称 Cube 物理利用率。

## Agent 和 NPU 不在同一台机器

可用统一 `optimize --executor itask` 入口自动完成以下交换与轮询，见
[统一后台入口](orchestration.md)。以下保留手工流程，便于检查或接入已有 campaign。

在执行机创建 campaign，并进入文件交换模式：

```bash
python3 -m afd_rsi search /path/campaign --external-agent
```

`campaign/search/pending.json` 给出 request、response 的相对路径、请求 SHA 和
已经预扣的调用序号。将其中 request.json 复制到已登录的提案机，在同版本仓库
执行（手工模式自行传输；统一入口可自动传输）：

```bash
python3 -m afd_rsi propose /path/request.json --output /path/new-proposer-output
```

将输出目录整个保留以归档 Codex JSONL、usage、stderr 和进程回执，将
proposal.json 放回 pending 指定的 response 路径，再在执行机运行：

```bash
python3 -m afd_rsi resume /path/campaign
```

它消费已提交的回复，应用补丁并验证，完成后可能产生下一份 pending。外部编排
可重复上述文件交换。没有回复时返回等待，不重复扣调用次数。`propose` 只读
请求内的源码/证据/本仓库知识，校验提案器身份、响应大小和 schema，并实施
请求中的超时；不会拿到执行机的构建或 NPU。认证文件始终留在提案机。

## 方向路由

每轮真正要选的只有一件事：把下一次昂贵尝试花在哪个改动方向上。旧规则是
`priority + 类别晋升数 − 类别失败数`，难点类别失败一次就被永久压低，先试到的
类别靠早期运气长期占优。现在默认改为按能力估计排序，但**只改排序，不改晋升**。

- `search/ledger.jsonl` 追加记录每个已评估候选：类别、八维改动画像、难度、
  归一化得分、判定、证据哈希。`foundation` 变化的旧记录保留半权，知识更新后
  历史证据会自然变旧，而不是被无条件继续信任。
- 能力模型是 `logit(score) + 难度 ~ 1 + 特征` 的 ridge 加权最小二乘。样本不足
  （有效权重 < 9）时**回退到标量能力**，不硬拟合八维。当前配方 `max_rounds = 9`，
  但账本只在某轮判定后才多一行，单个类别通常仍停在标量档——这是预期行为，
  标量档已经带难度校正，收益来自这里，不是来自八维拟合。
- 得分用测量到的收益做连续映射，不是 PASS 标志：难度被减掉，所以难而值钱的方向
  仍然能超过容易的方向；只看 PASS 会把路由器锁死在微调上。
- 每轮 `search/rounds/<NNNN>/routing.json` 保存当时的输入、各臂得分、是否探索
  以及使用标量还是向量估计，事后可回答"当时为什么选这个方向"。
- 探索默认关闭（`info_floor = 0`）。打开后它也只是一次**请求**：控制器不能改写
  补丁，只能在下轮 request 的 `routing.preferred_category` 里点名一个在当前
  `foundation` 下没有证据的类别，是否采纳由模型结合证据决定。
- 搜索报告记录当前估计档位与账本行数，并明确这些分数只用于排序提案，
  **不能替代双 PASS 晋升**。

这组设置冻结在配方里，且进入 `fingerprint()`：改动权重会让跨 campaign 的经验
不再被复用。未声明这些字段的旧 campaign 继续用原规则，请求与证据场格式保持不变。

## 恢复和经验

```bash
python3 -m afd_rsi resume /path/campaign
```

campaign 锁阻止并发控制器。先按 host/boot/tag 核对拥有的进程，通过 pidfd 清理，
写独立恢复回执；已完成的原始回执不改写。Ascend 失败/中断后先执行冻结的
health 命令，失败则停止调度。该检查仅证明基础计算可用，不自动修复设备或
保证故障通信组已恢复。完成的阶段复核哈希后复用，未提交完成的 trial 保留并
换目录重跑；部分构建移到 abandoned-builds。原始 deadline 和已预扣调用不重置。

`search/state.json`、每轮 state/request/proposal、attempt progress、`best.json` 和
`search/experience/*.json` 分别保存控制状态、模型输入输出、测试进度和知识证据。
best 指向实际被测源码及两次 PASS 证据，不指向可随手修改的工作目录。
`export --best` 导出该版本相对于最初源码的累积补丁，即使最近一轮被拒绝也不变。

历史经验通过 `[search].experience_dirs` 指向已有 campaign 的 search/experience。
仅检索源码起点、底座/控制器、workload、原生拓扑、服务配置、门槛一致的记录，
并复核原始证据哈希。同一 parent 的重复补丁直接跳过；类别成功次数减失败次数
实际参与下一轮排序，并将历史决策与错误日志提供给模型。经验目录作为只读来源，
新记录写入当前 campaign，技能/知识不会被自动修改。

修改框架或底座后应创建新 campaign。要恢复旧 campaign，使用其匹配的 checkout，
也可从已冻结控制器运行：

```bash
PYTHONPATH="/path/campaign/foundation/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m afd_rsi resume /path/campaign
```

限制：目前是单 workload、单主指标、串行候选搜索；上下文有固定大小，省略的
源码文件会列在请求中。类别评分用于探索排序，不是独立策略质量评测。可信
benchmark/同权限模型进程不构成安全沙箱；哈希用于检测漂移。设备
重置、分布式调度和自动修改底座均不在范围内。

新 Ascend 配方默认启用 `validation="operator-first"`：初筛只测算子，独立复测重复原生
门槛后才做最终服务和 profile。旧 full 配方保持原流程，详见 [分层验证](staged-validation.md)。
