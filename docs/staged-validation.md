# 先算子、后服务的分层验证

`recipes/ascend_eager.toml` / `ascend_graph.toml` 已默认设置
`validation = "operator-first"`。统一 `optimize` 命令不变。未设置该字段或显式
设为 `full` 的配方沿用原顺序；旧 campaign 使用各自冻结的控制器，不能原地换策略。

| 阶段 | 执行内容 | 不通过时 |
| --- | --- | --- |
| 首次基线 | 原生构建、算子数值、计时、一次内部 profiling | 停止，不调用模型 |
| 方向审查 | 引用当前测量/profile，给出瓶颈、机制、预测、否定条件和风险 | 不派发候选验证 |
| 候选初筛 | 独立构建、算子数值、AB/BA 计时 | 拒绝，不启动 E2E 或完整 profiling |
| 独立复测 | 新 attempt 重跑原生构建、数值及 AB/BA，对照同一 incumbent | 拒绝，不启动 E2E 或完整 profiling |
| 最终资格 | 复测达标后，基线/候选分别一次启动服务，跑 smoke、GSM8K30、九点 prefill；最后采候选 profile | 拒绝，保留原始失败 |
| 晋升 | 两次原生 PASS，最终服务/profile 也全部通过 | 更新 best、导出补丁 |

这让数值不对、没有收益或收益无法重复的候选不再消耗服务启动、模型推理和诊断
重建时间。服务验收仍是晋升的必要条件，不能由 Agent 以“低风险”为由跳过。
GSM8K30 的 V4=0.9、V2 Lite=0.27、收益阈值、数值容差和完整独立复测都不降低。
初始 best 只是经算子验证的起点，不代表通过模型精度；无候选胜出时不产生服务收益结论。

## 轻量算子测试

复用已有真实 native 驱动，不加载 checkpoint、不启动 vLLM HTTP 服务、不跑 GSM8K。
它通过真实 runner / connector / MegaMoe 进行多卡事务测试，默认 1 Attention + 2 FFN。
MegaMoe 通信测试不能降成无真实通信的单卡 mock。主要检查：

- 基线 layered 输出作为两侧共同参考，候选不能生成自己的“正确答案”。
- 多层切换、重复输入、稀疏/空专家、7/16/129 token 和 tile 边界。
- W4A8 的 layered 一致性；容差沿用 rtol=0.04、atol=2e-5，layered 重跑精确相等。
- eager/FULL 独立冻结，FULL 检查捕获、replay、稳定地址和禁用 eager 回退。
- 正式未打点产物、实际加载库身份、完整 rank、配对 AB/BA、原始逐 case 样本。

该指标是受控 DS→FFN→CR 原生事务耗时，含通信；不是融合 kernel 单任务时间，
也不是实际模型延迟。H256/I512 等小形状与合成权重加快筛选，但不能代替真实
模型形状、EP8/top-k6 与真实权重的最终验证。不得把小形状筛选结果宣称为生产收益。

手工候选可以只跑算子层：

```bash
PYTHONPATH=src python3 -m afd_rsi prepare recipes/ascend_graph.toml --output runs/operator-check
PYTHONPATH=src python3 -m afd_rsi patch runs/operator-check /path/change.patch
PYTHONPATH=src python3 -m afd_rsi compare runs/operator-check --operator-only
```

每一步等待执行完成；若需要一条后台优化命令，继续使用 `optimize`。这里的
`--operator-only` 不启动服务或完整 profiling，报告 `validation_level=operator`，
即使 PASS 也不能替代最终资格。`compare` 不加该标志为完整验证，operator-first
配方仍先做原生计时，达到原生阈值后才进入配置的服务/profile。

## 提高方向质量

首次基线的细粒度逐 tile 打点继续保留。提案必须解释测到的等待、计算、通信与
源码依赖之间的关系。`experiment` 字段要求：

- bottleneck：当前证据显示的瓶颈，不能只引用总 kernel 时长。
- mechanism：哪条源码/依赖改变会消除或隐藏这段开销。
- prediction：独立验证应出现的可观察变化。
- falsification：何种结果会否定假设，避免失败后重新解释成成功。
- risk：数值、buffer 生命周期、发布/消费与同步风险。

证据必须引用 baseline，以及提供时的 profile-incumbent；结构或引用不完整不能
进入构建。分析阶段不满足要求时，也不继续调用编码模型。对通算重叠的假设，需要
说明哪些 tile 的通信与计算可以并行、等待点和所有权如何保持；逻辑时段不能冒充
物理 Cube 利用率。没有证据支持就停止，不为用完轮数而猜参数。

经验保留完整 experiment、实际结果与原始哈希，并交给后续请求；失败类别继续
降低选择分数。结构检查不能证明假设正确，最终仍由真实独立测量判定。

## 恢复与审计

验证层级写进 attempt.lock.json，不能把 operator 结果改名为 full。独立复测进入
服务前保存 operator-gate.json；恢复时复核原生结果与该判定，完成的原生试验复用。
审计仅允许“原生门槛未通过”导致后续阶段缺失；原生 PASS 不能豁免缺失服务或
profile。中断、异常设备健康检查、原始 deadline、失败留存和模型额度规则均不变。

本次缩减的是无效方向的昂贵验收次数；没有引入跨候选构建缓存，也未削减数值覆盖
或配对计时次数。真实硬件上的耗时节省需要另行测量。
