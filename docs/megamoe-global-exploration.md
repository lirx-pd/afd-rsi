# MegaMoE 全局重构探索：2026-11-05

本次分析使用 `megamoe-codex-20261105-0247` 的基线源码和 FFN Profile。它是架构假设，尚无重构候选通过原生算子或模型服务测试。

## 关键路径

两个 FFN rank 的 `sync_in` 分别约为 8790 万和 8523 万 core cycles；`round_prepare` 分别约为 1256 万和 1218 万，其中 `rx_select` 占 `round_prepare` 的 99.97%。GMM1、激活、GMM2 和输出各为数万 core cycles 量级。这些是不同核心上的诊断区间，不能相加为事务延迟，也不能直接换算为模型加速。

`afd_pull_protocol.h::Select` 在所有 Attention TP 源的 `COMMIT` 到达后才选择事务。这段等待用于隔离 FFN 执行，不应作为 FFN 待优化热点。旧 Profile 中 `rx_select` 还包含少量头校验与选择工作；应先剔除等待，再比较 Pull、GMM、激活和输出等 FFN 内部阶段。

## 两种整体方案

1. **重构 FFN 分段流水。** 输入提交后，按专家或有界块组织 Pull、GMM1、激活、GMM2 和输出回传，调整缓冲所有权和阶段同步，争取重叠 FFN 内部工作。需要证明量化数值、窗口所有权和图重放正确。
2. **重做 FFN 计算与数据搬运调度。** 保留当前事务协议，围绕有效专家、tile 形状、缓冲复用及输出 DMA 重新安排计算。以去除源等待后的 Profile 确定值得优化的阶段。

新 Profile 会记录源就绪边界，并从 FFN 阶段占比中排除就绪前等待和入口同步。旧 Profile 缺少该边界时，保守排除 `rx_select` 中除已测头校验外的部分；它可能同时排除少量 FFN 选择工作。各指标均是 core cycles 总和，不能当作事务墙钟时间。

候选仍需先过原生数值和配对算子事务，再用少量有代表性的模型服务场景验证 GSM8K、prefill TTFT 与吞吐。任何大幅改写都不因源码规模或 Profile 区间缩短而直接判定胜出。
