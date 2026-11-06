# MegaMoE 全局重构探索：2026-11-05

本次分析使用 `megamoe-codex-20261105-0247` 的基线源码和 FFN Profile。它是架构假设，尚无重构候选通过原生算子或模型服务测试。

## 关键路径

两个 FFN rank 的 `sync_in` 分别约为 8790 万和 8523 万 core cycles；`round_prepare` 分别约为 1256 万和 1218 万，其中 `rx_select` 占 `round_prepare` 的 99.97%。GMM1、激活、GMM2 和输出各为数万 core cycles 量级。这些是不同核心上的诊断区间，不能相加为事务延迟，也不能直接换算为模型加速。

`afd_pull_protocol.h::Select` 在所有 Attention TP 源的 `COMMIT` 到达后才选择事务；`afd_mega_moe_comm.cpp::Publish` 负责统计路由、量化并写入 payload。现有 FFN Profile 没有给出 Attention 侧从输入就绪到 `COMMIT` 的分段时间，因此 `rx_select` 主要说明 FFN 在等源，不能证明轮询或 FFN 计算本身是瓶颈。此前仅调整 FFN 局部参数，难以触及这段等待。

## 两种整体方案

1. **分段发布和消费。** Producer 完成路由计数后，按专家或有界块写入 payload，并在每段数据可见后发布独立序号；FFN 依据序号分段 Pull、GMM1、激活、GMM2、回传，同时保持原有事务 `COMMIT`、窗口所有权和最终完成语义。这可能把 Producer 后半段与 FFN 前半段重叠，但需要一起重写 Publish、协议、Pull 调度和输出回传。必须证明 TP 源一致、量化数值不变、分段无重复/遗漏、图重放不读取未提交数据。
2. **保留整事务提交，重做 Producer 数据路径。** 在 Publish 中对路由统计、量化和 payload 写入做更完整的流水与缓冲调度，维持当前 `COMMIT` 协议。改动面较小，但若等待主要来自 Attention 上游而不是 Publish，本方案不会带来模型收益。

优先获取同一事务的 Attention 输入就绪、Publish 开始/结束、`COMMIT`、FFN `rx_select` 退出和输出完成时间线；用同一 source、sequence、layer 对齐。若 Publish 占等待的主要部分，再把方案 1 作为一次完整候选；否则先定位 Attention 上游，避免为了消除空转表象重写协议。

候选仍需先过原生数值和配对算子事务，再用少量有代表性的模型服务场景验证 GSM8K、prefill TTFT 与吞吐。任何大幅改写都不因源码规模或 Profile 区间缩短而直接判定胜出。
