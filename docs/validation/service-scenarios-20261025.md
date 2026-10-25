# 专项服务指标验证：2026-10-25

本轮只验证新增指标的控制器、工作负载协议和数学归约，没有启动真实模型服务或 NPU
优化 campaign，没有修改 afd-plugin。已有任务和服务未停止、重启或被接管。

- 完整本地回归：151 项通过、2 项跳过，305.33 秒。跳过涉及本机套接字权限和缺少 httpx。
- 其后将场景重算器显式绑定到冻结基础，最终定向回归：52 项通过、1 项缺少 httpx 跳过，104.53 秒。
- 最终 iTASK CPU 协议回归：26 项通过、无跳过，1.18 秒，包含真实 localhost HTTP/SSE/Prometheus 测试替身。
- 已只读检查任务中安装的 vLLM：固定 DP 请求头、completion token ID 和 stream interval 参数均存在。
- 专项配方实际生成 9 个矩阵点和 2 个场景，Attention DP3/TP4、FFN EP4、DBO 仅在 Attention。
- 基础闭包、服务源码 provenance 哈希及 diff 空白检查通过；原有手工维护的 expert-packing 知识保留。

测试覆盖配置与路由数量、实际计数器核对、预热隔离、共同窗口与整请求 TPOT 区分、
缺请求/错误 DP/抢占/合并事件/短窗口/非有限数拒绝、原始文件与汇总一致性、比较报告、
同一 client 连续跑两个场景、失败保留、冻结校验器选择和 Agent goal 传入。

原始传输身份和两轮远端日志保存在工作区 `results/afd-rsi-scenario-validation-20261025/`。
最终快照与路径见 [机器可读记录](service-scenarios-20261025.json)。测试替身不代表真实
模型性能；当前 expert-packed 算子内部打点兼容性和实际 32K decode KV 容量尚需另行 NPU 验证。
模型性能指标目前为辅助对照，原生事务仍是晋升主指标，本轮没有建立模型收益结论。
