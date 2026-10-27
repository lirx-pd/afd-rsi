# MegaMoe 原生路由性能目标

专用配方 `recipes/ascend_pipeline_scenarios.toml` 已纳入统一 `optimize` 流程，
核心仍先做算子数值/性能筛选与独立复测，通过后才启动最终服务。现有 eager/graph
配方默认测试行为不变；新配方使用 Attention DP3/TP4 eager + DBO、FFN EP4 FULL。
当前专用配方只保留以下三项，删除小 prefill 矩阵和之前的 16K 重载点。

| 场景 | 冻结负载 | 指标 |
| --- | --- | --- |
| native-long-decode | 6 并发，32768→512；6 个同上下文/32 输出预热 | 公共窗口 TPOT、输出 tokens/s、整请求 TPOT |
| mixed-prefill | 8 个 16384＋16 个 512 输入，输出 1，QPS3，3 个预热 | 总体与输入组 TTFT、输入 tokens/s |
| heavy-prefill | 128 个 65536→1 请求，QPS3，3 个预热；允许 128 在途请求 | 含排队的 TTFT、完成 req/s、输入 tokens/s |

沿用 2026-10-26 已完成实验的原生 DP 路由，不发送 affinity header。
六并发是客户端总并发；实际每 DP 请求分布来自计数器，不能把负载分组当成路由结果。
历史固定 DP 测试仍可显式选择，但不混入新配方，也不混用其性能基线。
已跑实验的配置与来源固化在仓库 knowledge/measurement/native-service-targets.json。
本框架沿用自身固定种子的 token prompt 生成器，不声称与历史文本 prompt 逐字相同。

共同窗口为“最晚首 token 后 1 秒”至“最早末 token 前 1 秒”。每条请求在窗口内
至少 128 个 token 事件；逐请求用窗口内首末事件间隔除以 token 间隔数，再汇总分布。
窗口不足、流式事件合并多个 token、缺失 DP 计数器或实际路由不符均判失败。
这些是 HTTP 客户端指标，不能当作算子时间或物理通算重叠证据。

一套服务依次完成 smoke、GSM8K30 和上述三个专项场景；服务不重复
加载。矩阵点和场景间沿用 `--perf-point-interval`。工作负载与阈值冻结在新 campaign，
`before-warmup.prom`、`before.prom`、`after.prom` 和 warmup/measured `requests.jsonl`
全部归档；报告、恢复审计会从原始文件重新计算，不信任单独填写的汇总。

## 启动

在已准备好的 WSL/提案机中，从仓库目录执行：

```bash
PYTHONPATH=src python3 -m afd_rsi optimize recipes/ascend_pipeline_scenarios.toml --executor itask --task afd-mrv2-async --workdir /a3_inference/itask/workdir/jcz02615514/lrx/afd-mrv2-async/rsi --setup-script /usr/local/Ascend/cann-9.1.0/set_env.sh --output runs/pipeline-scenarios-001
```

输出目录必须是新目录。开始前确认任务资源可用，配方源码、权重、设备、网络和
CANN 环境适用于现场。该命令启动整个框架，不是直接执行平台命令；本机用相同
入口及 `--executor local`。WSL 后台保留普通轮询进程，模型仅在提案时运行。

新配方最大上下文 65537、KV 显存利用率 0.92、单请求超时 1800 秒、整个服务阶段超时 3600 秒，基于已有 V4
场景配置；具体环境仍须具备足够 KV 容量。内存不足应停止并诊断，不能悄悄缩短
上下文、减少输出 token 或放宽门槛。原生 driver 仍为原来的小形状七 case 筛选，
新增的是最终模型性能场景，未将它伪装成长上下文原生 kernel benchmark。
expert-packed 内部打点使用已修复的诊断驱动；服务计时不启用 profiler。

## 可选参数与报告

服务入口增加：

- `--perf-scenarios native-long-decode,mixed-prefill,heavy-prefill`：任选组合，默认仍是 matrix。
- `--perf-dp-input-lens 16384,512,512`：每个 Attention DP 的 prefill 长度。
- `--perf-decode-context 32768 --perf-decode-tokens 512`：明确记录 decode 上下文及输出。

同一服务启动中的多个场景保存在 `performance.json` 的 `scenarios`；矩阵仍为
`points`。campaign observation 的 `service_metrics.scenarios`、比较报告的
`service_scenario_comparison` 和后续 Agent 请求包含这些指标。比较区分负载分组与实际 DP 计数器，不用整请求 TPOT 替代共同窗口 TPOT。

新配方 `search.goal` 明确将方向约束在 MegaMoe 内部 tile/专家打包、通信计算流水、
buffer 所有权和等待优化。它是提案目标，不能修改冻结验收。当前晋升主指标仍为
原生事务；新增服务性能为辅助比较及有效性门槛，不自动要求 TPOT 达到某个提升比例。
