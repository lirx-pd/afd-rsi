# GSM8K、小 prefill 与 profiling 验证

本轮完善框架能力，不以算子性能收益验收。Ascend eager/graph 样例配方现固定
smoke、30 条 GSM8K strict-match，以及一个 128 输入 token、1 输出 token、并发 1
的 prefill 点，2 次 warmup、8 次正式请求。精度门槛沿用 E2E 默认 0.27；
小样本用于功能验证，正式门槛应在 prepare 前固定。

服务观测归档实际 GSM8K 样本数、分数、threshold 和 prefill TTFT mean/p50/p90。
比较报告提供辅助 TTFT 对比，晋升仍要求原生主指标和独立确认通过。
HTTP TTFT 包含调度、模型、通信、客户端开销，不是单独的 FFN kernel 延迟。

`native.profile=true` 在基线通过门槛后、以及每个合格候选通过门槛后单独采集
profiling。使用既有 torch_npu profiler，不新增平台依赖。每个 FFN rank 必须提供
包含实际 MegaMoe 设备任务的 CSV；原始 trace、加载身份、回执和哈希均保留。
控制器汇总设备任务次数、总/平均时长和 top tasks 占比；Agent 收到当前最佳版本、
近期候选的 profile 分析和仓库内性能策略知识，结合源码提出可证伪的优化假设。
profile 时长不参与普通计时，失败/中断使用现有健康屏障，续跑复核后复用。

## 已确认的真实基线

- 目标为 `afd-plugin` `beea4aa1cfe53a1916f6bf1ed52d57b28609596f`，输入内容身份
  `1d290c11d286f7b81273d37a89a0a1bfd17fef0061dc46a2cd02bafdbcded179`。
- 在现有 A3 16 设备环境重新隔离构建；使用真实 V4 Flash W4A8、DP2TP4/EP8，
  Attention eager、FFN FULL。原生 fixture 数值、捕获/replay、实际库身份检查通过。
- 初始基线 attempt：`66b2d50d80444dfca91b25787133af48`。
- smoke 10/10；GSM8K 官方 strict-match **27/30 = 0.90**，实际样本数 30，
  保留官方结果和逐条输出；numeric-equivalent 诊断不用于提升 strict-match。
- prefill **8/8**，TTFT mean **112.346 ms**、p50 **112.931 ms**、p90 **115.002 ms**。
  这些数字仅证明指标链路，不声明正式性能。
- 两个 FFN rank 的设备 trace 验证通过，分别有 8、7 个 MegaMoe task；差异来自
  fixture 内刻意构造的稀疏/分块工作，不能据此判定负载失衡。
- 两轮普通 native benchmark 与 profiling 分进程执行；请求包含 `profile-incumbent`。
- Codex 真实提案同时引用 `baseline`、`profile-incumbent`，以减少空行 AIV 的输出
  初始化为假设，明确不从融合任务时长推断 Cube/通信 bound，也不把 HTTP TTFT
  当作原生 kernel 时长。补丁仅提交到隔离候选工作区。

关闭错误仍按用户要求暂不考察。原始服务 summary 的 status/shutdown_status 为
failed，错误均以 `shutdown:` 分类；inference、smoke/GSM8K/perf、实际加载身份和
cleanup 必须通过。未改写原始 failed 生命周期记录。

## 回归与证据边界

最终版本 WSL 回归 **90 passed、1 skipped**；跳过项为沙箱内无法绑定本地 HTTP
socket。NPU 容器内同一组 **91 passed、无跳过**。`afd_rsi check` 和资源闭包校验通过。
新增测试覆盖精确 GSM8K 数量/低分拒绝、prefill 工作量/请求完整性、设备时长与 rank
缺失、profile 触发顺序、证据漂移、上下文注入、辅助指标策略及异常必须保持 invalid。

长实验使用自身不可变快照。启动后补充的三处失败保护为：分析失败必须保持 invalid；
每个指定 FFN rank 自身的 CSV 必须包含 MegaMoe。服务未开始测试时保留真实启动错误，不读取缺失指标。最终控制器已重新核验真实基线的
源码/产物、回执、服务指标和 trace，并重建出完全一致的 Agent request，未改写旧实验。

当前只提供原生合成 workload 的 task 级 profiling 分析；尚未提供真实模型服务内部的
Cube/Vector/MTE/通信细粒度归因。融合任务总时长不等于关键路径或 PMU busy time。
本轮真实模型验证为 FULL；eager 配方/入口兼容性由回归覆盖。

自动比较 attempt `d29d918d506c46c39567ad3006be36eb` 在 baseline 服务启动时遇到
vLLM 内部端口 35989 的 EADDRINUSE，测试尚未开始。按规则停止、判为 INCONCLUSIVE，
搜索记录 rejected，健康检查通过；没有触发后续 candidate profiling 或晋升。
错误不是 shutdown-only，不豁免；占用进程未捕获，清理后端口已释放。

本轮候选源码身份为
`6e1254a396939c1f33ee42ac036469328da92a4d4bb451d91dde39b7cdb21f5c`。
它把 guard 放在 `pipe.Reset()` 之后。此前候选 `b1545c59...` 的 guard 在 Reset 前，
内容身份检查拒绝复用该构建。本轮改为从当前补丁重新编译，单独补充服务和原生
验证；补充结果不替换失败的自动判定，不作为优化晋升证据。

## 补充验证结果

当前候选已从精确源码重新构建，随后使用最终版框架通过真实服务验收：

| 项目 | 基线 | 本轮候选 |
| --- | --- | --- |
| GSM8K strict-match | 27/30 | 28/30 |
| prefill 测量请求 | 8/8 | 8/8 |
| TTFT mean / p50 / p90（ms） | 112.346 / 112.931 / 115.002 | 114.672 / 114.439 / 116.161 |
| 映射库身份验证的进程 | 16 | 16 |
| inference / cleanup | passed / passed | passed / passed |

两侧使用相同数据集文件和模型配置哈希，逐条 prefill 记录确认实际 128 输入、1 输出
以及完整 8 条 measured 请求。这个小样本不用于宣布精度或性能提升。

使用两侧真实构建运行独立原生 FULL 对比：layered 参考、两侧正确性、两组 AB/BA、
基线 profile、候选 profile 共 9 个阶段均完成；两个版本分别验证每个 FFN rank 的
真实 MegaMoe CSV，共 4 个 FFN profiler 进程。最终控制器从原始 CSV 重新计算分析，
与已归档结果完全一致。初始自动基线另有两个 FFN profiler 进程。

原自动 campaign 仍保持 INCONCLUSIVE/rejected，不把独立补测改写为自动晋升。
best 仍为原始基线，`export --best` 的 change.patch 为 0 字节。本轮没有向 WSL `afd-plugin` 应用候选补丁；冻结远端输入仍未改变。结束时全部 NPU
无运行进程，现有任务保留。

已验证相同 parent 的补丁去重；生成未调用模型的下一轮上下文预览，包含基线
profile 和历史拒绝事实。失败服务未触发候选 profile，因此该预览没有伪造候选
profile-history。合格候选 profile-history 的上下文注入由专门回归断言覆盖。

## 证据身份

最终控制器内容身份：
`450ab5d4acb628b989958bc9c0ababb49c4ccd7c4e5922e29ab4c92f1b92ee69`。
最终框架部署包 SHA256：
`1ee7e965173c2db778f43c4fb4fae110b56296f771b301380e6af57594165b3e`。
真实原始证据包 69,088,673 字节，SHA256：
`f35b76eb5f9fc3c93ac8c33eb1504e0d125b7d7788249237b2a601e165ec4ebe`。

[结构化结果](metrics-profiling-20261023.json) 保留合同、服务指标、profile 分析、
自动判定与补充验证的分界。原始压缩包位于
`runs/metrics-profile-20261023-01/evidence.tar.gz`；它保留输入源码、精确原生产物、
官方 GSM8K 输出、HTTP 请求明细、设备 trace/CSV、Codex 输入输出和逐文件哈希清单。


最后检查发现 WSL `afd-plugin` 工作树在执行期间出现了其他 kernel/pipeline 修改及
性能文档，本轮没有写入或回退它们。该工作树内容身份为
`166b87ebed84f73c4da579d9aa04145864b480cb3e7d650fbf11e31d81ca5616`。
本次验收只覆盖冻结的 `beea4aa`/`1d290c11...` 及隔离候选 `6e1254a3...`，
不覆盖这些后续工作树修改。

证据包已回传本机并核对完整压缩包 SHA256；流式校验全部 **2744** 个文件的 SHA256
及可执行位一致。常用报告和模型/Agent 指标已提取到
`runs/metrics-profile-20261023-01/evidence/`，原始二进制和 trace 保存在压缩包中。
