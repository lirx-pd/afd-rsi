# 本机真实服务验证

本仓库的服务测试执行代码位于 `benchmarks/service/`，是固定版本的 AFD E2E
工具本机实现，代码和许可证均已纳入底座清单。不需要另一个工具 checkout。

服务验证使用已准备的本机 NPU 推理环境：CANN、torch-npu、vLLM、vLLM-Ascend、
待测 afd-plugin、对应算子以及权重。测试本身不安装或升级推理依赖。
使用显式设备编号或预先分配的可见设备，不申请设备、不调度远端任务。

执行链为：生成并核对 plan -> 预检设备/环境/模型 -> 标准 vllm serve 启动
Attention 与 FFN -> 同时监控服务存活和 HTTP 就绪 -> smoke 实际请求 ->
指定的 GSM8K/性能请求 -> 保存原始结果 -> 有界清理本次进程与后代。

必须提供待测 --repo；它决定目标 checkout 的路径注入和 Git 元数据。工具另行
记录目标 Python 已安装包版本。二者仍不是实际 native 库和 MegaMoe 内部调用的
证明；算子身份需要额外的真实加载路径/哈希与 profiler 证据。

MRV2、Async CAM、Layered、eager/graph 的参数从实际命令和环境传入。服务必须
真实接受配置并完成请求，plan 成功仅说明方案可以生成。不得用同步 V2 默认
配置替代当前 W4A8 MegaMoe 的 V4 异步 MRV2 场景。

当前目标的 AFDAsyncExtraInfo 只接受 dynamicQuant、attn_ranks_per_dp、
hccl_buffer_size、ffn_two_stream、ffn_compute_aic_num、ffn_compute_aiv_num、
ffn_combine_aiv_num。历史 async_moe_ubatching 已删除；即使值为 false 也会
导致 worker 初始化失败。本地工具不再传该字段，默认关闭 splitting，对显式 on
在 plan 阶段报错。不要通过修改目标配置校验器绕过接口不匹配。

V4 async 默认上下文为 32768，max-batched-tokens 为 32768；更大上下文需要
重新核对当前 KV 容量。原始工具版本和本地适配后的文件哈希均保留在 provenance。

V4 工具模板为 dp2tp4-ep8、16 张本地设备。模型路径必须是已准备好的 V4 权重，
不可假造小模型或随机权重然后宣称 V4 服务通过。先用 smoke 核对服务，再用相同
配置的独立运行做性能和质量验收。GSM8K 使用固定 lm_eval 版本和数据条件。

原始证据包含 plan.json、resolved-config.json、commands.json、manifest.json、
preflight/服务/工作负载日志、summary.json 及性能/质量明细。检查 inference、
shutdown、cleanup 和各 test_results，不以单个健康响应或返回码替代这些结果。
正常退出、强制终止和清理失败必须保留区别。campaign 可以在 prepare 前固定
require_clean_shutdown=false，暂不以关闭阶段错误验收；仅允许 E2E 已归类为
shutdown 的错误，原始失败仍保存，推理、各测试、加载身份和清理必须通过。

该入口用于模型/服务集成与服务性能验证。不能将 HTTP 性能结果转换为每 rank
kernel 时间，也不能给通用 compare 伪造 rank/source/artifact 字段。算子级原生计时
和加载身份使用独立的 [eager native 合同](../measurement/eager-native.md)；CPU 子进程
模拟服务的测试也不是 NPU 实测。
