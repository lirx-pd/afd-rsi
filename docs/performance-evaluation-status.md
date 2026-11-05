# 性能评估现状与使用边界（2026-11-04）

本次对照 `/mnt/d/lrx/RSI/afd-plugin` 的 `design/afd-megamoe-pull-dispatch`
版本及 `/mnt/d/lrx/megamoe/mega-moe-bench/mega_moe_performance.py` 的测量边界。
下面是入口和合同检查，不是新的 NPU 性能结果。

| 层级 | 当前入口与可判读内容 | 本次修正 | 仍需的设备证据 |
| --- | --- | --- | --- |
| 算子事务 | `benchmarks/native/eager_fixture.py`：合成权重、序列化 A 源，Attention 同时钟域 DS→FFN→CR 事务 | 对齐现版 `LayeredFFNWeights` 接口；保存每个 case 样本、配对中位增益，冻结 5% case 回退门槛 | 真实 H/I、专家数和路由下的 FFN kernel、等待、完整事务分别计时；同输入 AB/BA 与无打点复测 |
| E2E | `benchmarks/service/afd_e2e/`：真实模型 HTTP/SSE、GSM8K、服务生命周期和原始计数器 | Pull 接收容量随批量预算固定；模型 top-k 可读时预检 uint16 源行限制；冻结 5% 场景回退筛选 | 用当前镜像、模型、16 卡配置跑基线和候选；核对实际加载库、GSM8K、每个服务场景、设备错误及清理 |

`ascend_eager.toml`、`ascend_graph.toml` 和 `ascend_pipeline_scenarios.toml`
仍以 H256/I512、E_local4、2F 的合成事务作自动搜索主指标。它能筛掉数值错误和
明显的事务回退，不能证明 DeepSeek V4 模型端收益。配方里的 32764 批量上限
按 TP4、top-k8 的 Pull 路由边界留出一行余量；实际 checkpoint 的 top-k 由
服务计划读取 `config.json` 再检查。64K 请求使用 chunked prefill，不能作为
单次 64K Pull 算子样本。
需要观察较长单次事务时，可在新配方冻结 `native.lengths`；驱动和证据验证器
现在按同一长度列表生成/核验 case，不再把非默认长度误判为 7 个短算例。

每次探索方向保留：源码与原生库哈希、有效配置、输入/路由、warmup、所有算例
样本、AB/BA trial 顺序、逐 case 回退、数值结果、服务原始请求和计数器。总均值
达标但 case 回退超过 5% 为 FAIL；单轮 E2E 场景回退超过 5% 为 INCONCLUSIVE，
需要独立模型复测。服务 TTFT/TPOT/吞吐是 HTTP 端指标，不能推断 FFN kernel
本身加速或流水重叠。参考 megamoe 的 ready-input 与 cycle 区分，诊断 profile
与无打点计时分开，且不把单次 FFN kernel 时间当作完整 A→F→A 延迟。
原生首次采样过散会做预定的加密复测；复测仍过散则保留两轮原始数据并停止判定，
不从两轮中挑较平稳的一轮作为收益。

工作区原有 `raw_mega_perf.py` 仍是草稿，已保留在 `docs/prototypes/`，
未接入评估入口：
它未执行所宣称的 CPU 数值参考，且当前 op 需要逐 layer Tensor 列表。
不要把其输出纳入 campaign 判定；正式 raw FFN 对比需要先完成它的
API、数值和多 rank 完成性校验，并在 NPU 上验证。
