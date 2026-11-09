# 性能评估现状与使用边界（2026-11-04）

本次对照 `/mnt/d/lrx/RSI/afd-plugin` 的 `design/afd-megamoe-pull-dispatch`
版本及 `/mnt/d/lrx/megamoe/mega-moe-bench/mega_moe_performance.py` 的测量边界。
下面是入口和合同检查，不是新的 NPU 性能结果。

| 层级 | 当前入口与可判读内容 | 本次修正 | 仍需的设备证据 |
| --- | --- | --- | --- |
| 算子事务 | `benchmarks/native/eager_fixture.py`：合成权重、序列化 A 源，Attention 同时钟域 DS→FFN→CR 事务 | 对齐现版 `LayeredFFNWeights` 接口；保存每个 case 样本、配对中位增益；case 回退不再否决服务主指标 | 真实 H/I、专家数和路由下的 FFN kernel、等待、完整事务分别计时；同输入 AB/BA 与无打点复测 |
| E2E | `benchmarks/service/afd_e2e/`：真实模型 HTTP/SSE、GSM8K、服务生命周期和原始计数器 | Pull 接收容量随批量预算固定；模型 top-k 可读时预检 uint16 源行限制；冻结 5% 场景回退筛选 | 用当前镜像、模型、16 卡配置跑基线和候选；核对实际加载库、GSM8K、每个服务场景、设备错误及清理 |

`ascend_eager.toml`、`ascend_graph.toml` 现以真实模型 2048-token/并发 8 的
prefill TTFT p50 为搜索主指标；`ascend_pipeline_scenarios.toml` 对原生路由
六并发 32K→512 decode 的公共窗口 TPOT p50 与 16K/512 混合 prefill TTFT p50
取等权收益均值。初筛只运行 smoke 与这两项性能场景；独立复测才运行 GSM8K300
及 64K 重载 prefill。每个候选在初筛和独立复测中各做一次基线/候选服务对照，
顺序为 AB/BA。pipeline 配方的算子筛选改为 H4096/I2048、64 本地专家，
长度 7/128/511/512/513/1025；均衡路由下每个 FFN 精确接收相应行数，
另测集中路由和空 FFN。容量覆盖集中路由的完整 top-k 行数，检验多 cohort 及边界尾部。
这与当前 checkpoint 的 H/I 和 EP4 每卡专家数对应；轻量用例仍是两 FFN、top-k2，
真实四 FFN、top-k6 和并发 Attention 由服务门禁覆盖。
H256/I512 仍可手动用于快速数值诊断。合成事务收益不决定晋升。配方里的 32764 批量上限
按 TP4、top-k8 的 Pull 路由边界留出一行余量；实际 checkpoint 的 top-k 由
服务计划读取 `config.json` 再检查。64K 请求使用 chunked prefill，不能作为
单次 64K Pull 算子样本。

构建阶段保存 CMake 实际解析的 MC2/Catlass 依赖和生成 activation，关联源码/产物
哈希并核验缓存的头文件。提案分文件读取这些只读契约，allowed_files 内的协议与
布局源码不再因 3/4 预算被丢弃。缺失契约记录为 `context_required`，分析阶段即停止
代码生成。activation 诊断细分 GMM1 等待、外层同步、处理主体与收尾，并比较主体
同 GMM1/GMM2 的重叠；主体可能包含 SDK 内部同步，仍不是纯计算利用率。
需要观察较长单次事务时，可在新配方冻结 `native.lengths`；驱动和证据验证器
现在按同一长度列表生成/核验 case，不再把非默认长度误判为 7 个短算例。

每次探索方向保留：源码与原生库哈希、有效配置、输入/路由、warmup、所有算例
样本、AB/BA trial 顺序、逐 case 回退、数值结果、服务原始请求和计数器。合成
case 回退仍报告，但不再作为真实模型收益的拒绝条件；初筛只对主场景应用 5% 回退
筛查，完整复测对所有配置场景应用，超限为 INCONCLUSIVE，不能晋升。服务
TTFT/TPOT/吞吐是 HTTP 端指标，不能推断 FFN kernel
本身加速或流水重叠。参考 megamoe 的 ready-input 与 cycle 区分，诊断 profile
与无打点计时分开，且不把单次 FFN kernel 时间当作完整 A→F→A 延迟。
原生主指标配方的首次采样过散会做预定的加密复测；复测仍过散则保留两轮原始数据
并停止判定，不从两轮中挑较平稳的一轮作为收益。服务主指标配方仅归档小形状
原生计时离散度，不因其噪声增加测量或阻断真实模型评估。

工作区原有 `raw_mega_perf.py` 仍是草稿，已保留在 `docs/prototypes/`，
未接入评估入口：
它未执行所宣称的 CPU 数值参考，且当前 op 需要逐 layer Tensor 列表。
不要把其输出纳入 campaign 判定；正式 raw FFN 对比需要先完成它的
API、数值和多 rank 完成性校验，并在 NPU 上验证。
