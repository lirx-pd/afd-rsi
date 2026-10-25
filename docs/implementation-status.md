# 实施状态：2026-10-25

当前已实现可运行的实验闭环和受预算控制的 Agent 搜索：构建、验证、比较、服务
验收、报告、最佳候选、续跑和经验检索。支持独立冻结的 eager/FULL graph 配方；
自动优化按冻结主指标和完整独立复测验收，生产模型性能仍需单独验证。
核心实验控制器保持独立 Linux 执行；统一 `optimize` 入口提供 local/iTASK 执行器，不使用 SSH。

已实现：

- 8 个仓库内技能、完整本地知识/benchmark 闭包、配置和控制器冻结、源码内容身份。
- 独立工作区、允许文件范围、隔离构建、真实 rank/加载库身份、共同 layered 参考、AB/BA。
- eager/FULL graph 分开冻结；FULL 保持 Attention eager、FFN 显式 FULL，原生
  驱动检查捕获/replay、稳定地址与禁用 eager 回退。默认 eager 接口保持兼容。
- 固定版本 AFD E2E 已成为 campaign 阶段，使用实际 side 构建启动标准 vllm serve，
  保存真实请求、库映射、生命周期结果和日志；分层筛选后在最终资格执行配置的服务门槛。
- Codex CLI 的只读结构化提案；支持单机直接调用，也支持提案机/执行机文件交换。
  本轮按授权复用 WSL 登录，所有实验和验收在 iTASK，认证文件不复制。
- “证据 → 假设/补丁 → 隔离构建 → 验证 → 比较 → 决策”；当前最佳候选单独记录，
  初筛 PASS 后必须在新 attempt 独立确认，失败/不完整结果不能替换 best。
- 模型调用次数、轮数、上下文和原始 wall-time 预算；调用在派发前持久化预扣。
- campaign 排他锁、host/boot/tag + pidfd 归属检查、退出进程组的后代清理、恢复回执。
  已完成阶段复核后复用，未完成 trial 保留重跑，Ascend 异常后健康检查再调度。
- 历史经验包含假设、事实、源码/补丁身份、判定与证据哈希；相同合同的跨 campaign
  记录可检索。重复补丁去重、类别历史评分和模型上下文均实际影响下一轮选择。
- `export --best` 导出最佳版本的累积补丁；原始工作树及底座不参与候选自动修改。
- Ascend 样例服务验收固定为 smoke、30 条 GSM8K strict-match 和 9 点 prefill 矩阵；
  GSM8K 实际样本数/门槛校验，prefill HTTP TTFT mean/p50/p90 归档并进入比较和 Agent
  上下文。prefill 为辅助指标，主指标/独立确认规则保持冻结。
- 基线一次采集 profiling；候选在独立原生复测和最终服务门槛通过后采集；每个 FFN rank 的真实
  device trace、CSV、进程回执和身份均留档。按任务时长汇总 top kernels，复用/续跑
  时校验证据，异常触发现有健康屏障，profile 时长不混入普通 benchmark。
- 请求加入最佳/最近候选的 profile 分析以及仓库内性能策略 skill，驱动下一轮可证伪
  的源码优化假设。融合任务内的 Cube/MTE/通信归因仍需要更细证据，不能从总时长推断。
- 独立诊断副本采集 MegaMoe 72 核逐 tile 阶段、依赖等待和发布区间，检查数值、
  记录上限与 scratch guard；生产服务和 benchmark 继续使用未打点产物。
- 上下文优先保留源码依赖、测量事实和仓库知识，再加入日志；按实际 JSON 大小限额。
  当前 profile 包含前四个 tile，历史记录保留总量与原始归档；priority=3 为优先。


本轮关闭错误按用户要求暂不考察。样例 recipe 固定
`service.require_clean_shutdown=false`，只豁免 E2E 明确归类为 shutdown 的错误；
原始 failed summary 不改写，报告记录豁免。推理、全部测试、实际加载身份和清理
仍须通过。默认策略仍为严格关闭，Agent 不能在搜索中更改策略。

此前 eager 交付的 iTASK 完整回归 **69 项通过、无跳过**；真实 Codex 两轮提案、
初筛、独立确认、晋升和预算停止通过。按冻结关闭豁免策略，真实 NPU 基线的
构建、数值、服务和两轮 benchmark 均通过，下一轮提案请求已生成。

实际验证详情及证据身份见 [RSI 验证记录](validation/rsi-20261023.md)。
此前原生小/宽形状与 profiler 记录保留在 [eager 验证](validation/eager-20261023.md)。
FULL graph 接入后的 iTASK 完整回归 **73 项通过、无跳过**；原生 campaign
correctness/benchmark 与真实 V4 服务 gate 已通过。当时 CANN 编译错误导致该次
实测使用此前已构建基线；历史详情见 [FULL graph 验证](validation/graph-20261023.md)。
该编译错误现已在 `beea4aa` 修复，本轮真实候选检验已重新构建当前版本并通过。
当前使用方法见 [自动搜索与续跑](automatic-search.md)。

边界：

- 合成 fixture 和合成权重原生事务不等于真实模型的正式精度/性能验收。
- 已开展受控 MegaMoe 自动候选迭代；冻结原生事务收益不代表生产模型收益。
- 未验证 NPU 硬故障后的自动恢复或设备复位；健康检查只证明基础计算可用。
- 经验不会自动改写冻结技能/知识；独立策略质量评测、多 workload 搜索尚未实现。
- 已有旧 campaign 使用各自实际快照；升级底座后创建新 campaign，不改写旧判定。

真实候选单轮检验已完成：以 `beea4aa` 为基线，由 Codex 提出一行 OutputTile
空行 AIV 优化，构建、FULL 数值、真实 V4 服务和四组 AB/BA 均实际执行。
观测降幅 1.50% 未达到冻结 2% 门槛，按规则拒绝、保留经验，真实仓库未改；
best 导出为空、真实经验去重验证与 73 项回归通过。详见
[真实候选检验](validation/real-candidate-20261023.md)。

GSM8K/prefill/profiling 增强已完成：最终容器回归 91 项通过，真实 FULL 基线
GSM8K 27/30、候选补测 28/30，小 prefill 均 8/8，原生两侧数值与 profile 通过。
一次自动比较因内部端口 EADDRINUSE 判为不完整并拒绝，补测不改写其判定，未晋升；
本轮未应用优化补丁，期间出现的外部源码修改不在本次验收范围。详情见 [指标与 profiling 验证](validation/metrics-profiling-20261023.md)。


2026-10-24 多轮自动优化已完成六个真实候选验证；第六个输入提前回填候选
初筛增益 3.671313%、完整独立确认 2.521912%，均为 PASS，已由 best 导出并
保留唯一 pipeline 头文件修改。框架回归在 NPU Linux 全部 103 项通过。
细粒度打点支持等待缩短，但没有建立模型端到端提速或连续 Cube 活动结论。
完整数据、稀疏 case 边界和框架修复见 [多轮优化验证](validation/megamoe-iterations-20261024.md)。

2026-10-25 统一后台入口已完成：一条命令冻结、部署并执行整个优化流程，普通 worker
定时轮询；有限 Codex 提案结束后才启动设备验证。支持状态、显式恢复、结果下载校验；
默认服务单次启动覆盖精度与性能矩阵，优化策略由前期 deep 转为后期 incremental。
本轮仅验证入口与异步闭环，未启动新的真实 NPU campaign。详见
[后台入口验证](validation/orchestration-20261025.md)。

2026-10-25 增加 operator-first 分层验收：初筛和初始基线不启动 E2E，独立原生复测
达标后才执行最终 E2E 与候选 profile。提案需具备瓶颈、机制、预测、否定条件和风险，
引用当前测量及 profile；完整实验假设进入历史。见 [分层验证](staged-validation.md)。

分层验证完整回归 126 项通过、1 项本机套接字权限跳过；提案定向检查 7 项通过。
本轮为 CPU 控制器验证，未新增 NPU 收益或运行耗时结论，见
[分层验证回归](validation/staged-validation-20261025.md)。
