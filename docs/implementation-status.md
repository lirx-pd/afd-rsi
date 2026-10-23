# 实施状态：2026-10-23

当前已实现可运行的实验闭环和受预算控制的 Agent 搜索：构建、验证、比较、服务
验收、报告、最佳候选、续跑和经验检索。支持独立冻结的 eager/FULL graph 配方，不以本轮算子性能收益验收。
框架不耦合 iTASK 或 SSH；实际部署/执行由外部工具完成。

已实现：

- 8 个仓库内技能、完整本地知识/benchmark 闭包、配置和控制器冻结、源码内容身份。
- 独立工作区、允许文件范围、隔离构建、真实 rank/加载库身份、共同 layered 参考、AB/BA。
- eager/FULL graph 分开冻结；FULL 保持 Attention eager、FFN 显式 FULL，原生
  驱动检查捕获/replay、稳定地址与禁用 eager 回退。默认 eager 接口保持兼容。
- 固定版本 AFD E2E 已成为 campaign 阶段，使用实际 side 构建启动标准 vllm serve，
  保存真实请求、库映射、生命周期结果和日志；每次比较、复测均执行配置的服务门槛。
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
correctness/benchmark 与真实 V4 服务 gate 已通过。当前源码另有独立的 CANN
编译错误，实测使用此前已构建基线；详情见 [FULL graph 验证](validation/graph-20261023.md)。
当前使用方法见 [自动搜索与续跑](automatic-search.md)。

边界：

- 合成 fixture 和合成权重原生事务不等于真实模型的正式精度/性能验收。
- 本轮没有正式优化 MegaMoe，也不宣布生产收益。
- 未验证 NPU 硬故障后的自动恢复或设备复位；健康检查只证明基础计算可用。
- 经验不会自动改写冻结技能/知识；独立策略质量评测、多 workload 搜索尚未实现。
- 已有旧 campaign 使用各自实际快照；升级底座后创建新 campaign，不改写旧判定。
