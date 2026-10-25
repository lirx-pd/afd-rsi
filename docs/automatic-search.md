# 自动搜索、服务验收与续跑

框架在 Linux 本机运行，Agent 仅提出补丁。`search` 先资格验证基线，再执行
“证据 → 假设/补丁 → 隔离构建 → 正确性 → 配置的服务验收 → 按配方独立 profiling → AB/BA → 决策”。
通过初筛的候选还必须在新 attempt 完整复测后才能替换 best。两次比较均以同一
当前最佳版本为对照；报告不会把合成字段或 HTTP 指标当作算子计时。

## 单机 Codex

需要 Python 3.11+、Linux pidfd、已安装并登录的 Codex CLI。沿用 CLI 的 provider
配置和登录，不读取或复制认证文件；不需要额外模型 SDK。模型与推理强度显式固定：

- 实验操作：`gpt-6.1-sol` / `medium`。固定控制器直接执行实验，无需额外模型调用。
- 分析证据、profiling 和优化方向：`gpt-6-astra` / `xhigh`。
- 编写候选补丁：`gpt-6.1-sol` / `xhigh`。

这里将 ultra high 对应为 CLI 的 `xhigh`。提案先分析再写补丁；分析决定停止时不调用
编码模型。两步共享原有超时和 deadline，不静默降级；归档各自命令、输出、JSONL、
stderr 和带模型标签的 usage。`max_calls` 限制 Agent 命令派发次数，每次内含至多
两次模型调用。外部文件交换采用相同流程。模型规则固化在仓库
`knowledge/measurement/codex-models.md`。修改后的提案器用于新 campaign。
Ascend 样例给每次提案 1200 秒和 524288 字节上下文；两步模型调用共享超时。
priority=3 为优先、1 为备选，控制器按 priority 加类别晋升次数减失败次数取最大值。

使用仓库固定的 JSON
schema 和只读模式；补丁由控制器进行 git-apply 和允许文件检查。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m afd_rsi prepare recipes/cpu_search.toml --output /tmp/rsi-search-demo
python3 -m afd_rsi search /tmp/rsi-search-demo
python3 -m afd_rsi report /tmp/rsi-search-demo
python3 -m afd_rsi export /tmp/rsi-search-demo --best --output /tmp/rsi-best
```

该 recipe 的耗时是合成字段，只验证 Agent/控制器的合同。实际 NPU 使用
`recipes/ascend_eager.toml` 或 `recipes/ascend_graph.toml`：先按现场修改 source、已分配设备、模型路径、网络和
预算，再 prepare。其原生驱动确实运行目标的 build_ext 和 torchrun A/F
fixture，比较两个 MegaMoe 版本，使用共同 layered 数值参考，校验实际加载库和
设备对象身份；主指标是主 Attention rank 的七个 case 事务样本均值（us）。
FFN rank 没有伪造计时，仍必须给出完整数值/身份/清理检查。graph 配方额外检查
实际 FULL 捕获/replay 和禁止 eager 回退；参考输出仍来自 layered eager。

## 服务是 campaign 的固定阶段

`[service]` 不再只启动外部独立作业。控制器指向本 attempt/side 的隔离原生构建，
调用仓库固定版本 AFD E2E 的 plan/run：标准 vllm serve、HTTP 就绪、真实请求、
正常关闭和有界清理。服务输出、实际加载库映射、源码/产物身份及哈希归档在该
side/service/trial 中。启用服务时，两侧每次初筛和确认都必须通过该阶段。

服务默认要求完整生命周期通过。本轮按需求在样例中固定
`require_clean_shutdown = false`：只暂不考察 E2E 已归类为 shutdown 的错误，
原始 status/shutdown_status 仍保留失败，并记录 shutdown_errors_ignored。推理、
全部冻结测试、实际映射库身份和 cleanup 必须通过；报告/启动/推理错误不能
借此忽略。该策略只能在 prepare 前设置，Agent 无权修改。没有把服务结果伪装为
native 全 rank 指标。样例配置为真实 V4 W4A8、
16 设备 DP2TP4/EP8，eager/graph 按 recipe 分开冻结。默认执行 smoke、30 条 GSM8K
strict-match（V4 Flash 门槛 0.9，V2 Lite 默认门槛 0.27）以及一个小 prefill 点：128 输入 token、1 输出 token、并发 1，
2 次 warmup、8 次测量。离线数据集须已缓存，或在 prepare 前配置 `--dataset-path`。
GSM8K 样本数/分数和 HTTP TTFT mean/p50/p90 进入观察、比较报告及 Agent 上下文。
TTFT 是辅助性能指标，包含服务/网络开销；晋升仍使用冻结的原生主指标。
这组小样本用于跑通功能，不代表正式精度或性能资格。

## 自动 profiling 和瓶颈假设

样例 `[native].profile = true`：基线通过数值和服务门槛后，先单独采集 profiling，
再进行不带 profiler 的原生计时，最后生成首份 Agent 请求。比较时仅采集通过门槛
的候选；复测同样执行。失败的构建、数值或服务门槛不会启动 profiling。

复用已有 torch_npu profiler 子进程，保留每个 FFN rank 的原始 trace、
`kernel_details.csv`、加载身份与哈希；控制器汇总 top kernels 的次数、总/平均设备
任务时长及占比，不导入 torch。已完成 profile 可在续跑复核后复用，篡改或不完整
结果不能通过；异常使用现有健康检查屏障。profile 期间的耗时不进入主指标。

Agent 请求会包含当前最佳版本的 profile、最近候选的 profile 和历史判定，并加载
仓库内性能策略 skill。Agent 从设备耗时最大的任务出发，结合源码和 tiling 提出可
证伪的小补丁，后续仍走数值、GSM8K、prefill、独立计时和确认流程。MegaMoe 是
融合任务，仅凭任务总时长无法断言 Cube/MTE/通信 bound；原生合成 trace 也不能
单独解释模型 TTFT。具体证据边界已固化到仓库本地知识。

独立 profile 还会构建一份带算子内部打点的诊断副本，采集 72 个核的
GetSystemCycle 阶段、依赖等待和发布区间，并检查数值、记录上限和 scratch guard。
生产服务与普通计时继续使用未打点产物。当前 profile 摘要包含选定核前四个 tile，
覆盖双槽首次复用；历史 profile 保留阶段总量和原始归档，避免重复时间线占满
上下文。源码、测量事实和仓库知识优先于日志，预算按实际 JSON 文件大小计算。
每个核使用自己的时间原点，不能据此对齐跨核时钟或声称 Cube 物理利用率。

## Agent 和 NPU 不在同一台机器

可用统一 `optimize --executor itask` 入口自动完成以下交换与轮询，见
[统一后台入口](orchestration.md)。以下保留手工流程，便于检查或接入已有 campaign。

在执行机创建 campaign，并进入文件交换模式：

```bash
python3 -m afd_rsi search /path/campaign --external-agent
```

`campaign/search/pending.json` 给出 request、response 的相对路径、请求 SHA 和
已经预扣的调用序号。将其中 request.json 复制到已登录的提案机，在同版本仓库
执行（手工模式自行传输；统一入口可自动传输）：

```bash
python3 -m afd_rsi propose /path/request.json --output /path/new-proposer-output
```

将输出目录整个保留以归档 Codex JSONL、usage、stderr 和进程回执，将
proposal.json 放回 pending 指定的 response 路径，再在执行机运行：

```bash
python3 -m afd_rsi resume /path/campaign
```

它消费已提交的回复，应用补丁并验证，完成后可能产生下一份 pending。外部编排
可重复上述文件交换。没有回复时返回等待，不重复扣调用次数。`propose` 只读
请求内的源码/证据/本仓库知识，校验提案器身份、响应大小和 schema，并实施
请求中的超时；不会拿到执行机的构建或 NPU。认证文件始终留在提案机。

## 恢复和经验

```bash
python3 -m afd_rsi resume /path/campaign
```

campaign 锁阻止并发控制器。先按 host/boot/tag 核对拥有的进程，通过 pidfd 清理，
写独立恢复回执；已完成的原始回执不改写。Ascend 失败/中断后先执行冻结的
health 命令，失败则停止调度。该检查仅证明基础计算可用，不自动修复设备或
保证故障通信组已恢复。完成的阶段复核哈希后复用，未提交完成的 trial 保留并
换目录重跑；部分构建移到 abandoned-builds。原始 deadline 和已预扣调用不重置。

`search/state.json`、每轮 state/request/proposal、attempt progress、`best.json` 和
`search/experience/*.json` 分别保存控制状态、模型输入输出、测试进度和知识证据。
best 指向实际被测源码及两次 PASS 证据，不指向可随手修改的工作目录。
`export --best` 导出该版本相对于最初源码的累积补丁，即使最近一轮被拒绝也不变。

历史经验通过 `[search].experience_dirs` 指向已有 campaign 的 search/experience。
仅检索源码起点、底座/控制器、workload、原生拓扑、服务配置、门槛一致的记录，
并复核原始证据哈希。同一 parent 的重复补丁直接跳过；类别成功次数减失败次数
实际参与下一轮排序，并将历史决策与错误日志提供给模型。经验目录作为只读来源，
新记录写入当前 campaign，技能/知识不会被自动修改。

修改框架或底座后应创建新 campaign。要恢复旧 campaign，使用其匹配的 checkout，
也可从已冻结控制器运行：

```bash
PYTHONPATH="/path/campaign/foundation/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m afd_rsi resume /path/campaign
```

限制：目前是单 workload、单主指标、串行候选搜索；上下文有固定大小，省略的
源码文件会列在请求中。类别评分用于探索排序，不是独立策略质量评测。可信
benchmark/同权限模型进程不构成安全沙箱；哈希用于检测漂移。设备
重置、分布式调度和自动修改底座均不在范围内。
