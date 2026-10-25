# AFD-RSI

面向 AFD MegaMoe 的本机优化工作台。Python 3.11+；执行器要求 Linux 与可读的
`/proc`。控制程序没有运行时第三方依赖，不导入 torch 或 torch_npu。

当前支持仓库内技能/知识闭包、冻结实验、隔离构建、正确性、真实服务验收、
AB/BA、报告和补丁导出，以及 Codex 提案、受预算约束的自动搜索、独立复测晋升、
历史经验选择和中断续跑。它不依赖其他 RSI 框架代码。

8 个技能是为本机 A3 MegaMoe 重新整理的工作流，技术正文、参考头文件及完整
内部依赖均在本仓库；不是完整上游技能库的逐文件迁移。来源范围与参考代码身份
记录在 `knowledge/provenance.json`。未知的 CANN API 能力仍需目标 SDK 验证。

真实模型/服务测试已接入仓库内固定版本的 AFD E2E 实现，使用标准 `vllm serve`
启动 A/F、预检、就绪、真实 HTTP 请求和进程清理。详见
[NPU 服务测试](docs/npu-service-testing.md)，入口为 `python3 -m afd_rsi service --help`。

真实 eager/FULL graph 原生实验提供 `native build/compare`，调用目标自己的构建入口、真实
runner/connector、加载身份检查、layered 正确性参考、AB/BA 事务计时和独立 profiler。
支持独立 MegaMoe 基线/候选和自动 JSON/Markdown 报告，见 [原生实验](docs/eager-native.md)。FULL 使用 eager layered 参考，并检查实际捕获和 replay。
本轮样例服务门槛按约定暂不考察正常关闭错误；原始失败仍归档，推理、身份和清理必须通过。

自动搜索使用已有登录的 Codex CLI，也支持提案机和执行机通过文件交换，详见
[自动搜索与续跑](docs/automatic-search.md)。`recipes/ascend_eager.toml` 和 `recipes/ascend_graph.toml` 将真实
原生构建和 AFD E2E 服务接入同一个 campaign。默认验收 30 条 GSM8K，并记录
128/512/2048 输入、1 输出 token、并发 1/4/8 的 prefill TTFT 矩阵；通过门槛后自动采集独立 profiling，
把每个 FFN rank 的设备任务分析和历史结果交给 Agent 提出下一轮优化假设。固定控制器不耦合调度平台；统一 optimize 入口支持本机和可选 iTASK 执行器，不使用 SSH。
独立诊断构建还记录 MegaMoe 内部逐核、逐 tile 的阶段和等待；当前摘要覆盖首次双槽
复用，生产服务与普通计时使用未打点产物。内部周期不是 Cube PMU 利用率。
通用 benchmark 遵守 [结果合同](docs/result-contract.md)。

Ascend 配方默认先做算子数值和配对性能筛选，再独立复测；只有两轮原生收益都达标，
才运行最终 E2E 精度/性能矩阵和候选 profiling。初始基线只做原生资格与一次 profile。
提案必须给出测量证据、改动机制、预测、否定条件和风险。可用 `compare --operator-only`
手动只测算子，结果不能替代最终晋升资格。详见 [分层验证](docs/staged-validation.md)。

专项配方 `recipes/ascend_pipeline_scenarios.toml` 新增固定 DP 不均衡 prefill 和
长上下文 decode TPOT，同一次服务启动完成，详见 [专项服务指标](docs/service-scenarios.md)。

## 一条命令启动优化

```bash
PYTHONPATH=src python3 -m afd_rsi optimize recipes/ascend_graph.toml --executor itask --task YOUR_TASK --workdir /absolute/persistent/workdir --setup-script /usr/local/Ascend/cann-9.1.0/set_env.sh --output runs/optimization-001
```

先按现场配置 recipe 中的目标源码、设备、权重、离线数据集与网络。框架会启动已有
所属任务（已运行则保持），快照代码、部署、prepare 并后台搜索；命令返回后不需要
保持 Codex 对话。WSL 上的后台 Python worker 定时查看执行状态，仅提案时调用
已登录 Codex CLI，模型退出后才启动设备实验。WSL/提案机需保持运行。
本机执行同一命令，使用 `--executor local` 并省略 task/workdir。

```bash
PYTHONPATH=src python3 -m afd_rsi optimize status runs/optimization-001
PYTHONPATH=src python3 -m afd_rsi optimize status runs/optimization-001 --follow
PYTHONPATH=src python3 -m afd_rsi optimize resume runs/optimization-001
```

简要进度在 `worker.log`，完整状态与事件在 `status.json` / `events.jsonl`；结束后
自动下载并校验 `results.tar.gz`，导出最佳补丁但不自动修改真实源码。默认轮询 60 秒，
可用 `--poll-seconds` 设置。任务不会自动停止或删除。一次服务启动内完成 GSM8K
与默认 9 点 prefill 矩阵，套件间隔 5 秒、点间隔 3 秒；等待不计入性能样本。
前约 2/3 轮允许较深的协调优化，后期收敛为小改动；全部验收门槛保持不变。
详细行为、恢复限制见 [统一后台入口](docs/orchestration.md)。

## 本机演练

从仓库根目录执行。以下输入中的耗时是合成字段，只验证控制链路。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m afd_rsi check recipes/cpu_demo.toml
python3 -m afd_rsi context afd-megamoe

DEMO_ROOT=$(mktemp -d)
python3 -m afd_rsi prepare recipes/cpu_demo.toml --output "$DEMO_ROOT/campaign"
python3 -m afd_rsi baseline "$DEMO_ROOT/campaign"
python3 -m afd_rsi patch "$DEMO_ROOT/campaign" examples/faster.patch
python3 -m afd_rsi compare "$DEMO_ROOT/campaign"
python3 -m afd_rsi report "$DEMO_ROOT/campaign"
python3 -m afd_rsi export "$DEMO_ROOT/campaign" --output "$DEMO_ROOT/export"
```

`prepare` 的输出目录必须尚不存在。它复制目标源码（包括未提交的新文件）、
技能、知识、benchmark 与控制程序快照。原始源码不参与候选修改。
Git 目标使用未忽略文件和已跟踪文件；普通目录包含除 `.git`、`__pycache__`
以外的文件。源码符号链接不支持。大型模型与输出应放在源码目录之外。

`baseline` 保存基线检查；`compare` 会重新构建两侧、先验证正确性，再交替执行
冻结数量的测量。它保存新 attempt，保留所有原始失败。`min_gain_pct` 是延迟
下降的百分比，必须为正收益且达到冻结门槛；当前仅支持一个 workload、一个
主指标及明确 primary rank；case 集合的聚合由冻结 driver 定义。

`patch` 先在临时副本核对允许文件，再应用到候选。也可以手工编辑
`workspaces/candidate` 中允许的文件。构建输出必须放在提供的 build 目录。

`report` 复核原始结果哈希和判定一致性；`export` 核对候选仍为被测版本，导出的
patch 按导出包 `APPLY.md` 指定工作树后应用到相同起点，避免父仓库的路径发现
影响补丁。导出失败候选不会将其变成
已接受候选，报告保留原判定。

## 技能与知识维护

运行时只读仓库内容和已冻结快照，不查全局技能、其他仓库或网络资料。所有实际
资源由 `skills/manifest.json` 统一锁定；Markdown 资源链接必须存在且位于闭包内。

维护者有意变更底座后执行：

```bash
PYTHONPATH=src python3 tools/freeze_foundation.py
git diff -- skills/manifest.json
PYTHONPATH=src python3 -m afd_rsi check
```

冻结工具不由实验执行器调用。修改底座或控制程序后需创建新 campaign；旧实验
保留实际快照。哈希用于检测漂移，不是对恶意同权限程序的隔离或签名。

新补充的 [A3 专家合批经验](knowledge/performance/megamoe-expert-packing-20261025.md)
已接入 `afd-megamoe` 和性能分析技能上下文：包含 512 行跨专家 tile、G2/SENT
所有权、构建身份、真实不均衡 prefill/长上下文 decode 的结果和边界。
这是人工审阅的外部经验，不是匹配实验指纹的动态 experience，也不代表框架晋升；
旧 campaign、配方和门槛保持原样。

## 验证与下一步

```bash
python3 -m pytest -q
```

测试覆盖完整演练、错误输出/退化拒绝、缺失或错误 rank、非有限指标、越界补丁、
证据变更、预算停止和真实子进程组清理。CPU fixture 不是 NPU 或模型验证。

模型/服务验证采用上述 E2E 流程；`search` 用冻结基线和当前最佳候选完成初筛及
独立复测，`resume` 先协调进程和设备健康再复用已完成阶段，`export --best` 导出
确认过的最佳版本。原生入口的合成权重和串行事务边界见其文档。

本次交付与后续边界记录在 [实施记录](docs/implementation-status.md)，实际功能验证见 [RSI 验证](docs/validation/rsi-20261023.md)，FULL graph 实测见 [graph 验证](docs/validation/graph-20261023.md)，此前原生记录见 [eager 验证](docs/validation/eager-20261023.md)。

当前限制：本地命令仍属于可信 benchmark；执行器用继承的试验标记与 pidfd 跟踪
退出进程组的子进程。跨主机/重启后的回执无法自动证明资源归属，会停止续跑。
Ascend 中断后基础计算健康检查通过才继续，不自动重置设备，也不保证所有通信
故障都能恢复。历史经验不会自动改写底座；原生事务优化不能替代正式模型性能验收。

项目尚未选定整体发布许可证。复制的 AFD 参考头文件保留原版权声明和其
`knowledge/reference-code/LICENSE`；不把该许可证自动扩展到其他内容。

真实单轮 Codex 候选的构建、FULL 验证、服务、AB/BA 与拒绝经验检验见
[真实候选验证记录](docs/validation/real-candidate-20261023.md)。

30 条 GSM8K、小 prefill 与自动 profiling 的实现及真实验证见
[指标与 profiling 验证](docs/validation/metrics-profiling-20261023.md)。

多轮真实候选、内部流水打点、独立确认和保留版本见
[MegaMoe 多轮优化验证](docs/validation/megamoe-iterations-20261024.md)。
