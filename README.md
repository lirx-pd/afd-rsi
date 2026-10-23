# AFD-RSI

面向 AFD MegaMoe 的本机优化工作台。Python 3.11+；执行器要求 Linux 与可读的
`/proc`。控制程序没有运行时第三方依赖，不导入 torch 或 torch_npu。

当前实现是 **M0 的 A3 知识子集与 M1 的手工候选评测链路**。支持仓库内技能和
知识闭包校验、内容快照、独立源码工作区、构建、正确性、交替 AB/BA、判定、
报告与补丁导出。它不依赖其他 RSI 框架代码。

8 个技能是为本机 A3 MegaMoe 重新整理的工作流，技术正文、参考头文件及完整
内部依赖均在本仓库；不是完整上游技能库的逐文件迁移。来源范围与参考代码身份
记录在 `knowledge/provenance.json`。未知的 CANN API 能力仍需目标 SDK 验证。

真实模型/服务测试已接入仓库内固定版本的 AFD E2E 实现，使用标准 `vllm serve`
启动 A/F、预检、就绪、真实 HTTP 请求和进程清理。详见
[NPU 服务测试](docs/npu-service-testing.md)，入口为 `python3 -m afd_rsi service --help`。

自动 Agent 搜索、经验晋升、真实 MegaMoe 算子 benchmark/加载身份探针以及完整
崩溃后恢复尚未实现。当前 `compare` 使用维护者明确提供的本地程序；程序负责
启动它的所有 rank，并遵守 [结果合同](docs/result-contract.md)。没有远程接口。

## 本机演练

从仓库根目录执行。以下输入中的耗时是合成字段，只验证控制链路。

```bash
export PYTHONPATH="$PWD/src"
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
主指标及明确 primary rank。复杂 case 集合需拆成不同 campaign。

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

## 验证与下一步

```bash
python3 -m pytest -q
```

测试覆盖完整演练、错误输出/退化拒绝、缺失或错误 rank、非有限指标、越界补丁、
证据变更、预算停止和真实子进程组清理。CPU fixture 不是 NPU 或模型验证。

模型/服务验证采用上述 E2E 流程。下一步按 `docs/design.md` 接入真实 A3 的构建与加载身份核验、独立正确性参考、
完整 FFN/事务计时，然后实现预算受控的候选生成和经验闭环。

本次交付与后续边界记录在 [实施记录](docs/implementation-status.md)。

当前限制：本地命令属于可信 benchmark；子进程必须留在自己的进程组，不支持
daemonize/setsid 逃离监管。仅核实主机进程退出，未证明 NPU 健康。遇到未完成的
进程记录会停止，要求先核对现场；不自动恢复或重复启动可能仍活跃的工作。

项目尚未选定整体发布许可证。复制的 AFD 参考头文件保留原版权声明和其
`knowledge/reference-code/LICENSE`；不把该许可证自动扩展到其他内容。
