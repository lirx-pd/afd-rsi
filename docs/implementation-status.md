# 实施记录：2026-10-23

当前实现为仓库内 A3 能力底座和本机手工候选评测的第一条完整链路。

已实现：

- 独立 Git 仓库与无运行时第三方依赖的 Python 控制程序。
- 8 个本地技能、A3/CAM/W4A8/测量/排错正文、arch22 流水与切分材料、MTE 地址和可见性材料。
- 实际 AFD workspace 头文件及许可证、内容来源与哈希记录。
- 底座清单、仓库内资源闭包校验、技能上下文装配和 campaign 实际内容快照。
- 配置冻结、含新增/修改/删除与执行模式的源码身份、独立基线/候选工作区。
- 构建、正确性、AB/BA、本机进程组超时与清理检查、全 rank 身份检查、判定、报告与补丁导出。
- 允许文件之外的补丁先拒绝，原工作区不参与候选编辑。

验证结果：

- `python3 -m pytest -q`：32 项通过（40.41 秒）。
- `ruff check src tests tools benchmarks`：通过。
- 8 个技能的 frontmatter 校验与底座资源闭包校验：通过。
- 仓库内 CPU 合成演练完成 prepare、baseline、patch、compare、export。
- C++ workspace 参考头文件的主机编译、区间与对齐检查：通过。

首次仓库内演练发现 Git 父仓库发现导致补丁跳过，保留其未改进的 FAIL 记录；
已通过显式指定工作树修复，并增加嵌套仓库回归测试。

修复后的合成演练证据保存在未提交的 `runs/cpu-smoke-20261023-v2/`，报告为该目录下的
`report.md`；导出保存在 `runs/cpu-smoke-export-20261023-v2/`。指标是 fixture 输入，
不是任何实际算子耗时或 NPU 加速比。新环境按 README 可重新生成同类证据。

仍待实施：

- 实际 A3 环境、编译产物加载身份探针与原生 MegaMoe benchmark。
- 多 case/真实 checkpoint、图 replay、事务时间和服务验证。
- Agent 自动候选生成、假设队列、调用预算及经验检索/晋升。
- 完整的崩溃恢复与设备健康确认；目前未完成的进程记录会阻止新执行，等待人工核对。
- 跨 campaign 策略改进的独立任务评测。

已有命令型 Ascend 结果合同不代表目标环境已经适配或验证。当前不发布 NPU
正确性、性能、完整 M0–M4 完成或自主 RSI 能力提升的结论。

## 本机服务测试接入

按 AFD-Plugin-e2e-tools 的实际方法，将版本
`1cec506d7acb0c0bffd311987ba4f2405e118272` 的服务测试代码和许可证固化到
`benchmarks/service/`。新增 `afd-rsi service plan/run/status/pack` 入口，保留其
标准 vllm serve、预检、模型就绪、HTTP 工作负载与进程清理。详情和运行命令见
`docs/npu-service-testing.md`。

当前完整测试结果：35 项通过，1 项因沙箱禁止本地 socket 绑定而跳过；随后
加强 V4 配置断言的专项测试为 3 项通过、1 项相同原因跳过。静态检查、底座
校验和 diff 检查通过。底座现有 8 个技能、44 个文件。

V4 计划验证覆盖 8A/8F、Attention TP4、FFN TP1，以及 async MRV2、Layered 和
MegaMoe 开关。离线计划不启动服务，不证明目标栈支持对应参数。
本机缺少 NPU/推理和 HTTP 依赖，未执行真实 NPU 服务 run。算子级 benchmark
和实际加载身份核验仍待实施；服务结果不伪装成算子级 compare 结果。
