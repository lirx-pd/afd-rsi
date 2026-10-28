# 统一后台优化入口

`afd-rsi optimize RECIPE --executor local|itask --output JOB` 是整个框架的启动入口。
它冻结当前框架和知识副本，将后续工作交给独立 Python worker，然后返回。既有
prepare/search/propose/resume/export 入口继续可用。无需在 Codex 对话中等待测试。

## 执行流程

1. 验证配方，复制控制器/知识；后台保存目标源码快照和内容哈希。
2. 本机直接执行；iTASK 使用官方 itask-guian 的 auth status、start、cp、exec，
   不复制登录文件，不创建任务，不使用 SSH。start 已运行任务不会重建环境。
3. 执行端独立 prepare/search；每个 launch 留存 intent、归属、进程与退出回执。
4. worker 默认每 60 秒查看进度。服务阶段显示 startup/gsm8k/performance 等状态，
   仅状态变化时输出；完整事件保存在 events.jsonl。
5. 需要提案时下载并校验请求 SHA，在提案机调用已有登录的 Codex CLI。Astra xhigh
   分析、Sol 6.1 xhigh 写补丁；有限的 CLI 调用结束后，提交补丁并启动执行端验证。
   同一任务保存一个 thread ID，分析、编码和后续轮次均恢复该对话。每次调用结束后退出 CLI；
   设备测试期间只有普通进程轮询，不消耗模型轮次等待。
6. 框架双 PASS 才替换 best。结束后自动导出、打包和下载，核对归档及文件哈希。
   本地 results/ 包含 best-export 和实验记录；真实源码不自动修改。

归档解包使用标准库 tarfile 的 data 安全过滤器，提案机和执行机的 Python
须包含该能力（建议 Python 3.12+）。
实际 NPU 环境仍须具备 CANN、Python、vLLM/Ascend/AFD、设备与模型。可选
`--setup-script` 只在执行端 source，例如 CANN set_env.sh；现有环境变量保留。
源代码由本地 recipe.source 复制；绝对模型/数据集路径必须是执行端已准备的路径。
workdir 为执行端持久化父目录，自动创建唯一 rsi-ID 子目录；JOB 必须是新目录。
上传和下载归档当前支持小于 95 MB；超限明确停止传输并保留远端文件。不要将权重、
构建输出放进目标源树。iTASK 暂不自动迁移本地 experience_dirs；使用仓库固化经验，
或通过已有执行端 campaign 配置历史目录，不能把本机路径当作远端路径。

## 服务复用与阶段策略

新 Ascend 配方采用 [分层验证](staged-validation.md)：普通初筛不启动服务，
两轮原生收益都通过后才执行最终服务验收。
每个进入服务验收的候选/基线的加载产物必须独立验证，不能跨版本复用旧服务。每次启动的服务
同时用于 smoke、30 条 GSM8K 和完整性能矩阵。默认 Ascend 配方为 prefill 输入
128/512/2048，输出 1，并发 1/4/8，共 9 点。可用已有参数扩展输入/输出、并发和 RPS。
当前 MegaMoe 专用配方 `ascend_pipeline_scenarios.toml` 已改为原生路由的
32K→512 六并发 decode、16K/512 混合 prefill 和 64K 重载 prefill；
不运行上述小矩阵，详见 [性能场景](service-scenarios.md)。
`--test-interval` 在套件之间等待，`--perf-point-interval` 在矩阵点之间等待，均支持
零和有限非负秒数。套件等待会检查服务仍存活；取消可中断等待并触发原有清理。
单点失败会保存错误并尽可能继续后续点；最终 perf 仍为失败，不绕过验收。

前约 2/3 轮的请求包含 strategy.mode=deep，允许一个有证据的协调改动涉及多个
允许文件；最后约 1/3 为 incremental，围绕当前 best 做局部、单机制的改进。
单轮默认 deep。阶段不是改动行数指标，也不会扩大允许路径、降低门槛或改写知识。

## 状态、恢复与边界

用 `optimize status JOB` 查看中文状态摘要；`--json` 保留原始字段并增加 monitor 判定。
`--follow` 在普通 CLI 中定时刷新，Ctrl-C
仅结束查看。`optimize resume JOB` 在原提案机/同一 boot 重新接管同一 job，仍在
运行的 worker 不会重复启动。已完成请求和 clean proposal 可复用；中断提案不会
自动重试或退还模型调用额度，需检查其 process.json 并使用既有恢复流程。
执行端失败后的显式 resume 使用冻结控制器的健康屏障与续跑规则。

上传封存后的 payload 字节及 SHA 会在重试时原样复用，避免压缩时间戳改变身份。
网络超时不能推断远端实验结束，状态 error 时先查看执行端回执；launch intent
可避免同一阶段重复提交。不要删除 JOB、源码快照或远端 campaign 来重置预算。
需要保持提案机运行；本功能不是系统服务，也不提供跨 boot 自动恢复或调度容灾。
后台任务不自动停止 iTASK 或杀死其他作业。CPU 合成验证只验证编排，不是 NPU 证据。

## Codex 对话保留

新任务的 `JOB/codex-session/session.json` 保存 thread ID 和 campaign 归属，
`workspace/` 是稳定的只读提案工作目录。首次 `codex exec` 建立会话，随后
`codex exec resume THREAD_ID` 恢复同一对话；分析仍用 Astra xhigh，编码仍用 Sol xhigh。
每轮的 schema、analysis.json、命令、事件、stderr 和用量保存在对应提案目录。
当轮 request.json 是当前源码/证据的依据，不能因为保留历史而引用过期 ID。

直接 search 运行时，会话位于 campaign 的 `search/codex-session/`。
手工文件交换时使用 `propose REQUEST --output OUT --session-dir SESSION_DIR`，
同一任务的各轮指定同一目录；不同 campaign 禁止共享。
恢复失败、thread ID 改变或初次启动中断都明确失败，不自动新建对话或退还调用额度。
已冻结旧任务继续使用原适配器；本次修复须新建任务，不改写旧快照。

## 清晰的后台状态与启动等待

启动会输出实际任务目录和可复制的持续查看命令，并明确提交不代表 NPU 已就绪。
在原 optimize 启动命令末尾加 `--follow`，可以直接持续查看后台任务；Ctrl-C 仅退出
查看器，不停止 worker、服务或远端任务。默认后台启动方式保持不变。

`optimize status JOB` 默认显示中文状态、当前阶段、轮次、Agent 调用数、最近回执
距今秒数、原因及日志/结果目录。状态明确区分运行中、异常退出、无法核验、基线失败、
已结束但无独立确认改进、已完成且有独立确认候选。`finished` 不再被直接解释为优化成功。
失败/异常查看返回非零退出码；正常结束返回 0；Ctrl-C 退出查看返回 130。
`--json` 输出保留持久化状态并附带 monitor，不改写历史任务记录。

查看器用原提案机/boot 和 worker 生命周期锁检查存活，避免 PID namespace 或复用
误判。跨提案机或 boot 无法核验时显示未知，不声称 worker 已死亡。worker 每次成功
轮询更新心跳，即使实验阶段没有变化；阶段无变化本身不是卡死证据。长时间构建、
Agent 调用和平台迁移期间显示最近状态年龄；worker 存活不证明设备健康或实验进度。

iTASK start 不再复用普通命令 180 秒超时：改用有归属标签、30 分钟上限的执行器，
完整 stdout/stderr、process.json 和 readiness.json 保存在 JOB/task-start/独立目录。
启动客户端完成或超时后，都通过官方 exec true 检查 Pod 是否可执行；客户端超时但
Pod 已就绪时继续部署，不重复启动。就绪无法确认时保留错误和回执并停止，提示先
核对远端状态。就绪检查不验证 NPU 数值或模型推理。现有冻结任务继续使用原控制器，
新逻辑用于新任务，不修改历史快照或自动重跑 NPU campaign。
