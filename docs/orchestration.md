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
   设备测试期间只有普通进程轮询，不保留模型会话，不消耗模型轮次等待。
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

每个候选/基线的加载产物必须独立验证，不能跨版本复用旧服务。每次启动的服务
同时用于 smoke、30 条 GSM8K 和完整性能矩阵。默认 Ascend 配方为 prefill 输入
128/512/2048，输出 1，并发 1/4/8，共 9 点。可用已有参数扩展输入/输出、并发和 RPS。
`--test-interval` 在套件之间等待，`--perf-point-interval` 在矩阵点之间等待，均支持
零和有限非负秒数。套件等待会检查服务仍存活；取消可中断等待并触发原有清理。
单点失败会保存错误并尽可能继续后续点；最终 perf 仍为失败，不绕过验收。

前约 2/3 轮的请求包含 strategy.mode=deep，允许一个有证据的协调改动涉及多个
允许文件；最后约 1/3 为 incremental，围绕当前 best 做局部、单机制的改进。
单轮默认 deep。阶段不是改动行数指标，也不会扩大允许路径、降低门槛或改写知识。

## 状态、恢复与边界

用 `optimize status JOB` 查看保存状态；`--follow` 在普通 CLI 中定时刷新，Ctrl-C
仅结束查看。`optimize resume JOB` 在原提案机/同一 boot 重新接管同一 job，仍在
运行的 worker 不会重复启动。已完成请求和 clean proposal 可复用；中断提案不会
自动重试或退还模型调用额度，需检查其 process.json 并使用既有恢复流程。
执行端失败后的显式 resume 使用冻结控制器的健康屏障与续跑规则。

上传封存后的 payload 字节及 SHA 会在重试时原样复用，避免压缩时间戳改变身份。
网络超时不能推断远端实验结束，状态 error 时先查看执行端回执；launch intent
可避免同一阶段重复提交。不要删除 JOB、源码快照或远端 campaign 来重置预算。
需要保持提案机运行；本功能不是系统服务，也不提供跨 boot 自动恢复或调度容灾。
后台任务不自动停止 iTASK 或杀死其他作业。CPU 合成验证只验证编排，不是 NPU 证据。
