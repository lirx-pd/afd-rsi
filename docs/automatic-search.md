# 自动搜索、服务验收与续跑

框架在 Linux 本机运行，Agent 仅提出补丁。`search` 先资格验证基线，再执行
“证据 → 假设/补丁 → 隔离构建 → 正确性 → 配置的服务验收 → AB/BA → 决策”。
通过初筛的候选还必须在新 attempt 完整复测后才能替换 best。两次比较均以同一
当前最佳版本为对照；报告不会把合成字段或 HTTP 指标当作算子计时。

## 单机 Codex

需要 Python 3.11+、Linux pidfd、已安装并登录的 Codex CLI。继承 CLI 的模型/provider
配置和登录，不读取或复制认证文件；不需要额外模型 SDK。使用仓库固定的 JSON
schema 和只读模式；补丁由控制器进行 git-apply 和允许文件检查。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m afd_rsi prepare recipes/cpu_search.toml --output /tmp/rsi-search-demo
python3 -m afd_rsi search /tmp/rsi-search-demo
python3 -m afd_rsi report /tmp/rsi-search-demo
python3 -m afd_rsi export /tmp/rsi-search-demo --best --output /tmp/rsi-best
```

该 recipe 的耗时是合成字段，只验证 Agent/控制器的合同。实际 NPU 使用
`recipes/ascend_eager.toml`：先按现场修改 source、已分配设备、模型路径、网络和
预算，再 prepare。其原生驱动确实运行目标的 build_ext 和 torchrun eager A/F
fixture，比较两个 MegaMoe 版本，使用共同 layered 数值参考，校验实际加载库和
设备对象身份；主指标是主 Attention rank 的七个 case 事务样本均值（us）。
FFN rank 没有伪造计时，仍必须给出完整数值/身份/清理检查。

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
16 设备 DP2TP4/EP8 的 eager smoke；正式精度/性能应在 prepare 前固定对应的
GSM8K/perf 参数和数据，不能在搜索中放宽门槛。

## Agent 和 NPU 不在同一台机器

在执行机创建 campaign，并进入文件交换模式：

```bash
python3 -m afd_rsi search /path/campaign --external-agent
```

`campaign/search/pending.json` 给出 request、response 的相对路径、请求 SHA 和
已经预扣的调用序号。将其中 request.json 复制到已登录的提案机，在同版本仓库
执行（框架不提供传输/调度接口）：

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
benchmark/同权限模型进程不构成安全沙箱；哈希用于检测漂移。FULL graph、设备
重置、分布式调度和自动修改底座均不在范围内。
