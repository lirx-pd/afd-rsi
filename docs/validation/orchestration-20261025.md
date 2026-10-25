# 统一后台入口验证：2026-10-25

本轮按要求只验证新入口和异步闭环，不启动新的真实 NPU 优化 campaign。

## 验证范围

- WSL 运行统一 `optimize` 命令，前台返回后后台 worker 继续工作。
- 使用真实 itask-guian 官方 CLI，在已运行的所属任务 afd-mrv2-async 上完成部署、
  detached exec、请求/提案交换、结果下载与 SHA256 校验。未创建、停止或删除任务。
- CPU 合成配方和明确的 fake Codex CLI 验证模型进程退出后继续执行实验的协议。
  初筛与全新独立确认分别留下 attempt，导出 best，按一轮/一次调用预算结束。
- 本地回归覆盖真实独立 worker、重复 launch、终态恢复、模型中断不重试、
  封存上传包的续传身份，以及 early deep / late incremental 请求策略。
- 服务测试使用假服务进程和 HTTP fixture 验证单次启动、测试间隔、九点矩阵，
  单点失败后继续并保留最终失败。没有将这些 fixture 当作模型精度或性能证据。

## 证据

实际本地 job 位于 `results/afd-rsi-orchestration-20261025/transport-job`（工作区父目录），
包含冻结控制器、worker.log、events.jsonl、模型回执与已下载的实验归档。
执行路径、基线/初筛/确认 attempt 和归档 SHA 见
[机器可读记录](orchestration-20261025.json)。该记录对应真实传输烟测的冻结快照；
其后补充的参数边界和封存上传包恢复修复由最终本地回归覆盖。

最终完整回归 **117 项通过、1 项因 WSL 套接字权限跳过**（208.38 秒）；
编排定向测试 6 项通过。
服务 provenance 文件哈希、基础知识闭包和 diff 空白检查通过。

## 边界

此验证未运行真实 Codex 模型、真实 NPU 计算或新的服务矩阵，也不构成收益结论。
没有修改当前 afd-plugin 工作树；后台运行仍要求 WSL/提案机持续运行。
