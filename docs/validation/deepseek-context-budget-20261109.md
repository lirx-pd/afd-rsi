# megamoe-deepseek-20261109-0010 请求预算故障修复

该运行在首次提案前失败，未调用模型。原始控制器错误是
`request metadata exceeds the frozen context budget`，外层误以最后完成的
baseline profile 作为当前阶段，并仅展示通用 executor 失败。

远端重新审计确认：正确性、两次 benchmark 和 profile 均为 valid；基线
`7a329bed3f6e4719a3dc501b7a6456ad` 已完成。故障在请求构建阶段，与 NPU
算子失败无关，不需要为定位或验证本次打包修复重跑基线。

## 原因与修复

冻结上限为 2097152 字节，原请求实际为 2968602 字节。其中 evidence 约
1920970 字节，SDK 依赖约 782192 字节。未设上限的 selected-core 明细占据
主要空间；上一轮原生打点验证未覆盖完整 profile 到请求打包的组合路径。

- 发给模型的每个核心时间线最多保留 128 个事件，保留每种阶段首次/末次事件，
  其余位置均匀采样。标记原事件数和省略数，保留原始 trace 的路径与哈希。
- 完整 trace、阶段总量、FFN 等待分解和重叠界限保持不变，诊断样本不作为新计时。
- 请求显式携带冻结预算，DeepSeek 默认沿用该预算；旧请求回退到 200000 字符，
  显式 `AFD_RSI_DEEPSEEK_MAX_CHARS` 仍优先。
- 状态将准备上下文作为独立阶段。控制器失败包含子进程退出码、stderr 和回执路径，
  可归档的失败结果同样下载并校验；归档失败另行记录，不替换原错误。

## 真实数据回放

在原任务 `afd-v032-mega-20261104b` 的独立诊断目录加载修复后的打包器，
使用该运行的原计划、基线、原始 profile 和 69 个完整 SDK 依赖。
未改冻结 campaign、未重置预算，模型调用和新增 NPU 试验均为 0。

| 项目 | 结果 |
| --- | --- |
| 修复后请求 | 1412700 字节，小于冻结上限 2097152 |
| DeepSeek 请求正文 | 1089954 字符，通过冻结预算检查 |
| SDK / 仓库源码文件 | 69 / 16，保留 Pull 协议源码 |
| rank 1 明细事件 | 7383 → 1024，完整统计不变 |
| rank 2 明细事件 | 5136 → 1024，完整统计不变 |
| 逐 rank 核对 | stage totals、FFN execution、overlap、named stages、substage breakdown 完全相同 |
| 原 campaign | plan、plan digest、search state、best 哈希不变 |

证据保存在该本地 run 的 `diagnostics/context-budget-fix/`。证据包 SHA256：
`d499ba1950309d5f87dcbffa62592d27575afa1317b05226237e5440d9bf56c6`。
修复后请求 SHA256：`94a67dce0d7c75fd1fb9c626d3423c79d44e6dfe256933e473dafad4766213af`。

本地全量回归为 **322 passed, 3 skipped**；foundation 校验和 `git diff --check`
通过。新增回归覆盖膨胀时间线、统计不变、DeepSeek 冻结预算、原始子进程错误与
失败归档下载，以及上下文阶段的状态展示。

本次验证的是请求构建和预算处理，未向真实模型接口发送请求，也未重启这个冻结
campaign。新的仓库代码不会自动替换已有 run 内封存的控制器。
