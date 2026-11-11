# MegaMoE 自适应搜索：core-balance/tiling 与 publication/lookahead 的负结果

Scope: A3 MegaMoe，pull-native FFN。本文只记录实测的负结果，防止在新 campaign
中重复已被否定的机制。它不是对全部 tiling 或发布策略的通用否定；新机制仍需新的
源码证据和可证伪实验。

## 测量条件

- campaign：`megamoe-adaptive2-20261110-0445`；源码
  `abe594cd0fb3147b127faee0cd1604adc930eb3513d72dd2f037e97bc0cdf77a`；
  基线 `643b4ad5380b4917a000df2a307e2692`。
- 冻结负载：`dp3tp4-ep4`、async、mrv2、graph、layered GMM、MegaMoe FFN；
  原生为七 case 配对事务计时，服务为 `native-long-decode`（common-window TPOT）
  与 `mixed-prefill`（TTFT）等权均值。
- 晋升门槛：服务均值增益 ≥ 2%，且任一项服务回归不超过 5%。原生计时只作诊断，
  不能证明模型收益。

增益为相对同一基线，负数表示变慢。所有候选均被独立判定为 rejected，best 保持基线。

## core-balance / tiling：强负结果

| 变更 | 原生 | 模型服务 | 判定 |
| --- | ---: | ---: | --- |
| scheduler N tile 减半，让 per-expert GMM job 填满 AIC（core-balance） | -3.22% | **-12.97%** | rejected |

这是本 campaign 中最大的服务退化，且原生同向为负，不是“原生收益被服务噪声抵消”的
情形：缩小 N tile 同时损失了事务吞吐和服务性能。在该 source 与负载上，以减小 tile
提高核占用率的做法不可重复；除非新的 N/K、核占用实测和隔离对照能给出不同机制。

本 campaign 没有单独测量其它纯 tiling 变体；不得把这条结果外推为“所有 tile 改动无效”。

## publication / lookahead：服务侧全负

| 轮 | 变更 | 原生 | 模型服务 | 判定 |
| --- | --- | ---: | ---: | --- |
| 0001 | 专家组边界发布 GMM1_READY / ACT_READY（synchronization） | -3.30% | -1.90% | rejected |
| 0002 | 除末专家外提前发布 per-expert W4A8 GMM2 完成 | +2.97% | -1.20% | rejected |
| 0003 | boundary-driven W4A8 GMM2 drain | +1.09% | -0.38% | rejected |
| 0004 | Combine/Activate 单组 lookahead 交错执行 | +1.69% | -3.83% | rejected |

四个候选都改变了 GMM2 完成发布点或让 Combine 提前消费，服务均值全部为负：

- 第 2、3 轮原生为正（最高 +2.97%），服务仍为负（-1.20%、-0.38%），再次说明合成
  事务计时不能预测模型服务；
- 最接近中性的是 boundary-driven drain（-0.38%），仍远低于 +2% 门槛；
- 第 4 轮 lookahead 把发布点前移一个专家组，服务反而 -3.83%，说明把 FFN 阶段
  提前重叠会把等待搬到服务路径，而不是消除等待。

现有 profile 中 `sync_in`/source readiness 占主导，发布点前移改变的是 FFN 内部的
重叠顺序，触及不到 Attention 侧的就绪与提交等待。重复或组合这些发布/lookahead
改动缺少新的源码证据。

## 边界

- 结果局限于上述 source、基线、负载和冻结门槛；源码或工作负载变化后必须重新测量。
- 负结果保留原始 attempt 与哈希；归档因超过执行端 `cp` 大小上限未下载到本机，
  原始记录仍在执行端 campaign 目录中。
- 原生计时不能用于服务收益声明；本文所有服务结论来自真实 `vllm serve` 场景。

## 证据位置

- 本机：`runs/megamoe-adaptive2-20261110-0445/proposals/<round>/request.json` 中的
  `history-*` 证据包含每轮 category、decision、`gain_pct` 与 `native_gain_pct`；
  `worker.log`、`status.json` 记录轮次与预算。
- 执行端：`.../rsi-8734ae66675d4d62b08ee751248ab108/campaign/search/rounds/`
  与 `campaign/search-report.md`（含每轮 decision 与 adaptive 统计）。
