# 2026-10-23：真实 MegaMoe 单轮自动候选检验

本轮完整走通真实 NPU 候选的拒绝路径。Codex 提出一行优化，构建、数值和真实服务均通过；
事务指标中位数下降 **1.50%**，未达到冻结的 **2%** 门槛。框架拒绝候选，
没有进入独立确认，也没有把补丁应用到真实 afd-plugin 仓库。

## 输入和固定条件

- 基线：`beea4aa1cfe53a1916f6bf1ed52d57b28609596f`，已包含 CANN const 修复。
- 源码身份：`1d290c11d286f7b81273d37a89a0a1bfd17fef0061dc46a2cd02bafdbcded179`。
- 框架：`05d77a2`；在既有 iTASK `afd-mrv2-async` 中执行，提案仅使用 WSL 已登录 Codex。
- FULL 配置：Attention eager、FFN FULL；H4096/I2048/每 FFN 32 专家、1A+2F、合成 W4A8 权重。
- 七个 case、固定 seed、每组热身 5 次、采样 20 次，四组 AB/BA。主指标是 Attention 的七 case 事务样本均值；用四组中位数判定收益。
- 一次 Codex 提案调用、一个候选、6 小时 wall-time 上限；只豁免 E2E 归类的关闭错误。
- 真实服务：V4 Flash W4A8 checkpoint、16 设备、DP2TP4/EP8、Async CAM MRV2、smoke。

## 自动候选及检视

Codex 在 `afd_mega_moe_ffn.cpp:90` 增加：

```cpp
if (GetBlockIdx() - Cam::MEGA_COMM_AIVS >= rows) return;
```

它跳过无输出行计算 AIV 的 OutputTile 初始化、bias 搬运和 pipe reset。静态检视确认 guard 与原行循环的入口条件一致，
外层全部计算 AIV 仍执行 OUTPUT 到达信号；活动行算术、量化、workspace 和通信协议未变。
0..128 行及受支持 AIV 数量的 host 枚举保持行归属相同；这项枚举不作为设备证据。

## 真实执行结果

- 基线资格验证：实际编译、共同 layered eager 参考、FULL 数值、服务 gate、四组 benchmark 通过。
- 初筛：重新构建两侧，候选 FULL 数值与真实模型 smoke、加载身份、清理通过。
- 每个 FFN 实际 capture=2；计时阶段完整 replay，禁用 Python eager 事务并检查稳定地址。
- 两侧设备二进制确实不同：基线 `8b4b0ee93d70513e3a32ca4dae655bda24981bf25ad20a5f8be5aa061f3f1146`；候选 `f6a1c944b3a049e89f29f713ec135cac3e08a81653d7e706d1488e7a4df5f3d7`。

| 独立启动配对 | 基线 μs | 候选 μs | 降幅 |
| --- | ---: | ---: | ---: |
| 1 | 1278.57 | 1232.33 | 3.62% |
| 2 | 1239.69 | 1250.26 | -0.85% |
| 3 | 1293.69 | 1218.09 | 5.84% |
| 4 | 1241.79 | 1278.62 | -2.97% |
| 中位数判定 | 1260.18 | 1241.29 | 1.50% |

四组中有两组候选退化。所有慢样本保留，没有降低门槛或重新挑选样本。该结果不能证明稳定收益。

| Case | tokens | sparse | 基线 μs | 候选 μs | 降幅 |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 7 | False | 1148.04 | 1129.05 | 1.65% |
| 1 | 7 | True | 380.74 | 376.40 | 1.14% |
| 2 | 7 | False | 1144.86 | 1126.40 | 1.61% |
| 3 | 16 | False | 2298.89 | 2252.08 | 2.04% |
| 4 | 16 | True | 390.12 | 385.15 | 1.27% |
| 5 | 129 | True | 974.69 | 977.46 | -0.28% |
| 6 | 129 | False | 2473.78 | 2430.32 | 1.76% |

逐 case 表是该 case 的逐启动样本均值再取中位数；验收仍按预先冻结的整套事务指标。

## 闭环与保留结果

- `search` 在一轮/一次提案预算后正常结束，round decision=`rejected`，best 仍为基线。
- 拒绝经验带有补丁、父版本和证据哈希，已写入 search/experience；实际读回 1 条记录并验证相同父版本的重复提案会被排除。没有发起第二次模型调用，也未自动改写 skills 或知识。
- `report`、全部 attempt/best 证据审计通过，`export --best` 的补丁是 0 字节。
- 真实 afd-plugin 仓库 git 状态干净，内容身份重新核验与冻结基线完全相同。
- NPU 测试进程已清理；既有任务保持运行，没有复位设备或停止其他任务。
- 同一 iTASK 环境的完整框架回归：**73 passed，无跳过**。

完整索引见 [JSON 记录](real-candidate-20261023.json)。原始日志、逐样本、模型事件、
候选补丁、经验和 SHA256 已核验归档在 `runs/real-candidate-20261023-01`。
远端完整实验保留在任务 workdir 的 `afd-rsi-real-candidate-20261023-01/campaign`；大型原生产物保留远端。

本轮没有确认可保留的优化，也没有进行 GSM8K、服务性能、其他 shape/模式的完整验收。
正常关闭错误按约定未修复；合成权重事务结果不等于真实服务吞吐或精度提升。
