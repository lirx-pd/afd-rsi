# Eager 原生实验

`native build/compare` 用于已准备好的 A3 Linux 主机。框架自身不分配设备，也不依赖
SSH 或调度平台。先加载本机 CANN 环境，使用已有 torch-npu、vLLM、vLLM-Ascend、
pybind11 和编译工具。控制程序使用标准库；只有原生 fixture 子进程导入推理库。

从仓库根目录执行，源码与所有输出使用绝对路径：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m afd_rsi check
python3 -m afd_rsi native build --source /work/afd-plugin --output /work/build-01
python3 -m afd_rsi native compare --repo /work/build-01/source \
  --output /work/native-01 --device 0,1,2 --profile
```

正式候选使用 MegaMoe 对 MegaMoe：先从基线和人工修改后的源码分别执行 build，
再用下列入口。两侧独立源码和产物身份均写入冻结计划，每个 rank 对各自构建核验。

```bash
python3 -m afd_rsi native build --source /work/afd-plugin-candidate --output /work/build-02
python3 -m afd_rsi native compare --baseline-repo /work/build-01/source \
  --repo /work/build-02/source --output /work/candidate-01 --device 0,1,2 --profile
```

提供 `--baseline-repo` 时先用基线 layered 输出做共同数值参考，再分别验证两侧
MegaMoe，交替测量 baseline/candidate；profiler 分别核对两侧 FFN。省略该参数为
layered 对 MegaMoe 的适配诊断。原始数值与 JSON 汇总保留，自动生成 `report.md`，
正确性通过不会变成性能晋升。本轮只验收可运行闭环，不调整算子追求收益。

输入/输出目录必须分离，输出必须不存在。build 复制完整源码内容快照，包括未提交
源码，随后调用目标自己的 `setup.py build_ext --inplace`。编译器产生的文件均留在
该实验的 build 输出树，不改原 checkout 或共享 CANN。源码快照不携带 `.git`，
因此通过 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN=0.0.0+rsi.<hash>`
提供构建本地版本，真实身份仍以完整源码与二进制哈希为准。

compare 使用仓库内固定的真实 `AsyncNPUFFNModelRunner` + CAM connector fixture。
它保留原测试的冷启动、多层切换、重复输入、7/16/129 token、稀疏/零工作专家和
256 行容量边界。每个 Attention source 依次发送，跨 source 有测试用 barrier。
默认 1A+2F；`--attn-ranks 2 --device 0,1,2,3` 为 2A+2F，仍是串行 source，
不能宣称同时 fan-in。FFN TP1、top-k2、W4A8、seed=7+FFN rank、routed scale=2.5。

默认 H256/I512/每 FFN 4 专家；`--hidden 4096 --intermediate 2048 --experts 32`
可覆盖模型的矩阵宽度及每 rank 专家数，但仍使用合成权重、两 FFN、两层和 top-k2，
不能代替真实 checkpoint、EP8 或 top-k6 的服务验证。`--per-group` 切换两组 scale。

每次实验依次执行：layered reference 保存实际输出、MegaMoe 正确性、AB/BA 原生
计时、可选 profiler。所有阶段均为 eager，新建 torchrun 进程组。数值门槛沿用目标
fixture：MegaMoe 对保存的 layered 输出 rtol=0.04、atol=2e-5；layered 重跑要求精确
相等。它验证对 layered 的一致性，不是独立数学实现的正确性证明。

计时默认两组 AB/BA，每个 case 热身 3 次、保留 10 次；`--pairs/--warmups/--repeats`
在启动前冻结。Attention 侧两个 NPU event 包围 DS→FFN→CR，同一 Attention 时钟域，
单位微秒。统计保留逐 case、逐 source、逐独立启动的中位数及增益，包含退化样本。
计时阶段关闭 FFN 的逐事务 host 观察；Attention 仍在 event 结束后校验输出并控制
下一个事务，因此这是有限、串行的 A→F→A 事务延迟，不是饱和服务吞吐或单 kernel
耗时。profiler 独立启动，不进入计时统计。

每个 rank 在真实 warmup 后记录 import 的包/绑定和 `/proc/self/maps` 中的
`libcust_opapi.so` 路径及哈希，拒绝其他 checkout 的库。控制器在每阶段前后核对
原始源码集合及预期绑定/API/MegaMoe 设备对象；所有 rank 必须与预期一致。
`--profile` 还要求两 FFN 的 `kernel_details.csv` 均含真实 MegaMoe kernel，
单有已构建的 `.o` 不算执行证明。

输出包含 `plan.lock.json`、底座快照、layered `.pt`、每阶段 argv/PID/退出码、
stdout/stderr、全 rank 身份与结果、原始 profiler 和 `summary.json`。任一缺失 rank、
未清理进程、非有限时间、身份不符或正确性失败均停止后续阶段并保留现场。
阶段进程组由现有执行器有界清理；硬件健康和异常后的设备恢复仍需单独核查。

服务验证继续使用 [固化的 E2E 流程](npu-service-testing.md)。不要将此原生实验结果
转换成 HTTP 指标或通用 campaign 的虚构 rank 结果；本入口当前不做候选自动晋升。
