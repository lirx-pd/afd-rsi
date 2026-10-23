# NPU 测试接入与实际执行

模型/服务验证以仓库内固定版本的 AFD E2E 实现为准：
`benchmarks/service/afd_e2e/`。该实现来自 AFD-Plugin-e2e-tools，原代码、许可证、
来源版本和内容哈希保存在仓库内。无需运行时读取原工具仓库。

入口为 `PYTHONPATH=src python3 -m afd_rsi service`，后接该工具真实的
`plan/run/status/pack` 参数。控制器先检查底座完整性，再以 exec 交接给服务测试
程序，让其原有信号处理、工作负载和清理流程完整生效。不可套用通用 command
超时后直接杀整个服务测试程序，否则可能跳过它自己的有界清理。

## 测试分工

| 范围 | 执行方式 | 必须保留的证据 |
| --- | --- | --- |
| 模块控制逻辑 | 当前 CPU 测试 | 仅声明控制合同通过 |
| 本机真实模型/服务 | `service run` 的标准 vllm serve + HTTP | 预检、全部服务、请求、质量/性能、退出和清理结果 |
| 算子与 FFN 工作项 | 后续独立原生 benchmark | 实际 native 加载身份、设备计时、数值与路由参考 |
| 证明 MegaMoe 实际执行 | 运行配置加 profiler/加载身份探针 | 不能仅凭 HTTP 成功或环境变量判断 |

服务验证不伪造通用 compare 的 rank/source/artifact 输出，服务性能也不直接
转写为算子延迟。当前服务入口与算子级 compare 分开归档，未实现自动导入或晋升。

## 本机环境

使用已安装并匹配的 CANN、torch-npu、vLLM、vLLM-Ascend、待测 afd-plugin、
CAM/Layered/MegaMoe 算子、模型权重；httpx 用于 HTTP 请求。GSM8K 需要
lm_eval==0.4.12 及对应数据。工具不会安装或替换推理依赖。

`--python` 选择预检和测试 Python，`--vllm-bin` 选择实际服务 CLI，两者必须
属于所验证的环境。必须显式传 `--repo` 指定被测 checkout。它负责 Python 路径
和 Git 元数据，但不证明已安装算子正好来自该 checkout。两侧构建产物、包和
算子路径还要单独核验；测试器不负责构建或安装候选。

只使用已分配设备，通过 `--device` 或 ASCEND_RT_VISIBLE_DEVICES 提供。
不调用任何设备申请工具，不使用远程环境。plan 无需 NPU，run 需要真实设备。

## MegaMoe 首次验证命令

从仓库根目录执行。下面参数来自工具支持的 V4 部署模板：16 张设备，
dp2tp4-ep8。设备、网卡、Python、vLLM 和权重必须替换成实际值；不能用小模型
成功代替当前 DeepSeek-V4 W4A8 MegaMoe 的验证。

```bash
export PYTHONPATH="$PWD/src"
# 预先设置 TARGET_REPO、MODEL_PATH、DEVICES、RUNTIME_PYTHON、VLLM_BIN、NIC、AFD_HOST。
ARGS=(--repo "$TARGET_REPO" --model v4-flash --model-path "$MODEL_PATH"
      --device "$DEVICES" --topology dp2tp4-ep8
      --comm async --mrv mrv2 --layered-gmm on --moe-ubatching off
      --execution eager --dbo off
      --serve-env AFD_ASYNC_CAM_MEGA_MOE=1
      --python "$RUNTIME_PYTHON" --vllm-bin "$VLLM_BIN"
      --nic "$NIC" --afd-host "$AFD_HOST" --offline --test smoke
      --output runs/megamoe-service-smoke)
python3 -m afd_rsi service plan "${ARGS[@]}"
python3 -m afd_rsi service run  "${ARGS[@]}"
python3 -m afd_rsi service status runs/megamoe-service-smoke --json
```

plan 与 run 传入同一组完整参数；run 不会根据旧 plan 自动恢复参数。输出目录
只允许复用仅含普通 plan.json 的目录，不覆盖已有结果。plan 成功不说明目标
vLLM 版本接受命令或 NPU 算子可运行。

先检查 commands.json 中的 A/F serve 命令、插件环境、模型和设备切分。
run 将实际预检、启动、监控就绪、发送 smoke 请求并清理。检查 summary.json 的
status、inference_status、shutdown_status、cleanup_status 和 test_results；
保留 manifest、preflight、attention/ffn 以及请求日志。强制结束与正常退出不混淆。

eager smoke 通过后，独立目录运行 graph smoke。性能和 GSM8K 同样使用新目录、
明确的样本数/并发/长度/种子和冻结判据。工具支持 --test all,perf，一次启动后
执行多个工作负载；它不把质量不达标当成服务崩溃，结果须逐项审核。

与 layered 基线比较时显式设置 AFD_ASYNC_CAM_MEGA_MOE=0，其他参数和负载相同。
每一侧均实际构建并配置自己的包/算子，使用全新服务进程和结果目录。当前没有
自动 A/B 集成，不允许省略身份核验后直接宣称候选加速。

## 验证范围

仓库提供 CPU-safe plan 与真实 HTTP/子进程模拟测试，覆盖启动、请求失败和清理。
这些测试验证测试器合同，不能替代上面的真实 NPU run。当前开发主机没有 NPU
环境；服务部署与 MegaMoe 路径的上卡结论保留未验证。
