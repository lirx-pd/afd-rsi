# 本机命令与结果合同

命令以 argv 列表运行，不经过 shell。占位符仅有 `{python}`、`{repo}`、
`{foundation}`、`{build}`。工作目录是该侧源码快照。build、correctness、benchmark
为三个必选阶段；配置 `[service]` 后还包括真实服务验收阶段。所有输出、预热与测量由明确的 benchmark 实现，控制器不从
进程墙钟推算算子延迟。主机 elapsed 仅记录执行与超时。

子进程获得：

- `AFD_RSI_RESULT`：本次唯一 JSON 结果路径，正确性与测量阶段必须创建。
- `AFD_RSI_BUILD_DIR`：本 attempt/side 的独立产物目录。
- `AFD_RSI_SOURCE_HASH`：当前源码内容指纹。
- `AFD_RSI_SIDE`：baseline 或 candidate。
- `AFD_RSI_WORKLOAD_ID`：冻结的 workload 身份。
- `AFD_RSI_PLAN` / `AFD_RSI_RUN` / `AFD_RSI_ATTEMPT`：冻结配置与证据目录。
- `AFD_RSI_TIMEOUT`：本阶段剩余时间上限。

build 成功后控制器独立计算 recipe 声明的 artifact 文件哈希，写入产物锁。
每个命令前后检查源码与产物，检查所有 rank 的结果身份。声明的 artifacts
必须在 build 目录内；不接受越界软链接。环境和结果原始文件保存在 run 中。

correctness 和 benchmark 成功执行时返回 0。数值不合格通过 checks=false
表达，依然写完整 JSON；程序故障/超时与 completed negative judgment 分开。

```json
{
  "backend": "cpu-synthetic",
  "workload_id": "cpu-fixture-v1",
  "scope": "synthetic-controller-only",
  "unit": "synthetic_us",
  "metric": "synthetic_latency",
  "ranks": [
    {
      "rank": 0,
      "source_hash": "实际源码指纹",
      "artifacts": {"kernel.json": "实际产物 SHA256"},
      "checks": {"numerical_output": true},
      "value": 10.0
    }
  ]
}
```

示例是结构说明，实际 ranks 必须与 recipe 完全匹配且无重复。每个 rank
必须提供非空 boolean checks；benchmark 的 primary_rank 必须提供有限正 value，
不能为 bool。没有计时职责的 FFN rank 可以省略 value 或设为 null，仍须完成全部
身份与数值检查。其他 rank 若提供 value，也必须是有限正数。最终只取冻结主 rank。

合成 fixture 的 artifacts 和源身份只能证明演练流程。在 Ascend benchmark 中，
必须实际检查加载的 native 库、包路径、有效配置、设备与计时范围。不能只原样
回显环境变量就宣称真实运行身份已经验证。仓库提供真实
`benchmarks/native/campaign_driver.py` 和 `recipes/ascend_eager.toml`；参见
[自动搜索](automatic-search.md) 与 [eager 原生实验](eager-native.md)。

真实模型/服务测试既可通过 `afd-rsi service` 独立运行，也可通过 recipe 的 service
表接入每个 side 的构建/验证流程；两者使用同一固定版本 AFD E2E。服务阶段保存
其生命周期 summary、库映射和原始日志，不返回本页算子级 rank 指标。
service.require_clean_shutdown 默认 true；设为 false 只豁免关闭阶段错误，仍要求
推理、全部测试、加载身份、清理通过，并原样保留失败的 shutdown summary。

一次 rank 结果是一次分布式 trial 的组成部分，不是额外独立样本。一次 benchmark
调用应提供其预热后测量的明确定义值。当前汇总为多轮 trial 的中位数；初筛和
独立复测使用不同 attempt；自动搜索仅在两次完整 PASS 后晋升最佳版本。

输入与参考、warmup 规则、全部有效开关应由冻结的 benchmark 正文确定。当前
recipe 不实现多 case 权重、置信区间或 P99 样本门槛。需要这些语义时先扩展
独立评测合同，不能隐藏在某个候选中。
