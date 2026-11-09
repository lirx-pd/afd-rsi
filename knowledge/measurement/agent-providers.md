# 提案模型提供方

控制器只接受"提案"，不接受模型给出的判定。换模型只改变谁写补丁，
数值、计时、双 PASS 晋升和证据哈希全部不变。

## 选择方式

配方 `[search]` 用一个字段切换，冻结进 plan.lock：

```toml
[search]
provider = "codex"     # 默认；也可写 "deepseek"
```

声明后适配器命令自动补齐为 `{foundation}/benchmarks/agents/<provider>_agent.py`；
也可以继续手写完整 `agent_command`（旧配方的写法），两者必须指向同一个适配器，
不一致在 prepare 阶段就报错。请求里会记录所选适配器的相对路径与哈希，
`propose` 与控制器都会复核——换了模型却沿用旧请求会被拒绝，而不是静默降级。

## codex

- 适配器：`benchmarks/agents/codex_agent.py`。
- 依赖已登录的 Codex CLI，沿用其 provider 配置，不复制认证文件。
- 模型与推理强度固定：分析 `gpt-6-astra` / `xhigh`，编码 `gpt-6.1-sol` / `xhigh`。
- 同一 campaign 复用同一个 thread，跨机通过文件交换。

## deepseek

- 适配器：`benchmarks/agents/deepseek_agent.py`，仅用标准库，不需要额外安装包。
- 三个输入必须和发起提案的那台机器上既有的模型网关配置一致：网关根地址、
  凭据变量、模型标识。适配器不内置任何一项——凭空给的默认值正是"调用一个
  根本不服务该模型的地址"的来源。
  - 网关根地址：`AFD_RSI_DEEPSEEK_BASE_URL`，含 API 前缀（如 `/v1`）。
  - 凭据变量：`AFD_RSI_DEEPSEEK_KEY_ENV` 指定变量名（默认 `DEEPSEEK_API_KEY`），
    值本身只存在提案机环境里，不写入仓库、请求或 campaign。
  - 模型标识：`[search.models]` 或 `AFD_RSI_DEEPSEEK_MODEL`，默认
    `DeepSeek-V4.1-Flash-1017`。
- 先用 `python -m afd_rsi probe`（提案机不是 Linux 时用
  `python benchmarks/agents/deepseek_agent.py --check`）验证。它按顺序报告：
  配置是否齐全、网关是否可达、网关自己的模型清单里有没有该标识、能否真正生成、
  JSON 输出是否被接受；第一个失败项直接给出该改哪一个变量。
  `401` 是凭据问题，`404` 多为地址缺 API 前缀，`503` 是网关自身无健康上游——
  属于可用性问题，不是配置问题。
- 思考等级按阶段分开：分析优化方向是最需要推理的一步，用最高等级 `max`；在已定
  方向下写补丁是收窄的工作，用 `high`。通过请求的 `reasoning_effort` 传给网关，
  记录在 `usage.json` 每轮和 `session.json` 里。可用
  `AFD_RSI_DEEPSEEK_EFFORT_ANALYSIS`、`AFD_RSI_DEEPSEEK_EFFORT_CODE` 按主机覆盖。
  网关若拒绝该参数，适配器只去掉被拒绝的那一个并记录，不会连带关闭 JSON 输出。
- 默认模型：`DeepSeek-V4.1-Flash-1017`，分析与编码两个阶段都用同一个模型；
  两阶段仍然分开，因为区分的是"只出假设"和"出补丁"的要求，不是模型。
- 按阶段替换：配方里声明 `[search.models]` 后，两个阶段各自冻结，只写一个阶段时
  另一个取默认。以后有新模型时只替换它擅长的那个阶段：

  ```toml
  [search]
  provider = "deepseek"

  [search.models]
  analysis = "DeepSeek-V4.1-Flash-1017"
  code = "以后的编码模型"
  ```

  该表必须写在文件末尾，不能插在 `[search]` 的其他键中间，否则后续键会归属它。
- 冻结值优先于环境变量：`AFD_RSI_DEEPSEEK_MODEL`（同时替换两阶段）与
  `AFD_RSI_DEEPSEEK_ANALYSIS_MODEL`、`AFD_RSI_DEEPSEEK_CODE_MODEL`（按阶段）
  只对没有声明 `[search.models]` 的 campaign 生效，避免环境悄悄改变已冻结的实验。
- codex 的模型策略固定在本仓库（见 [codex models](codex-models.md)），
  不接受 `[search.models]`，声明即报错。
- 默认请求 JSON 输出与 temperature=0；网关不接受这些参数时，适配器会去掉它们
  重试一次并记录 `unsupported-parameters.json`，校验仍在本地进行。
  `AFD_RSI_DEEPSEEK_JSON_MODE=0` 可完全关闭。
- 网关的报错正文可能回显提交的凭据，适配器在写入任何报告前先脱敏。
- 无服务端会话：每轮把请求内联进提示，按 `AFD_RSI_DEEPSEEK_MAX_CHARS`
  裁剪；默认沿用请求冻结的 `max_context_bytes` 数值作为字符上限，旧请求缺少该字段
  时回退到 200000 字符。显式环境变量优先。裁剪顺序是日志、知识正文、可省略的源码、较旧证据，
  被删内容写进请求的 `context_truncation`，模型知道缺了什么。
- JSON 由提示约束并本地校验；首次不合规会重试一次，仍不合规即失败并保留原始响应。

## 共同约束

- 适配器只读请求与仓库内知识，不接触真实源码树、设备或验收流程。
- 提案必须引用请求中已存在的证据 ID；实验字段与必引证据由控制器复核。
- 模型无权修改门槛、允许文件、知识或预算。
- 提示与响应原文、用量记录在提案目录，供事后复核；认证信息不落盘。
