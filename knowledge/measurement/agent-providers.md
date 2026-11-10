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
- 旧协议无服务端会话：每轮把请求内联进提示，按 `AFD_RSI_DEEPSEEK_MAX_CHARS`
  裁剪；默认沿用请求冻结的 `max_context_bytes` 数值作为字符上限，旧请求缺少该字段
  时回退到 200000 字符。显式环境变量优先。裁剪顺序是日志、知识正文、可省略的源码、较旧证据，
  被删内容写进请求的 `context_truncation`，模型知道缺了什么。
- JSON 由提示约束并本地校验；首次不合规会重试一次，仍不合规即失败并保留原始响应。

## 紧凑模型协议 v2

新 campaign 的 request.model_protocol=2。原始请求仍包含完整源码、SDK、证据和哈希，
验收方仍接收标准 proposal 与 git diff；只有模型可见的视图和模型响应变小。

- 文件用 s/d/e/k ID 表示源码、SDK、完整证据和知识。目录包含路径、权限、行数和符号位置。
  源码原文用 FILE/END 块传输，不进行二进制压缩或删除代码/注释。SDK、知识及请求内完整证据
  均保留可寻址原文，不让模型根据残缺契约猜测实现。
- HTTP 分析输入包含请求中的完整源码闭包、MC2/生成的 activation、阶段/等待/重叠汇总。
  请求内原有采样 event 数组留在 e 文件，必要时补读；汇总数值不变。完整设备 trace 仍在原有
  归档路径，不通过文件 ID 接口读取。通用 Catlass 与知识先发目录。
  稳定源码/契约位于提示前部，变化的历史、指标和 schema 位于后部，便于网关复用前缀缓存。
  可返回 {"read":[["d0",1,100]]} 补读一次：最多八段、每段 400 行、合计 64 KiB。
  补读与格式/截断恢复共用分析阶段的两次调用上限，补读后必须给最终分析或 needs_context。
  不增加派发预算，也不增加阶段调用上限。
- 分析仍输出可证伪实验，并用 files 指定编码所需文件 ID。编码仅收到所选文件、补读的 SDK
  片段和分析结果，不重复整份 SDK、profiling 与知识。Codex 原生只读工具按同一 ID 清单
  读取工作区文件；证据先读 summary 文件，再按需读取完整事实。
- 编码响应只含当前 base、提案 id 和 edits(file,old,new)。元数据从该编号的分析恢复，
  不让编码模型重复 hypothesis/category/evidence/experiment。old 必须在原 parent 中唯一匹配，
  修改不能重叠。new:路径表示新文件，new=null 且 old 为完整原文表示删除。
  适配器生成 diff 并预检查；旧文本歧义、越权、未知 ID 或旧 base 均拒绝。
- debug 继续只有一次编码调用。提供原失败补丁、日志、源码与之前按源/SDK 哈希绑定的补读；
  不启动额外分析或补读。修复仍返回针对原 parent 的完整修改。

保存原始模型响应、派生输入和标准 proposal。HTTP usage 逐次记录实际 prompt/completion
tokens，以及输入/输出 UTF-8 字节数；字节数不是 tokenizer 估计。离线比较使用
tools/measure_model_protocol.py，真实成本与提案质量仍以网关 usage 和实际搜索为准。
旧冻结请求未声明 v2 时沿用旧协议。

## 共同约束

- 适配器只读请求与仓库内知识，不接触真实源码树、设备或验收流程。
- 提案必须引用请求中已存在的证据 ID；实验字段与必引证据由控制器复核。
- 模型无权修改门槛、允许文件、知识或预算。
- 提示与响应原文、用量记录在提案目录，供事后复核；认证信息不落盘。
