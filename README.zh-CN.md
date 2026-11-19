# 模型量化保真度评估

[English](README.md) | [简体中文](README.zh-CN.md)

通过 OpenAI 兼容的模型 server，比较参考模型与量化候选模型的下一 token 概率和短续写一致性。参考数据缓存一次，即可重复评估多个候选模型。

参考模型与候选模型需要使用兼容的 tokenizer 和 chat template。工具已用于 MiniMax 和 Qwen，评估核心只依赖 Python 标准库；模型加载、量化 kernel、补丁和 GPU 分配由模型 server 管理。

## 完整词表 KL：全部 300 条输入

默认在全部 32 个续写位置逐一计算完整词表 KL；MiniMax词表为200,064项。vLLM启动加入 `--max-logprobs -1 --logprobs-mode raw_logprobs`，参考捕获使用 `--metric full`，完整词表是唯一支持的 KL 模式，也是 CLI 默认值。下方历史表格保留原始分桶 KL 标签，仅供查阅，不能作为完整词表结果。

参考缓存按prompt压缩保存每个位置的完整原始log-probabilities。候选使用完整分布在内存中计算，保存逐位置KL、Top-1、参考Top-1的概率证据、词表项数及完整分布hash，同时保存生成token ID/text；候选完整分布计算后释放，不写入大型JSON。

## 2026-11-18：恢复 300 条保真度输入

默认测试集已加回原来的 60 条 DeepSWE prompt，恢复为五类各 60 条，共 300 条。移除的仅是独立 DeepSWE 解题 benchmark。11 月 16–18 日已完成的复测仍属于 240 条范围，加回输入不会自动改变这些历史结果。

## MiniMax LH 与 RT：公开 240 条复测

2026-11-16 重新采集同环境 MXFP8 基线，公开 240 条 prompt、全部 32 个位置、每候选 3 轮，评测错误为 0。

| 模型 | 平均分桶 KL（nat）↓ | 同前缀 Top-1↑ | 完整 32-token↑ | 三轮评测总耗时（分钟） |
|---|---:|---:|---:|---:|
| LH | 0.03843885 | 96.21% | 34.31% | 36.79 |
| RT | 0.04018061 | 95.93% | 33.89% | 40.98 |

[新结果 README：逐轮、分类与极差](examples/minimax-lh-rt-public240-20261116/README.zh-CN.md)

## MiniMax M3.1 历史测试结果

以下是此前 **300 条 prompt、每模型 3 轮**测试的原始总体结果。每条生成 32 个 token，请求全部 32 个位置的概率。原始 300 条输入现已恢复；下表仍是当时参考与环境下的历史结果，与另行捕获参考的 240 条复测分别解读。

| 模型 | 平均分桶 KL（nat）↓ | 同前缀 Top-1↑ | 完整 32-token↑ |
|---|---:|---:|---:|
| MXFP8（复测） | 0.01577440 | 97.55% | 49.56% |
| MXFP4 | 0.03158934 | 96.50% | 38.44% |
| MXFP4 RT | 0.03519367 | 96.20% | 38.67% |
| MXFP4 QADv1 | 0.02610190 | 96.86% | 44.33% |

四个候选模型在同一个后续引擎环境中评估，统一对比旧环境生成的固定 MXFP8 参考缓存。MXFP8 这一行是复测，因此它与参考缓存也不完全一致。每模型三轮合计 900 条生成序列、28,800 个同前缀位置，各轮均完整完成，评估错误为 0。

相对普通 MXFP4，QADv1 的平均分桶 KL 降低 **17.37%**，同前缀 Top-1 提高 **0.36 个百分点**，完整 32-token 一致率提高 **5.89 个百分点**。QADv1 三轮评估耗时合计 **44.81 分钟**，不含服务启动。这些结果反映历史输入上的保真度变化，不能直接当作任务答题正确率。

[原始总体结果与口径](docs/minimax-results.md) · [按公开类别重新汇总的结果](docs/results.md)

## 快速开始

```bash
python -m pip install -e .

# 检查参考与候选 server 的接口能力。
m31-eval doctor --base-url http://localhost:8000/v1 --probe \
  --output cache/reference-doctor.json
m31-eval doctor --base-url http://localhost:8001/v1 --probe \
  --output cache/candidate-doctor.json

# 缓存参考模型，明确评分全部 32 个位置。
m31-eval reference \
  --dataset data/mixed/quick300.jsonl \
  --base-url http://localhost:8000/v1 \
  --reference-kind mxfp8 \
  --metric full --tokens 32 --positions {0..31} \
  --max-prompt-tokens 65536 --concurrency 2 \
  --time-budget 1800 --output cache/reference

# 将候选模型与同一份固定参考比较。
m31-eval evaluate \
  --reference cache/reference \
  --base-url http://localhost:8001/v1 \
  --concurrency 2 --time-budget 1800 \
  --output logs/candidate-r1
```

例如：在 8000 端口启动 MXFP8、在 8001 端口启动 MXFP4，先缓存 MXFP8，再评估 MXFP4。BF16 和应用算法后的模型使用相同接入方法，并按实际参考设置 `--reference-kind`。

以上命令使用 Bash 的 `{0..31}` 展开；其他 shell 请显式列出 0 到 31。CLI 默认评分全部 32 个位置；该参数也可显式设置评分位置。一个接口提供多个模型时，用 `--model` 指定模型名。认证默认读取 `OPENAI_API_KEY`，也可通过 `--api-key-env` 指定其他环境变量。

## Server 接口要求

token 级保真度需要原生 `/tokenize`、`/detokenize`、`/v1/completions` 接口，返回 token ID 和原始下一 token logprobs。每个评分位置必须返回完整词表概率。

vLLM 使用 `--max-logprobs -1 --logprobs-mode raw_logprobs`。正式评估前，用 `doctor --probe-full --expected-vocab-size 200064` 检查 `tokenize` 和 `full_vocabulary_logprobs`；其他模型按 config 设置词表大小。只有 Chat 接口的代理无法计算 token KL；`chat`、`responses` 模式计算的是消息一致性。

工具只连接已有 server，不负责启动服务、安装模型补丁或配置量化 kernel。

## 指标定义

| 指标 | 含义 | 方向 |
|---|---|---|
| 平均完整词表 KL（nat） | 在相同参考前缀下，对全部词表求和计算 KL，再按位置与 prompt 平均 | 越低越好 |
| 同前缀 Top-1 一致率 | 使用同一输入和参考续写前缀时，最高概率 token 相同的位置比例 | 越高越好 |
| 完整 32-token 一致率 | 两个模型分别独立生成，32 个 token ID 全部相同的输入比例 | 越高越好 |

每个位置按完整词表计算 `sum(P(v) * (log P(v) - log Q(v)))`。先对词表 token 的贡献求和，再对每条 prompt 的评分位置平均，最后对 prompt 平均，得到 `prompt_mean_kl`。300 条输入、每条 32 个位置，完整结果应包含 9,600 个成功评分位置。

生成使用 greedy decoding（`temperature=0`），概率评分使用 raw logprobs。参考分布接近确定的位置也参与 KL。缺少 token ID、概率总质量异常或候选对参考正概率 token 给出零概率会记为错误，不静默跳过；不完整运行不能作为完整结果发布。

原生 summary 中的部分指标采用原始 task/document 等权汇总。展示直接按位置计数的 Top-1 时，应统计 `samples.jsonl` 中的 `top1_agreement`，不能用 task 等权均值替代。详细公式和输出字段见[指标说明](docs/metrics.md)。

## 固定公开数据集

默认 `data/mixed/quick300.jsonl` 提供 **300 条 prompt，5 类各 60 条**；保留 `quick240.jsonl` 用于复现四类历史复测：

| 类别 | 数量 | 来源 |
|---|---:|---|
| DeepSWE | 60 | 原始任务及 agent 上下文输入快照 |
| 终端任务 | 60 | Terminal-Bench 4.0 原始任务 instruction |
| 数学 | 60 | NVIDIA OpenMathReasoning problem |
| 非拉丁语 | 60 | Global-MMLU-Lite：阿拉伯语、孟加拉语、印地语、日语、韩语、中文各 10 条 |
| 长文档 | 60 | CPython 文档和 Project Gutenberg 文本 |

共有 260 个原始 task/document。长文档部分从 20 份文档各取 3 个窗口；这些窗口不是 60 份独立文档。development/holdout 为 240/60 条，同一文档的窗口放在同一 split。该划分没有证明与模型实际校准集或训练集不重叠。

长文档窗口的 32K/64K/128K 指**字符数**，不是 token 数；正式评估前应使用当前模型的 tokenizer 做长度审计。原始来源、固定版本、转换方式和许可见[data/mixed](data/mixed)与[数据集说明](docs/dataset.md)。

恢复的 300 条与保留的 240 条数据 hash 不同，240 条参考缓存无法评估额外 60 条；新一批 300 条比较需要重新捕获并冻结 300 条参考。也可以通过 `--dataset` 接入自己的 JSONL，数量和领域配额不限于当前公开版。

## 多轮测试与候选对比

固定 server 参数、输入和参考缓存，每轮使用不同输出目录：

```bash
m31-eval evaluate --reference cache/reference \
  --base-url http://localhost:8001/v1 --concurrency 2 \
  --output logs/candidate-r2

m31-eval evaluate --reference cache/reference \
  --base-url http://localhost:8001/v1 --concurrency 2 \
  --output logs/candidate-r3

m31-eval compare --old logs/candidate-r1 --new logs/improved-r1 \
  --aggregation domain_macro --output logs/comparison.json
```

建议先复测参考模型，观察 A/A 的数值波动。比较算法时保持引擎版本和服务参数一致，同时查看分类结果、KL 覆盖率和置信区间。保真度提升是快速筛选信号，真实任务能力仍需独立验证。

## 保存的数据与历史结果

评估保存 `run.json`、`summary.json` 和逐条 `samples.jsonl`，包含生成 token ID/text、逐位置概率证据、KL、Top-1 和来源标识。本地缓存与运行输出默认不提交到 Git。

[历史公开类别结果](docs/results.md)仅保留数值汇总，由旧实验中保留的 4 个公开类别重新计算，不是使用新参考缓存重新测试。它与 README 中的原始 300 条总体结果采用不同数据范围，应分别解读。

## 方法背景

本项目参考 [Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs#divergence-300-32) 的多领域输入、短续写和参考模型比较策略。具体输入、完整词表 KL 和报告口径由本项目定义，不是对 Unsloth 未公开的 300 条清单及 Divergence-300@32 完整计分公式的精确复现。

Harbor 是评估框架与 registry，Terminal-Bench 是可以在其中运行的 benchmark。终端类别采用 Terminal-Bench 4.0 输入，DeepSWE 类别采用恢复的任务及 agent 上下文输入；它们都通过 server API 测保真度，不运行完整任务容器和 verifier。

## 开发验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
