# DeepEval - MMLU 评测脚本

本目录包含一个使用 DeepEval 在 MMLU 基准上评测 OpenAI 兼容 API 模型的脚本。

## 安装依赖

```bash
pip install -U deepeval
```

## 使用方式

- 需要提供 OpenAI 兼容的 API Base（如 `https://api.openai.com/v1` 或自建兼容地址）。
- API Key 默认使用环境变量 `OPENAI_API_KEY`，也可通过 `--api_key` 显式传入。

```bash
python evaluate_mmlu.py \
  --api_base https://your-openai-compatible-endpoint/v1 \
  --model your-model-name \
  --api_key sk-... \
  --shots 0 \
  --tasks all \
  --limit_per_task 50
```

参数说明：
- `--api_base`: OpenAI 兼容 API Base URL。
- `--model`: 模型名称。
- `--api_key`: API Key（可选，不传则使用 `OPENAI_API_KEY`）。
- `--shots`: few-shot 样本数（默认 0）。
- `--tasks`: 逗号分隔的 `MMLUTask` 名称或 `all`。默认仅跑一小部分任务：`high_school_computer_science,astronomy`。
- `--limit_per_task`: 每个任务的样本上限（不同 deepeval 版本可能不支持，将自动忽略）。

> 注意：全量 MMLU 任务较大，建议先用小子集/较小的 `--limit_per_task` 验证流程。

## 常见问题
- 若提示无法导入 `deepeval` 或 `MMLU`/`MMLUTask`，请升级 `deepeval`：`pip install -U deepeval`。
- 若使用自建 OpenAI 兼容服务，请确保其实现了 `chat.completions` 接口，并正确设置 `OPENAI_API_BASE` 与 `OPENAI_API_KEY`。
