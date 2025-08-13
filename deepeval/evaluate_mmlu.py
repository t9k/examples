import os
import argparse
import sys
from urllib.parse import urlparse


def parse_tasks_arg(tasks_arg):
    try:
        from deepeval.benchmarks.tasks import MMLUTask
    except Exception as e:
        print("[ERROR] 无法导入 deepeval 的 MMLU 任务枚举，请确认已安装 deepeval。", file=sys.stderr)
        raise

    if tasks_arg.lower() in {"all", "*"}:
        try:
            # Python Enum is iterable; convert to list of all tasks
            return list(MMLUTask)
        except Exception:
            # 兼容性兜底：常见的一些子集，避免一次性跑全量
            return [
                MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
                MMLUTask.ASTRONOMY,
                MMLUTask.BIOLOGY,
                MMLUTask.CHEMISTRY,
            ]

    # 支持逗号分隔的枚举名（大小写不敏感）
    selected = []
    name_to_member = {m.name.lower(): m for m in MMLUTask}
    for name in [n.strip().lower() for n in tasks_arg.split(",") if n.strip()]:
        if name not in name_to_member:
            valid = ", ".join(sorted(m.name for m in MMLUTask))
            raise ValueError(f"未知任务 '{name}'. 可选: {valid}")
        selected.append(name_to_member[name])
    if not selected:
        raise ValueError("--tasks 至少指定一个任务，或使用 'all'")
    return selected


def main():
    parser = argparse.ArgumentParser(description="使用 DeepEval 在 MMLU 基准上评测 OpenAI 兼容模型")
    parser.add_argument("--api_base", required=True, help="OpenAI 兼容 API Base，如 https://api.openai.com/v1 或自建兼容地址")
    parser.add_argument("--model", required=True, help="模型名称，如 gpt-4o-mini 或本地兼容服务中的模型名")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API Key（可选，不传则使用环境变量 OPENAI_API_KEY）")
    parser.add_argument("--shots", type=int, default=0, help="few-shot 样本数 n_shots，默认 0")
    parser.add_argument("--tasks", default="high_school_computer_science,astronomy", help="评测任务，逗号分隔的 MMLUTask 名称，或 'all'")
    parser.add_argument("--limit_per_task", type=int, default=None, help="每个任务的样本上限（可选，若 deepeval 版本不支持将被忽略）")

    args = parser.parse_args()

    # 设置 OpenAI 兼容参数到环境变量，DeepEval/各类 OpenAI 兼容客户端通常读取这些变量
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_API_BASE"] = args.api_base
    # 兼容不同实现/SDK 的环境变量名
    os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["OPENAI_BASE"] = args.api_base

    # 若是本机地址，确保不走代理
    try:
        parsed = urlparse(args.api_base)
        host = parsed.hostname or ""
        if host in {"127.0.0.1", "localhost"}:
            no_proxy = os.environ.get("NO_PROXY", "")
            tokens = {h.strip() for h in no_proxy.split(",") if h.strip()}
            tokens.update({"127.0.0.1", "localhost"})
            os.environ["NO_PROXY"] = ",".join(sorted(tokens))
            os.environ["no_proxy"] = os.environ["NO_PROXY"]
            # 避免企业代理截获本地流量
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                if host in {"127.0.0.1", "localhost"}:
                    os.environ.pop(k, None)
    except Exception:
        pass

    # 预检连通性（可快速给出明确错误）
    try:
        import requests
        resp = requests.get(
            args.api_base.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=5,
        )
        # 200/401/403 都视为服务可达
        if resp.status_code not in {200, 401, 403}:
            print(f"[WARN] 预检 /models 返回状态码 {resp.status_code}，继续尝试评测…", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] 无法连接到 {args.api_base}（预检失败: {e}）。请确认本地服务已启动并监听该地址。", file=sys.stderr)
        sys.exit(2)

    try:
        from deepeval.benchmarks import MMLU
        from deepeval.benchmarks.tasks import MMLUTask
        from deepeval.models.llms.local_model import LocalModel
    except Exception as e:
        print("[ERROR] 导入 deepeval 失败，请先安装: pip install -U deepeval", file=sys.stderr)
        raise

    try:
        selected_tasks = parse_tasks_arg(args.tasks)
    except Exception as e:
        print(f"[ERROR] 解析 --tasks 失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 构建基准
    try:
        benchmark_kwargs = {
            "tasks": selected_tasks,
            "n_shots": args.shots,
        }
        # 使用正确的入参名以限制每任务题目数量
        if args.limit_per_task is not None:
            benchmark_kwargs["n_problems_per_task"] = args.limit_per_task

        benchmark = MMLU(**benchmark_kwargs)
    except TypeError:
        # 回退：移除可能不被支持的参数
        benchmark = MMLU(tasks=selected_tasks, n_shots=args.shots)

    # 运行评测：构造本地 OpenAI 兼容模型适配器
    class CompatLocalModel(LocalModel):
        def generate(self, prompt: str, schema: None | object = None):
            # 让 MMLU 走回退路径（不要求结构化输出）
            if schema is not None:
                raise TypeError("Schema not supported in CompatLocalModel")
            return super().generate(prompt, schema=None)

    model = CompatLocalModel(model=args.model, base_url=args.api_base, api_key=os.environ.get("OPENAI_API_KEY", ""))
    benchmark.evaluate(model=model)

    # 输出结果
    # 兼容不同版本属性名
    overall = getattr(benchmark, "overall_score", None) or getattr(benchmark, "score", None)
    if overall is not None:
        print(f"Overall Score: {overall}")

    # 打印每个任务分数（若可用）
    per_task_df = getattr(benchmark, "task_scores", None)
    if per_task_df is not None:
        try:
            # 期望为 DataFrame，包含 ['Task', 'Score']
            rows = per_task_df.to_records(index=False)
            print("Per-task Scores:")
            for task_value, score in rows:
                print(f"  {task_value}: {score}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
