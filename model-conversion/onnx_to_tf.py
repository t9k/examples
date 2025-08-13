import argparse
from pathlib import Path


def main() -> None:
    import onnx
    try:
        from onnx_tf.backend import prepare
    except Exception as e:
        raise RuntimeError("需要安装 onnx-tf：pip install onnx-tf tensorflow onnx") from e

    parser = argparse.ArgumentParser(description="将 ONNX 转换为 TensorFlow SavedModel（基于 onnx-tf）")
    parser.add_argument("--src", required=True, help="输入 .onnx 文件路径")
    parser.add_argument("--out", required=True, help="输出 SavedModel 目录")
    args = parser.parse_args()

    onnx_model = onnx.load(args.src)

    tf_rep = prepare(onnx_model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf_rep.export_graph(str(out_dir))
    print(f"ONNX 已转换为 SavedModel: {out_dir}")


if __name__ == "__main__":
    main()
