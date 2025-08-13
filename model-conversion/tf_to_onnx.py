import argparse
from pathlib import Path
import subprocess


def from_saved_model(saved_model_dir: str, out_path: str, opset: int) -> None:
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        saved_model_dir,
        "--output",
        out_path,
        "--opset",
        str(opset),
    ]
    subprocess.check_call(cmd)


def from_keras(keras_path: str, out_path: str, opset: int) -> None:
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    try:
        import tf2onnx  # noqa: F401
        # 尝试使用 Python API（较新版本提供）
        if hasattr(tf2onnx, "convert") and hasattr(tf2onnx.convert,
                                                   "from_keras"):
            model_proto, _ = tf2onnx.convert.from_keras(
                keras_path,
                output_path=out_path,
                opset=opset,
            )
            return
        raise AttributeError
    except Exception:
        # 回退到 CLI
        cmd = [
            "python",
            "-m",
            "tf2onnx.convert",
            "--keras",
            keras_path,
            "--output",
            out_path,
            "--opset",
            str(opset),
        ]
        subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 TensorFlow2 模型导出为 ONNX")
    parser.add_argument("--saved_model", help="SavedModel 目录")
    parser.add_argument("--keras", help="Keras 模型文件路径（.h5 或 .keras）")
    parser.add_argument("--out", required=True, help="输出 .onnx 路径")
    parser.add_argument("--opset",
                        type=int,
                        default=17,
                        help="ONNX opset 版本，默认 17")
    args = parser.parse_args()

    if not args.saved_model and not args.keras:
        raise ValueError("需要提供 --saved_model 或 --keras 之一")

    if args.saved_model:
        from_saved_model(args.saved_model, args.out, args.opset)
    else:
        from_keras(args.keras, args.out, args.opset)

    print(f"已导出 ONNX: {args.out}")


if __name__ == "__main__":
    main()
