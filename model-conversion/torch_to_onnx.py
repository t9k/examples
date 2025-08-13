import argparse
from pathlib import Path


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    parser = argparse.ArgumentParser(description="将 SimpleCNN(PyTorch) 模型导出到 ONNX（适配 generate_torch_model.py）")
    parser.add_argument("--ckpt", default="simple_cnn.pth", help="模型参数文件(.pth)，默认 simple_cnn.pth")
    parser.add_argument("--out", required=True, help="输出 .onnx 路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本，默认 17")
    parser.add_argument("--batch", type=int, default=1, help="导出时的 batch size，默认 1")
    parser.add_argument("--height", type=int, default=28, help="输入高度，默认 28")
    parser.add_argument("--width", type=int, default=28, help="输入宽度，默认 28")
    args = parser.parse_args()

    model = SimpleCNN()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(args.batch, 1, args.height, args.width)

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        args=(dummy,),
        f=args.out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    print(f"已导出 ONNX: {args.out}")


if __name__ == "__main__":
    main()
