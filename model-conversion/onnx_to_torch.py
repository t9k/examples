import argparse
from pathlib import Path
from typing import Dict, List

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper


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
		return self.fc2(x)


def load_onnx_initializers(onnx_path: str) -> Dict[str, torch.Tensor]:
	model = onnx.load(onnx_path)
	weights: Dict[str, torch.Tensor] = {}
	for init in model.graph.initializer:
		arr = numpy_helper.to_array(init)
		weights[init.name] = torch.from_numpy(arr)
	return weights


def candidate_names(param_name: str) -> List[str]:
	# 生成若干候选名以尽可能匹配 ONNX initializer 的命名
	names = [param_name]
	names.append(param_name.replace(".", "_"))
	names.append(param_name.replace(".", "/"))
	return list(dict.fromkeys(names))  # 去重并保持顺序


def map_weights_to_state_dict(onnx_weights: Dict[str, torch.Tensor], state_dict: Dict[str, torch.Tensor], strict: bool) -> Dict[str, torch.Tensor]:
	mapped = 0
	skipped: List[str] = []
	for k, v in state_dict.items():
		matched = False
		for cand in candidate_names(k):
			if cand in onnx_weights:
				w = onnx_weights[cand]
				if tuple(w.shape) != tuple(v.shape):
					# 形状不一致，跳过
					continue
				state_dict[k] = w.to(dtype=v.dtype)
				mapped += 1
				matched = True
				break
		if not matched:
			skipped.append(k)

	print(f"已映射参数个数: {mapped}/{len(state_dict)}")
	if skipped:
		print("未匹配参数:")
		for name in skipped:
			print(" -", name)
		if strict and skipped:
			raise RuntimeError("部分参数未能从 ONNX 映射。可关闭 --strict 放行。")
	return state_dict


def main() -> None:
	parser = argparse.ArgumentParser(description="从 ONNX 解析权重并映射到 SimpleCNN，导出为 PyTorch .pth")
	parser.add_argument("--src", required=True, help="输入 .onnx 文件路径")
	parser.add_argument("--out", required=True, help="输出 .pth 文件路径")
	parser.add_argument("--strict", action="store_true", help="严格模式：若存在未映射参数则报错")
	args = parser.parse_args()

	onnx_weights = load_onnx_initializers(args.src)

	model = SimpleCNN()
	sd = model.state_dict()
	sd = map_weights_to_state_dict(onnx_weights, sd, strict=args.strict)
	model.load_state_dict(sd, strict=False)

	Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), args.out)
	print(f"已保存为 {args.out}")


if __name__ == "__main__":
	main()
