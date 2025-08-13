import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file as st_save_file


def _discover_files(root: str, patterns: Tuple[str, ...]) -> List[Path]:
    p = Path(root)
    if p.is_file():
        return [p]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(p.glob(pattern)))
    return files


def load_torch_bins_as_state_dict(path: str) -> Dict[str, torch.Tensor]:
    files = _discover_files(path, ("*.bin", "*.pt", "*.pth"))
    if not files:
        raise FileNotFoundError(f"未找到 PyTorch 权重文件(.bin/.pt/.pth): {path}")
    state_dict: Dict[str, torch.Tensor] = {}
    for f in files:
        obj = torch.load(str(f), map_location="cpu")
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                part = obj["state_dict"]
            else:
                part = obj
        else:
            raise ValueError(f"无法识别的 PyTorch 权重结构: {f}")
        conflicts = set(state_dict.keys()) & set(part.keys())
        if conflicts:
            raise ValueError(f"键冲突: {conflicts} in {f}")
        state_dict.update(part)
    return state_dict


def save_state_dict_as_safetensors(state_dict: Dict[str, torch.Tensor], out_file: str) -> None:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_save_file(state_dict, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PyTorch .bin/.pt 合并并导出为 Hugging Face safetensors")
    parser.add_argument("--src", required=True, help="PyTorch 权重文件或目录（支持分片 .bin/.pt/.pth）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--outfile", default="model.safetensors", help="输出文件名，默认 model.safetensors")
    args = parser.parse_args()

    state_dict = load_torch_bins_as_state_dict(args.src)

    out_path = str(Path(args.out) / args.outfile)
    save_state_dict_as_safetensors(state_dict, out_path)
    print(f"已导出: {out_path}")


if __name__ == "__main__":
    main()
