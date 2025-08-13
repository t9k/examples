import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file as st_load_file


def _discover_files(root: str, patterns: Tuple[str, ...]) -> List[Path]:
    p = Path(root)
    if p.is_file():
        return [p]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(p.glob(pattern)))
    return files


def load_safetensors_as_state_dict(path: str) -> Dict[str, torch.Tensor]:
    files = _discover_files(path, ("*.safetensors",))
    if not files:
        raise FileNotFoundError(f"未找到 safetensors 文件: {path}")
    state_dict: Dict[str, torch.Tensor] = {}
    for f in files:
        part = st_load_file(str(f))
        conflicts = set(state_dict.keys()) & set(part.keys())
        if conflicts:
            raise ValueError(f"键冲突: {conflicts} in {f}")
        state_dict.update(part)
    return state_dict


def save_state_dict_as_torch_bin(state_dict: Dict[str, torch.Tensor], out_file: str) -> None:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Hugging Face safetensors 合并并导出为 PyTorch .bin")
    parser.add_argument("--src", required=True, help="HF 模型目录，或单个/分片 .safetensors 文件所在路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--outfile", default="pytorch_model.bin", help="输出文件名，默认 pytorch_model.bin")
    args = parser.parse_args()

    state_dict = load_safetensors_as_state_dict(args.src)

    out_path = str(Path(args.out) / args.outfile)
    save_state_dict_as_torch_bin(state_dict, out_path)
    print(f"已导出: {out_path}")


if __name__ == "__main__":
    main()
