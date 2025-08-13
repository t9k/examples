# 模型格式转换

一组用于常见 LLM 权重与格式转换的脚本集合。涵盖：

1. safetensors ⇄ PyTorch（state_dict .bin/.pt/.pth）
2. PyTorch（state_dict .bin/.pt/.pth）⇄ ONNX（model specific）
3. TensorFlow（SavedModel）→ ONNX：
4. safetensors → GGUF（调用 llama.cpp 官方转换脚本）

## 依赖

按需安装，对应脚本所需的最小依赖如下：

- 通用：
  - Python 3.9+
- safetensors ⇄ PyTorch（.bin/.pt/.pth）：
  - `pip install torch safetensors`
- PyTorch（.bin/.pt/.pth）⇄ ONNX：（model specific）
  - `pip install torch onnx`
- TensorFlow（SavedModel）→ ONNX：
  - `pip install tensorflow onnx tf2onnx`
- safetensors → GGUF：
  - 需要克隆 `llama.cpp` 仓库
  - `pip install sentencepiece transformers`

## 使用示例

- 生成 SimpleCNN（PyTorch）权重：
  ```bash
  python generate_torch_model.py
  # 将生成 simple_cnn.pth
  ```

- 生成 SimpleCNN（TensorFlow）权重：
  ```bash
  python generate_tf_model.py
  # 将生成 simple_cnn_tf_savedmodel/ 目录
  ```

- safetensors → PyTorch：
  ```bash
  python st_to_torch.py \
    --src /path/to/hf_model_dir \
    --out /path/to/out_dir
  ```

- PyTorch → safetensors：
  ```bash
  python torch_to_st.py \
    --src /path/to/pt_ckpt_dir_or_file \
    --out /path/to/out_dir
  ```

- PyTorch → ONNX：（SimpleCNN）
  ```bash
  python torch_to_onnx.py \
    --ckpt /path/to/model.pth \
    --out /path/to/model.onnx
  ```

- ONNX → PyTorch：（SimpleCNN）
  ```bash
  python onnx_to_torch.py \
    --src /path/to/model.onnx \
    --out /path/to/model_from_onnx.pth \
  ```

- TensorFlow → ONNX：
  ```bash
  # 从 SavedModel 目录导出
  python tf_to_onnx.py \
    --saved_model /path/to/saved_model_dir \
    --out /path/to/model.onnx \
  ```

- HF → GGUF（通过 llama.cpp 转换脚本）：
  ```bash
  modelscope download --model "Qwen/Qwen3-0.6B" --local_dir "./Qwen3-0.6B"
  git clone https://github.com/ggml-org/llama.cpp.git
  python llama.cpp/convert_hf_to_gguf.py ./Qwen3-0.6B
  ```

## 备注

- ONNX → PyTorch 的映射依赖 ONNX initializer 名与 PyTorch 参数名的对应关系；脚本提供了简单的候选名规则，必要时请按实际 ONNX 图调整映射逻辑。
- 大模型权重转换会占用大量内存与磁盘，请确保充足的资源。
- 建议在虚拟环境中按需安装依赖。
