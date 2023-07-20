pip install nltk
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings do"--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd .. && rm -rf apex

python ~/Megatron-LM/tools/preprocess_data.py \
       --input wiki-en/all \
       --output-prefix wiki-en/gpt \
       --vocab-file ../tokenizer/wiki-en-tokenizer/vocab.json \
       --merge-file ../tokenizer/wiki-en-tokenizer/merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --workers 16
