#!/bin/bash

python -m fastchat.serve.controller --host 0.0.0.0 &
python -m fastchat.serve.model_worker --host 0.0.0.0 --model-names "$1" --model-path "$1" &
python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 80
