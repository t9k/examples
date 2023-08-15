== sft ==

eval:

python src/train_bash.py --stage sft --model_name_or_path /t9k/mnt/models/Llama-2-7b-hf --do_eval --dataset alpaca_en --template llama2 --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/llama2/7b/ --output_dir /t9k/mnt/output/sft-eval/llama2/7b/ --per_device_eval_batch_size 8 --max_samples 100 --predict_with_generate

chat:

python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Llama-2-7b-hf --template llama2 --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/llama2/7b/

export:

python src/export_model.py --model_name_or_path /t9k/mnt/models/Llama-2-7b-hf --template llama2 --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/llama2/7b/ --output_dir /t9k/mnt/output/sft-models/llama2/7b/

== ppo ==






