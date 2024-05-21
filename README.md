# Acknowledgement

This repo is  for DSAA 6800 independent project. Thanks professor Jia Li and instructors in industry for your guidance. The second part is mainly based on the LLaMa Factory [hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs (github.com)](https://github.com/hiyouga/LLaMA-Factory). Thanks for their work.

# 1. Install dependencies
You can change to other sources if the installing process is slow.

```bash
git clone https://github.com/mjy2357/Toxic_content_detection.git
conda create -n Toxic_content_detection python==3.10
conda activate Toxic_content_detection
cd Toxic_content_detection
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

# 2. Using rationale augmentation Toxicn dataset to fine-tune a 0.5B model

## On the one hand, we use the original training dataset to train the model as a benchmark. The training process maybe long, so the trained LoRA adapter is also provided for use. 

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset ToxiCN_train_alpaca_fomat \
    --cutoff_len 5120 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --output_dir saves/Qwen1.5-0.5B-Chat/lora/train_on_toxicntrain_5epoch_1 \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```

## On the other hand, we use the rationale augmentation dataset (1922 samples with rationales) to train the 0.5B model and on the base of this, we use remaining 7686 samples without rationales to further train the trained model. The training process maybe long, so the trained LoRA adapters are also provided for use. 

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset ToxiCN_train_1922_withreason \
    --cutoff_len 5120 \
    --learning_rate 0.0001 \
    --num_train_epochs 8.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --output_dir saves/Qwen1.5-0.5B-Chat/lora/train_on_toxicn1922_8epoch_1 \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --adapter_name_or_path train_on_toxicn1922reason_8epoch \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset ToxiCN_train_7686 \
    --cutoff_len 5120 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --output_dir saves/Qwen1.5-0.5B-Chat/lora/train_on_toxicn1922_7686_8_5epoch_1 \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True 
```

## Then, we can use 2 trained model to inference on the test sets.

This is the code for testing model trained on original training set on Toxicn test set. To test on COLD test set, you can simply change the dataset to **COLD_test_alpaca_fomat**

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --adapter_name_or_path train_on_toxicntrain4_5epoch \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset ToxiCN_test_alpaca_fomat \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 256 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen1.5-0.5B-Chat/lora/eval_on_toxicntest_sft_on_toxicntrain_5epoch \
    --do_predict True 
```
This is the code for testing model trained on rationale augmentation training set on Toxicn test set. To test on COLD test set, you can simply change the dataset to **COLD_test_alpaca_fomat**

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --adapter_name_or_path train_on_toxicn1922reason_7686_8_5epoch \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset ToxiCN_test_alpaca_fomat \
    --cutoff_len 5120 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 256 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen1.5-0.5B-Chat/lora/eval_on_toxicntest_sft_on_toxicntrain1922reason_7686_8_5epoch \
    --do_predict True 
```
## Run this .py file to get accuracy. You may need to change the "input file path" to corresponding file path. For example "saves/Qwen1.5-0.5B-Chat/lora/eval_on_toxicntest_sft_on_toxicntrain_5epoch/generated_predictions.jsonl"
```bash
python evaluation.py
```

# 3. Filter low confidence predictions and get assistance from 14B model

## Run this .py file. Downloading 14B model may take some time so the whole process may be a bit long. 
```bash
python get_score_final.py
```

