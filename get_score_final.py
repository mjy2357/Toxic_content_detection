from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import GenerationConfig
import pandas as pd

device = "cuda"  # the device to load the model onto

# 加载较小的模型，这里以微调后的Qwen1.5-0.5B-Chat为例
model = AutoModelForCausalLM.from_pretrained(
    "dango2357/Qwen1.5-0.5B-chat_sft_on_Toxicn",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("dango2357/Qwen1.5-0.5B-chat_sft_on_Toxicn")

# 读取测试集
with open('data/COLD_test_alpaca_fomat.json', 'r') as file:
    test_set = json.load(file)

# 读取测试集的label
label = [sample['output'] for sample in test_set]

# 设定batch size
batch_size = 8

# 初始化一个空列表，用于存储tokenized的样本
tokenized_samples = []

# 遍历测试集，转换其格式
for i, sample in enumerate(test_set):
    # if i == 16:
    #     break
    prompt = sample['instruction'] + '\n' + sample['input']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # messages = [
    #     {"role": "system", "content": sample['instruction']},
    #     {"role": "user", "content": sample['input']}
    # ]
    current_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(current_text)
    # 将tokenized的文本添加到tokenized_samples列表中
    tokenized_samples.append(current_text)

generation_config = GenerationConfig(
    max_new_tokens=256,  # 设置最大生成长度
    do_sample=True,
    top_p=0.7,
    # top_k=50,
    output_scores=True,  # 输出logits以便计算置信度
    return_dict_in_generate=True  # 返回生成的字典，包括logits和生成的token
)

# 定义根据logits计算置信度的函数
def calculate_confidence(logits, token_ids):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    # 由于微调后的模型会以“不良内容：是”的格式输出，因此这里直接读取倒数第2个位置的confidence（倒数第1个位置是特殊字符）
    token_probabilities = probabilities[-2, token_ids[-2]]
    # print(token_ids[-2])
    confidence = token_probabilities.item()
    return confidence

# 初始化一个空列表，用于存储结果和置信度
results = []

# 将样本分成batch进行处理
for i in range(0, len(tokenized_samples), batch_size):
    # if i == 8:
    #     break
    batch = tokenized_samples[i:i + batch_size]

    # 将样本tokenize并pad
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # 解码生成的输出
    decoded_outputs = tokenizer.batch_decode(outputs.sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    for j, output in enumerate(decoded_outputs):
        # print(f"Sample {i + j + 1}: {output}")

        # 获取生成的token id和logits
        token_ids = outputs.sequences[j, inputs['input_ids'].shape[1]:]
        logits = torch.stack(outputs.scores, dim=1)

        # 计算置信度
        # print(logits[j])
        # print(token_ids)
        confidence = calculate_confidence(logits[j], token_ids)
        # print(f"Confidence: {confidence:.8f}")

        # 将结果和置信度添加到结果列表中
        results.append([output, confidence])

# 将结果保存到CSV文件，方便做可视化
df = pd.DataFrame(results, columns=['Output', 'Confidence'])
df['label'] = label
df.to_csv('results_from_0.5B_cold.csv', index=False)
print("Results and confidence scores saved to results_from_0.5B_cold.csv")

# 计算较小的模型的准确率，同时得到置信度较低样本的序号的列表
correct = 0
idx_with_low_confidence = []
for i, result in enumerate(results):
    if label[i] in result[0]:
        correct += 1
    if result[1] <= generation_config.top_p:
        idx_with_low_confidence.append(i)
Accuracy = correct / len(results) if len(results) != 0 else 0
print(f'Accuracy: {Accuracy*100:.2f}%')

# 加载较大的模型，这里以未经过微调的Qwen1.5-14B-Chat为例
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")
batch_size = 2
samples_with_low_confidence = [tokenized_samples[i] for i in idx_with_low_confidence]
print(idx_with_low_confidence)
print(len(samples_with_low_confidence))

for i in range(0, len(samples_with_low_confidence), batch_size):
    if i % 100 == 0:
        print(f'处理到第{i}个')
    batch = samples_with_low_confidence[i:i + batch_size]

    # 将样本tokenize并pad
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # 解码生成的输出
    decoded_outputs = tokenizer.batch_decode(outputs.sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    for j, output in enumerate(decoded_outputs):
        # 将较大模型的结果添加到results的对应位置，覆盖掉较小模型的预测结果
        idx_in_results = idx_with_low_confidence[i + j]
        results[idx_in_results][0] = output

# 将结果保存到CSV文件
df = pd.DataFrame(results, columns=['Output', 'Confidence'])
df['label'] = label
df.to_csv('results_with_assistance_cold.csv', index=False)
print("Results and confidence scores saved to results_with_assistance_cold.csv")

# 计算在较大的模型的协助下的准确率
correct = 0
for i, result in enumerate(results):
    if label[i] in result[0]:
        correct += 1
Accuracy_with_assistance = correct / len(results) if len(results) != 0 else 0
print(f'Accuracy_with_assistance: {Accuracy_with_assistance*100:.2f}%')