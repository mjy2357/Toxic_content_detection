import json

def label_to_numeric(label):
    """将标签转换为数字：无不良内容为0，有不良内容为1"""
    if "不良内容：否" in label or "没有不良内容" in label:
    # if "不良内容：否" in label:
        return 0
    else:
        return 1

def calculate_accuracy(file_path):
    """计算预测的准确率"""
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with open(file_path, 'r', encoding='utf-8') as file:

        for line in file:
            data = json.loads(line)
            label_numeric = label_to_numeric(data['label'])

            predict_numeric = label_to_numeric(data['predict'])

            TP += int((label_numeric == 1) & (predict_numeric == 1))
            FP += int((label_numeric == 0) & (predict_numeric == 1))
            TN += int((label_numeric == 0) & (predict_numeric == 0))
            FN += int((label_numeric == 1) & (predict_numeric == 0))

    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    # print(f'共有{total}条数据，其中预测正确的有{correct_predictions}条。')
    return accuracy, precision, recall

file_path = 'saves/Qwen1.5-0.5B-Chat/lora/eval_on_toxicntest_sft_on_toxicntrain1922reason_7686_8_5epoch/generated_predictions.jsonl'
accuracy, precision, recall = calculate_accuracy(file_path)
print(f'accuracy: {accuracy:.2%}')
print(f'precision: {precision:.2%}')
print(f'recall: {recall:.2%}')