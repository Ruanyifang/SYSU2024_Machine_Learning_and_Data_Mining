import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import gpt_no_mask
import gpt_optimization
from data_process_b import vocab, X_test, y_test  # 从a.py导入测试数据

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
EMBED_DIM = 128
NUM_LAYERS = 3
NUM_HEADS = 4
FFN_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 16
PAD_IDX = 0  # 填充符的索引

# 加载测试数据
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型加载函数
def load_model_all(pth_file):
    model = gpt_optimization.SmallGPT(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    )
    model.load_state_dict(torch.load(pth_file, map_location=device))  # 加载模型权重
    model.to(device)
    model.eval()  # 设置模型为评估模式
    return model

# 模型加载函数
def load_model_no_mask(pth_file):
    model = gpt_no_mask.SmallGPT(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    )
    model.load_state_dict(torch.load(pth_file, map_location=device))  # 加载模型权重
    model.to(device)
    model.eval()  # 设置模型为评估模式
    return model

# 评估函数
def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, inputs)  # 情感分类任务，使用输入作为源和目标
            logits = outputs[:, -1, :]  # 获取最后一个时间步的输出
            preds = torch.argmax(logits, dim=-1)  # 获取预测类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率和分类报告
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
    return accuracy, report

# 加载两个模型权重
model_1 = load_model_all("sentiment_model.pth")
model_2 = load_model_all("sentiment_model_last_block.pth")
model_3 = load_model_no_mask("sentiment_model_with_no_mask.pth")

# 对第一个模型进行评估
print("\nEvaluating Model 1...")
accuracy_1, report_1 = evaluate_model(model_1, test_loader, device)
print(f"Model 1 Test Accuracy: {accuracy_1:.4f}")
print("Model 1 Classification Report:")
print(report_1)

# 对第二个模型进行评估
print("\nEvaluating Model 2...")
accuracy_2, report_2 = evaluate_model(model_2, test_loader, device)
print(f"Model 2 Test Accuracy: {accuracy_2:.4f}")
print("Model 2 Classification Report:")
print(report_2)

# 对比评估结果
print("\nComparison of the Two Models:")
print(f"Model 1 Accuracy: {accuracy_1:.4f}")
print(f"Model 2 Accuracy: {accuracy_2:.4f}")
if accuracy_1 > accuracy_2:
    print("Model 1 performs better on the test set.")
elif accuracy_1 < accuracy_2:
    print("Model 2 performs better on the test set.")
else:
    print("Both models perform equally well on the test set.")

# 对第三个模型进行评估
print("\nEvaluating Model 3...")
accuracy_3, report_3 = evaluate_model(model_3, test_loader, device)
print(f"Model 3 Test Accuracy: {accuracy_3:.4f}")
print("Model 3 Classification Report:")
print(report_3)