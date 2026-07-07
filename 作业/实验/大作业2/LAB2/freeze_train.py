# 只训练最后一层+启用mask
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from gpt_optimization import SmallGPT  # 导入c.py中的模型定义
from data_process_b import vocab, X_train, y_train, X_test, y_test  # 从a.py中加载处理后的数据

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
EPOCHS = 20
LEARNING_RATE = 1e-4
PAD_IDX = 0  # 填充符的索引

# 数据加载
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = SmallGPT(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    dropout=DROPOUT,
    pad_idx=PAD_IDX,
)
model.to(device)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻最后一个 Transformer 块和输出层的参数
for param in model.decoder.layers[-1].parameters():
    param.requires_grad = True
for param in model.fc_out.parameters():
    param.requires_grad = True

# 确保只训练解冻的参数
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, inputs)  # 情感分类任务，使用输入作为源和目标
        logits = outputs[:, -1, :]  # 仅获取最后一个时间步的输出
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, inputs)
            logits = outputs[:, -1, :]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 训练和评估
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), "sentiment_model_last_block.pth")
print("Model saved as sentiment_model_last_block.pth")
