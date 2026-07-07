import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler
from data_process_a import load_data, convert_to_ids  # 从 a.py 导入

# 超参数配置
EMBED_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 16  # 每个批次的样本数量
MAX_SEQ_LENGTH = 40
EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_EVERY = 2
MAX_SAMPLES = 300000  # 限制数据集规模
PAD_IDX = 0  # <pad> 的索引


# 数据集类
class DialogueDataset(Dataset):
    def __init__(self, data, max_seq_length=40, pad_idx=0):
        self.data = data
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        question = self.pad_sequence(question)
        answer = self.pad_sequence(answer)
        return torch.tensor(question), torch.tensor(answer)

    def pad_sequence(self, sequence):
        if len(sequence) > self.max_seq_length:
            return sequence[: self.max_seq_length]
        else:
            return sequence + [self.pad_idx] * (self.max_seq_length - len(sequence))


# 小型 GPT 模型
class SmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ffn_dim, dropout, pad_idx):
        super(SmallGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, MAX_SEQ_LENGTH, embed_dim)
        )  # 可训练的位置编码
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        # src 是输入序列，tgt 是目标序列
        src_emb = self.embedding(src) + self.positional_encoding[:, : src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]
        output = self.decoder(tgt=tgt_emb, memory=src_emb)  # Decoder-only
        logits = self.fc_out(output)
        return logits


# 减少数据集规模
def reduce_dataset_size(data, max_samples=300000):
    if len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]
    print(f"Reduced dataset size: {len(data)} samples.")
    return data


# 训练函数
def train_model(model, data_loader, optimizer, scheduler, criterion, epochs, device):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for question, answer in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}"):
            question, answer = question.to(device), answer.to(device)

            # tgt_input 和 tgt_target 是 shifted
            tgt_input = answer[:, :-1]
            tgt_target = answer[:, 1:]

            optimizer.zero_grad()
            output = model(question, tgt_input)
            loss = criterion(output.view(-1, model.fc_out.out_features), tgt_target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} Loss: {epoch_loss / len(data_loader):.4f}")

        # 保存模型
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
            print(f"Model checkpoint saved at epoch {epoch}.")

# 主函数
def main():
    # 加载对话数据
    print("Loading dialogue data...")
    dialogue_data = load_data("train.txt")
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    dialogue_data = convert_to_ids(dialogue_data, vocab, max_length=MAX_SEQ_LENGTH)

    # 减少数据集规模
    dialogue_data = reduce_dataset_size(dialogue_data, max_samples=MAX_SAMPLES)

    # 创建数据集和数据加载器
    dataset = DialogueDataset(dialogue_data, max_seq_length=MAX_SEQ_LENGTH, pad_idx=PAD_IDX)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    print("Initializing model...")
    model = SmallGPT(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(data_loader))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 开始训练
    print("Starting training...")
    train_model(model, data_loader, optimizer, scheduler, criterion, EPOCHS, device)

if __name__ == "__main__":
    main()
