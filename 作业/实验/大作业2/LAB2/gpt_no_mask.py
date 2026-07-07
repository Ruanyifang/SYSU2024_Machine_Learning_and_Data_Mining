# 本身没有使用mask
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 超参数配置
MAX_SEQ_LENGTH = 128


# 数据集类
class DialogueDataset(Dataset):
    def __init__(self, data, max_seq_length=128, pad_idx=0):
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
        # 动态调整位置编码大小
        src_emb = self.embedding(src) + self.positional_encoding[:, : src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]
        output = self.decoder(tgt=tgt_emb, memory=src_emb)
        logits = self.fc_out(output)
        return logits
