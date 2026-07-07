# 启用mask
import torch
import torch.nn as nn

# 超参数配置
MAX_SEQ_LENGTH = 128
PAD_IDX = 0  # 填充符的索引


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

    # 生成 tgt_mask，用于防止 decoder 访问未来的 token
    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))


    # 生成 padding mask，用于屏蔽填充位置（PAD_IDX）
    # 输出形状： (batch_size, seq_len)
    def create_padding_mask(self, seq):
        return (seq == PAD_IDX)  # 直接返回 2D 的布尔张量
    

    # 支持 mask 参数的 forward 函数
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Embedding 和位置编码
        src_emb = self.embedding(src) + self.positional_encoding[:, : src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]

        # 调用 Transformer Decoder
        output = self.decoder(
            tgt=tgt_emb, 
            memory=src_emb, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask, 
            memory_key_padding_mask=src_mask  # 使用修正后的 mask
        )
        logits = self.fc_out(output)
        return logits
