import torch
import torch.nn.functional as F
import json
from gpt import SmallGPT  # 从 b.py 导入训练中定义的 GPT 模型结构

# 特殊标记
PAD_IDX = 0
UNK_IDX = 1
SEP_IDX = 2
EOS_IDX = 3

# 模型参数
EMBED_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_DIM = 256
DROPOUT = 0.1
MAX_SEQ_LENGTH = 40

# Top-k 和 Top-p 策略
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove[..., 0].item():
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def generate_response(
    model,
    input_text,
    token_to_idx,
    idx_to_token,
    max_length=30,
    top_k=100,
    temperature=1.2,
    repetition_penalty=1.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 将输入文本转换为 ID 序列
    input_ids = [token_to_idx.get(char, token_to_idx.get("<unk>")) for char in input_text[:MAX_SEQ_LENGTH]]
    input_tensor = torch.tensor([input_ids], device=device)

    # 初始化生成序列
    generated_ids = [SEP_IDX]  # 用分隔符标记作为起始标记
    generated_tensor = torch.tensor([generated_ids], device=device)

    for _ in range(max_length):
        # 前向传播：生成下一个 token 的 logits
        logits = model(input_tensor, generated_tensor)
        next_token_logits = logits[0, -1, :]

        # 应用温度缩放
        next_token_logits = next_token_logits / temperature

        # 使用 Top-k 策略过滤低概率
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k)

        # 引入重复惩罚
        for token_id in set(generated_ids):
            filtered_logits[token_id] /= repetition_penalty  # 对已生成 token 进行惩罚

        # 根据概率采样下一个 token
        probs = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # 如果生成结束标记，则停止生成
        if next_token_id == EOS_IDX:
            break

        # 避免连续生成完全无意义的字符
        if len(generated_ids) > 1 and next_token_id == generated_ids[-1]:
            continue  # 跳过连续重复的 token

        # 将生成的 token 添加到序列中
        generated_ids.append(next_token_id)
        generated_tensor = torch.tensor([generated_ids], device=device)

    # 将生成的 ID 序列转换为文本
    generated_text = "".join([idx_to_token[idx] for idx in generated_ids if idx not in [PAD_IDX, UNK_IDX, SEP_IDX, EOS_IDX]])
    return generated_text.strip()  # 去掉多余空格

def main():
    # 加载词汇表
    print("Loading vocabulary...")
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_idx = vocab
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # 检查词汇表是否有所有必要的特殊 token
    required_tokens = [PAD_IDX, UNK_IDX, SEP_IDX, EOS_IDX]
    missing_tokens = [token for token in required_tokens if token not in token_to_idx.values()]
    if missing_tokens:
        print(f"Warning: Missing tokens in vocabulary: {missing_tokens}")

    # 加载训练好的模型
    print("Loading model...")
    model = SmallGPT(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    )
    model.load_state_dict(torch.load("model_epoch_10.pt"))  # 加载训练完成的模型权重
    print("Model loaded successfully.")

    # 对话生成测试
    print("\n--- Chat Interface ---")
    print("Type 'exit' to quit.")
    while True:
        input_text = input("You: ").strip()
        if not input_text:
            print("Error: Input cannot be empty.")
            continue
        if input_text.lower() == "exit":
            break
        response = generate_response(model, input_text, token_to_idx, idx_to_token)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
