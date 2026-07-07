import json
from collections import Counter

# 特殊标记
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"  # 添加结束标记

# 文件路径
data_file = "train.txt"  # 数据文件路径
vocab_save_path = "vocab.json"  # 保存词汇表路径
reverse_vocab_save_path = "reverse_vocab.json"  # 保存反向词汇表路径

# 最大长度限制
MAX_SEQ_LENGTH = 40  # 最大句子长度限制
MIN_SEQ_LENGTH = 3  # 最小长度限制（过滤过短句子）
VOCAB_SIZE_LIMIT = 5000

# 数据加载函数
# 加载对话数据，分组，清洗并构建对话对
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        groups = content.split("\n\n")  # 按空行分组

        for group in groups:
            lines = [line.strip() for line in group.split("\n") if line.strip()]
            # 过滤过短、过长句子
            lines = [line for line in lines if MIN_SEQ_LENGTH <= len(line) <= MAX_SEQ_LENGTH]

            # 构建问答对
            for i in range(len(lines) - 1):
                data.append((lines[i], lines[i + 1]))

    print(f"Loaded {len(data)} dialogue samples.")
    return data


# 构建词汇表
def build_vocab(data, vocab_size_limit=VOCAB_SIZE_LIMIT):
    counter = Counter()
    for sample in data:
        for sentence in sample:
            counter.update(sentence)  # 统计字符频率

    # 保留前N个最常见字符，并且添加特殊标记
    most_common = counter.most_common(vocab_size_limit)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2, EOS_TOKEN: 3}  # 特殊标记初始化
    vocab.update({char: idx + 4 for idx, (char, _) in enumerate(most_common)})  # 更新词汇表

    print(f"Vocabulary built with {len(vocab)} tokens.")
    return vocab


# 转换文本数据为ID
def convert_to_ids(data, vocab, max_length=MAX_SEQ_LENGTH):
    def encode(sentence):
        return [vocab.get(char, vocab[UNK_TOKEN]) for char in sentence[:max_length]]

    encoded_data = []
    for question, answer in data:
        question_ids = encode(question) + [vocab[SEP_TOKEN]]
        answer_ids = encode(answer) + [vocab[EOS_TOKEN]]  # 添加结束标记
        encoded_data.append((question_ids, answer_ids))

    print(f"Converted {len(encoded_data)} samples to ID format.")
    return encoded_data

# 保存词汇表和反向词汇表
def save_vocab(vocab, save_path, reverse_save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary saved to {save_path}.")

    # 生成反向词汇表
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    with open(reverse_save_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_vocab, f, ensure_ascii=False, indent=4)
    print(f"Reverse vocabulary saved to {reverse_save_path}.")

# 主函数
def main():
    # 加载数据
    dialogue_data = load_data(data_file)

    # 构建词汇表
    vocab = build_vocab(dialogue_data)

    # 转换数据为ID形式
    # dialogue_ids = convert_to_ids(dialogue_data, vocab)

    # 保存词汇表和反向词汇表
    save_vocab(vocab, vocab_save_path, reverse_vocab_save_path)

    # 输出示例
    print(f"Vocabulary size: {len(vocab)}")
    # print(f"Sample dialogue data (ID): {dialogue_ids[:2]}")

if __name__ == "__main__":
    main()