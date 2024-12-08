from googletrans import Translator
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Tải tokenizer và mô hình PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")


# Hàm để mã hóa câu
def encode_sentence(sentence, tokenizer, model, max_length=128):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


# Hàm để tách từ và phân nhãn
def tokenize_and_label(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels = ["O"] * len(tokens)  # Gán nhãn "O" (Outside) cho tất cả các token
    return tokens, token_ids, labels


# Danh sách stopword (đơn giản hóa cho ví dụ này)
stopwords = {"và", "trong", "của", "là", "các", "theo"}


# Hàm để loại bỏ stopword
def remove_stopwords(tokens, stopwords):
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


# Đọc dữ liệu từ tệp
with open("dataset.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Dịch câu tiếng Lào sang tiếng Việt và tính toán độ tương đồng
translator = Translator()
for line in lines:
    lao_sentence, viet_sentence, old_similarity_score = line.strip().split("\t")

    # Dịch câu tiếng Lào sang tiếng Việt
    translated_sentence = translator.translate(lao_sentence, src='lo', dest='vi').text

    # Tách từ và phân nhãn
    tokens, token_ids, labels = tokenize_and_label(translated_sentence, tokenizer)
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Labels: {labels}")

    # Loại bỏ stopword
    filtered_tokens = remove_stopwords(tokens, stopwords)
    print(f"Filtered Tokens: {filtered_tokens}")

    # Mã hóa các câu thành các vector
    viet_vector = encode_sentence(viet_sentence, tokenizer, model)
    translated_vector = encode_sentence(translated_sentence, tokenizer, model)

    # Tính độ tương đồng cosine giữa các vector
    new_similarity = cosine_similarity(viet_vector.reshape(1, -1), translated_vector.reshape(1, -1))[0][0]

    # In ra màn hình kết quả
    print(f"{lao_sentence}\t{viet_sentence}\t{old_similarity_score}\t{new_similarity}")
