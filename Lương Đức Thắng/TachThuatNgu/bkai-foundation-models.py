from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Mean Pooling - Sử dụng attention mask để tính trung bình đúng
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả các token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def similarity(text1, text2):
    # Tải mô hình từ HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

    # Token hóa các câu
    encoded_input = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')

    # Tính toán các token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        model_output2 = model(**encoded_input2)

    # Thực hiện pooling. Trong trường hợp này, sử dụng mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

    embedding1 = sentence_embeddings[0].unsqueeze(0)
    embedding2 = sentence_embeddings2[0].unsqueeze(0)

    # Tính toán độ tương đồng cosine
    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    return similarity.item()

# Đọc dữ liệu từ file
f = open("F:/Viet_Lao/test/TachThuatNgu/DATA/lao_dich_viet_goc.txt", mode="r", encoding="utf-8")
simchuyengia = []
simtheobert = []
lst1 = []
lst2 = []

# Đọc từng dòng trong file
for s in f:
    s = s.replace('"', '')  # Xóa dấu ngoặc kép nếu có
    ss = s.split('\t')  # Tách các trường dựa trên tab
    if len(ss) != 5:  # Kiểm tra nếu có đủ 5 trường
        print(s)
        continue
    
    temp = int(ss[3])  # Điểm score
    vanban1 = ss[1]  # Văn bản 1
    vanban2 = ss[2]  # Văn bản 2

    lst1.append(vanban1)  # Thêm văn bản 1 vào danh sách
    lst2.append(vanban2)  # Thêm văn bản 2 vào danh sách
    simchuyengia.append(temp)  # Thêm điểm vào danh sách

f.close()

# Kiểm tra nếu có đủ dữ liệu trước khi tính toán tương quan
if len(simtheobert) < 2 or len(simchuyengia) < 2:
    print("Dữ liệu không đủ để tính tương quan!")
else:
    # Tính tương đồng cho từng cặp câu
    for i in range(len(lst1)):
        result = similarity(lst1[i], lst2[i])  # Tính độ tương đồng giữa hai câu
        simtheobert.append(result)  # Thêm kết quả vào danh sách tương đồng

    # In thông tin độ dài của danh sách
    print(f"Length of simtheobert: {len(simtheobert)}")
    print(f"Length of simchuyengia: {len(simchuyengia)}")

    # Tính tương quan giữa phương pháp Bi-Encoder và điểm trong dữ liệu
    pearson = pearsonr(simtheobert, simchuyengia)  # Pearson correlation
    spearman = spearmanr(simtheobert, simchuyengia)  # Spearman correlation

    # In kết quả
    print("Bi-Encoder similarities:", simtheobert)
    print("Expert score:", simchuyengia)
    print("Pearson correlation: ", pearson[0])
    print("Spearman correlation: ", spearman[0])
