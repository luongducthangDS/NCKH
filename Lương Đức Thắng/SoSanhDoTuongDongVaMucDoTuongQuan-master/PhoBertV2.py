from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
from difflib import SequenceMatcher

# Tải pre-trained model và tokenizer
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hàm để chuyển đoạn văn thành vector
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector


f = open("DuLieuChuyenGia.txt", mode="r", encoding="utf-8")
simchuyengia = []
simtheobert = []
lst1 = []
lst2 = []
for s in f:
    s = s.replace('"', '')
    ss = s.split('\t')
    if (len(ss) != 5):
        print(s)
        continue
    temp = int(ss[3]) / 4.0
    vanban1 = ss[1]
    vanban2 = ss[2]
    # with  open("list1.txt", "a", encoding="utf-8") as f:
    #     f.write(vanban1 + "\n")
    # with  open("list2.txt", "a", encoding="utf-8") as f:
    #     f.write(vanban2 + "\n")

    lst1.append(vanban1)
    lst2.append(vanban2)
    simchuyengia.append(temp)
f.close()

# Tạo vector cho mỗi đoạn văn bản
for i in range(len(lst1)):
    vector1 = get_vector(lst1[i])
    vector2 = get_vector(lst2[i])
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    simtheobert.append(similarity)

pearson = pearsonr(simtheobert, simchuyengia)
spearman = spearmanr(simtheobert, simchuyengia)
print(simtheobert)
# print(lst1)
print(simchuyengia)
print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])
