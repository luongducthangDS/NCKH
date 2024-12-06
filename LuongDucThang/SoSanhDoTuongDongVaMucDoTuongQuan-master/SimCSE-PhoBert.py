import torch
from scipy.stats._mstats_basic import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize


def similarity(text1, text2):
    # Tokenize và chuyển đổi văn bản thành các vector embedding
    tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

    # Tokenize văn bản
    tokens1 = tokenizer(tokenize(text1), return_tensors="pt")
    tokens2 = tokenizer(tokenize(text2), return_tensors="pt")

    # Tính toán embedding
    with torch.no_grad():
        embeddings1 = model(**tokens1).pooler_output
        embeddings2 = model(**tokens2).pooler_output

    # Tính toán độ tương đồng sử dụng cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return cosine_sim.item()

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
    lst1.append(vanban1)
    lst2.append(vanban2)
    simchuyengia.append(temp)
f.close()
for i in range(len(lst1)):
    result = similarity(lst1[i], lst2[i])
    simtheobert.append(result)
pearson = pearsonr(simtheobert, simchuyengia)
spearman = spearmanr(simtheobert, simchuyengia)
print(simtheobert)
print(simchuyengia)
print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])