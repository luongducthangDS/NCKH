from gensim.models import Word2Vec
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def word2vec(text1,text2):
    tokens1 = text1.split()
    tokens2 = text2.split()

    # Xây dựng mô hình Word2Vec
    model = Word2Vec([tokens1, tokens2], vector_size=100, window=5, min_count=1, workers=4)

    # Lấy vectơ biểu diễn của câu
    vector1 = np.mean([model.wv[word] for word in tokens1], axis=0)
    vector2 = np.mean([model.wv[word] for word in tokens2], axis=0)

    # Tính mức độ tương đồng sử dụng cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity


f = open("DuLieuChuyenGia.txt", mode="r", encoding="utf-8")
simchuyengia = []
simtheoWord2Vec = []
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
    simWord2Vec= word2vec(lst1[i],lst2[i])
    simtheoWord2Vec.append(simWord2Vec)

pearson = pearsonr(simtheoWord2Vec, simchuyengia)
spearman = spearmanr(simtheoWord2Vec, simchuyengia)

print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])