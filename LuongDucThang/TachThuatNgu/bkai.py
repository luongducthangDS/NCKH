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
    
    max_length = 512
          
    # Token hóa các câu
    encoded_input = tokenizer(text1, padding=True, truncation=True,max_length=max_length, return_tensors='pt')
    encoded_input2 = tokenizer(text2, padding=True, truncation=True,max_length=max_length, return_tensors='pt')

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






file = "F:/Viet_Lao/dataset final/text/lao_dich_viet_goc.txt"

with open(file, 'r', encoding='utf-8') as datasets:
          lines = datasets.readlines()
                  
data_lao = []

data_viet = []     

data_score = []    

for line in lines:
          columns = line.split('\t')
          
          if len(columns) == 3:
                    lao = columns[0].strip()
                    viet = columns[1].strip()
                    score = columns[2].strip()
                    
                    data_lao.append(lao)
                    data_viet.append(viet)
                    data_score.append(score)
          else:
                    print('dòng tách không đúng cách')
                    
simchuyengia = data_score[1:]
simtheobert = []

for i in range(1,len(data_lao)):
    result = similarity(data_lao[i], data_viet[i])
    simtheobert.append(result)


pearson = pearsonr(simtheobert, simchuyengia)
spearman = spearmanr(simtheobert, simchuyengia)
print(simtheobert)
print(simchuyengia)
print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])