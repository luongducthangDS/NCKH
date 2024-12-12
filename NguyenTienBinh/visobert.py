from transformers import AutoModel, AutoTokenizer
import torch
import re
from pyvi import ViTokenizer, ViPosTagger
from scipy.spatial.distance import cosine
import numpy as np
from scipy import stats

model = AutoModel.from_pretrained('uitnlp/visobert')
tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

def encode(text):
    encoding = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1.numpy(), vec2.numpy())

current_line_index1 = 0
text1=""
def read_file_by_sections1(file_path1):
    global current_line_index1, text1
    try:
        with open(file_path1, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if current_line_index1 < len(lines):
                text1 = lines[current_line_index1].strip()
                current_line_index1 += 1
            else:
                text1 = "Đã đọc hết file."
    except FileNotFoundError:
        text1 = f"File not found: {file_path1}"
    except Exception as e:
        text1 = f"An error occurred: {e}"

text2 =""
lao =""
sim =""
vietlao = ""
current_line_index2 = 0
def read_file_by_sections2(file_path2):
    global current_line_index2, vietlao, text2, lao, sim
    try:
        with open(file_path2, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if current_line_index2 < len(lines):
                vietlao = lines[current_line_index2].strip()
                lao, viet, sim = vietlao.split('\t')
                viet = ViTokenizer.tokenize(viet)
                lao = ViTokenizer.tokenize(lao)
                sim = ViTokenizer.tokenize(sim)
                text2 = viet
                current_line_index2 += 1
            else:
                vietlao = "Đã đọc hết file."
    except FileNotFoundError:
        vietlao = f"File not found: {file_path2}"
    except Exception as e:
        vietlao = f"An error occurred: {e}"

# Mã hóa văn bản
def encode(text):
    encoding = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1.numpy(), vec2.numpy())

i = 1
list_viso =[]
list_Sim =[]
for i  in range(2000):
    file_path1 = r'D:\DATA\bandichtienglao.txt'
    read_file_by_sections1(file_path1)
    file_path2 = r'D:\DATA\dataset.txt'
    read_file_by_sections2(file_path2)
    print(text1)
    file_path3 = 'outputvietlao.txt'
    with open(file_path3, 'a', encoding='utf-8') as file:
        file.write(f'{i+1}, ' + lao + '\n' + '=> '+ text1)
    print(text2)
    with open(file_path3, 'a', encoding='utf-8') as file:
        file.write('- ' + text2 + '\n')
    vector1 = encode(text1)
    vector2 = encode(text2)
    similarity = cosine_similarity(vector1, vector2)
    print(f"Similarity Visobert: {similarity:.4f}")
    with open(file_path3, 'a', encoding='utf-8') as file:
        file.write(f"- Similarity Visobert: {similarity:.4f}" + f"\tTự đánh giá: {sim}" + '\n' )
    list_viso.append(float(f'{similarity:.4f}'))
    list_Sim.append(float(sim.strip()))


# print(list_viso)
# print(list_Sim)
pearson_correlation, p_value = stats.pearsonr(list_viso, list_Sim)
print(f"Pearson Correlation: {pearson_correlation}")
print(f"P-value: {p_value}")