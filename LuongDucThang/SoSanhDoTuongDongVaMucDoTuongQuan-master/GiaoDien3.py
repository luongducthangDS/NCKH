from tkinter import *
from tkinter import ttk
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

win = Tk()

win.title('Phần Mềm So Sánh Văn Bản')
win.geometry('760x400')
win['bg'] = 'grey'
win.attributes('-topmost', True)

lb1 = Label(win, text="Văn Bản Thứ Nhất", font=('Times New Roman', 12), bg='grey', fg='white')
lb1.place(x=30, y=30)

lb2 = Label(win, text="Văn Bản Thứ Hai", font=('Times New Roman', 12), fg='white')
lb2['bg'] = 'grey'
lb2.place(x=400, y=30)

text1 = Text(win, width=40, height=3, font=('Times New Roman', 12))
text1.place(x=30, y=60)
text1.focus()

text2 = Text(win, width=40, height=3, font=('Times New Roman', 12))
text2.place(x=400, y=60)

model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

listvb1=[]
listvb2=[]
def click():
    vb1 = text1.get("1.0", "end-1c")
    vb2 = text2.get("1.0", "end-1c")
    sentences1 = vb1.split('.')
    sentences2 = vb2.split('.')
    for i in sentences1:
        listvb1.append(i)
    for j in sentences2:
        listvb2.append(j)
    print(listvb1)
    print(listvb2)
    tong=0.0
    sopt1=len(listvb1)
    sopt2=len(listvb2)
    for i in listvb1:
        listmax = []
        vector1 = get_vector(i)
        for j in listvb2:
            vector2 = get_vector(j)
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            listmax.append(similarity)
            maxss = 0
            for kt in listmax:
                if(float(kt)>maxss):
                    maxss=kt
        tong=tong+maxss
        print("max ",maxss)
    for j in listvb2:
        listmax = []
        vector2 = get_vector(j)
        for i in listvb1:
            vector1 = get_vector(i)
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            listmax.append(similarity)
            maxss = 0
            for kt in listmax:
                if(float(kt)>maxss):
                    maxss=kt
        tong=tong+maxss
        print("max ",maxss)
    print("Tong",tong)
    print("So pt1",sopt1)
    print("So pt2",sopt2)
    tong=tong/(sopt1+sopt2)
    print("Do tuong dong",tong)


btntk = Button(win, text="So sánh", font=('Times New Roman', 14), bg='yellow', fg='black', width=10, height=2,
               command=click)
btntk.place(x=330, y=170)

win.mainloop()
