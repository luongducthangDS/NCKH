from tkinter import *
from tkinter import ttk
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

win = Tk()
s = ttk.Style()
s.theme_use('clam')
win.title('Phần Mềm So Sánh Mức độ tương đồng giữa hai văn bản')
win.geometry('950x470')
win['bg'] = 'grey'
win.attributes('-topmost', True)

lb1 = Label(win, text="Văn Bản Thứ Nhất", font=('Times New Roman', 12), bg='grey', fg='white')
lb1.place(x=30, y=30)

text1 = Text(win, width=80, height=3, font=('Times New Roman', 12))
text1.place(x=30, y=60)
text1.focus()

btnopen1 = ttk.Button(win, text="Mở file văn bản thứ nhất")
btnopen1.place(x=700, y=70)

lb2 = Label(win, text="Văn Bản Thứ Hai", font=('Times New Roman', 12), fg='white')
lb2['bg'] = 'grey'
lb2.place(x=30, y=150)

text2 = Text(win, width=80, height=3, font=('Times New Roman', 12))
text2.place(x=30, y=180)

btnopen2 = ttk.Button(win, text="Mở file văn bản thứ hai")
btnopen2.place(x=700, y=190)

def phobertV2(text1,text2):
    # phobert-V2
    model_name = "vinai/phobert-base-v2"
    phobert = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids1 = tokenizer.encode(text1, return_tensors="pt")
    input_ids2 = tokenizer.encode(text2, return_tensors="pt")
    with torch.no_grad():
        outputs1 = phobert(input_ids1)
        outputs2 = phobert(input_ids2)
    vector1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
    vector2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()
    return cosine_similarity([vector1], [vector2])[0][0]

listvb1=[]
listvb2=[]
def click():
    vb1 = text1.get("1.0", "end-1c")
    vb2 = text2.get("1.0", "end-1c")
    sentences1 = vb1.split('.')
    sentences2 = vb2.split('.')
    listvb1.clear()  # Xóa danh sách cũ
    listvb2.clear()  # Xóa danh sách cũ
    for i in sentences1:
        listvb1.append(i.strip())  # Thêm vào danh sách mới
    for j in sentences2:
        listvb2.append(j.strip())  # Thêm vào danh sách mới

    sopt1 = len(listvb1)
    sopt2 = len(listvb2)

    def update_progress(value):
        tong = 0.0
        bar['value'] = value
        for i in listvb1:
            listmax = []
            for j in listvb2:
                similarity = phobertV2(i, j)
                listmax.append(similarity)
                maxss = max(listmax)
                if maxss < 0.8:
                    maxss = 0
            tong += maxss
        for j in listvb2:
            listmax = []
            for i in listvb1:
                similarity = phobertV2(i, j)
                listmax.append(similarity)
                maxss = max(listmax)
                if maxss < 0.8:
                    maxss = 0
            tong += maxss
        tong /= (sopt1 + sopt2)
        if value < tong * 100:
            win.after(20, update_progress, value + 1)
            if tong < 0.6:
                s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='red')
            elif 0.6 <= tong < 0.8:
                s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='yellow')
            else:
                s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='green')

        # Hiển thị kết quả
        lb5 = Label(win, text="PhobertV2: ", font=('Times New Roman', 14), bg="grey", fg="white")
        lb5.place(x=30, y=298)
        lb6 = Label(win, text=" " + str(tong), font=('Times New Roman', 14), bg="grey", fg="white")
        lb6.place(x=490, y=299)

    bar = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                          style='phobert.Horizontal.TProgressbar')
    bar.place(x=180, y=300, height="25")
    update_progress(0)


btnss = Button(win, text="So sánh", font=('Times New Roman', 14), bg='yellow', fg='black', width=10, height=2,
               command=click)
btnss.place(x=700, y=260)

win.mainloop()