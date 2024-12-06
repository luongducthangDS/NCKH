import numpy as np
import sys
import re
from copy import deepcopy 
from scipy import spatial

data="W2V_150.txt" #file word embedding
dim=150
def load_mappings(path):
    # save word mapping dictionary to pkl file for quickly load for further use
    word_mapping = dict()
    with open(path, 'r',encoding="utf-8") as f:
        # skip first two lines which are vocab size and embedding dimension
        for line in f.readlines():
            if len(line.split())<3:
            	continue
            word, vec = line.split(' ', 1)                        # split word and vector
            vec = np.fromstring(vec, sep=' ')                      # load str to np.array
            word=word.strip()
            word=word.replace(" ","_")
            #if word not in words:
                #words.append(word)
            word_mapping[word] = vec                               # append word:vec to dictionary
    return word_mapping
def getVectors(u1):
    u1 = u1.replace("-","_")
    vs=np.repeat(0.0001,dim)
    mk=1
    if (u1 in model.keys()):
        vs = model[u1]
    else:
        uu=u1.split("_")
        for i in uu:
            if (i in model.keys()):
                if (mk==1):
                    vs=model[i]
                    mk=0  
                else:
                    vs=vs+model[i]
    return  vs
model=load_mappings(data)
#tuong tu giua 2 tu
def sim2word(word1,word2):
    if word1 in model:
        v1 = model[word1]
    else:
        v1= getVectors(word1.strip())
    if word2 in model:
        v2 = model[word2]
    else:
        v2= getVectors(word2.strip())
    return ((2 - spatial.distance.cosine(v1, v2))/2)
#tuong tu giua tu va cau
def simWordSen(word,sen):
    list_sen=sen.strip().split()
    rse=[]
    for sen in list_sen:
        rse.append(sim2word(word,sen))
        print(word,sen,sim2word(word,sen))
    return max(rse)
#tuong tu giua 2 cau
def sim2sen(sen1,sen2):
    sen_item1=sen1.strip().split()
    sen_item2=sen2.strip().split()
    sum1=0
    sum2=0
    for word in sen_item1:
        sum1 = sum1 + simWordSen(word,sen2)
        #print(word,sen2,simWordSen(word,sen2))
    for word in sen_item2:
        sum2 = sum2 + simWordSen(word,sen1)
        #print(word,sen1,simWordSen(word,sen1))
    #print(sum1,sum1/len(sen_item1))
    #print(sum2,sum2/len(sen_item2))
    return (sum1/len(sen_item1)+sum2/len(sen_item2))/2
doc1=sys.argv[1]
doc2=sys.argv[2]
threshold = 0.77
goodpair=dict()
list1=[]
list2=[]

def sim2doc(doc1,doc2):
    f1=open(doc1,"r",encoding="utf-8")
    f2=open(doc2,"r",encoding="utf-8")
    a=""
    b=""
    for line in f1:
        temp=line.rstrip()
        a = a + temp
    for line in f2:
        temp=line.rstrip()
        b = b + temp
    temp1=a.split(".")
    temp2=b.split(".")
    for item in temp1:
        if item != '' and item != '.':
            list1.append(item.strip())
    for item in temp2:
        if item != '' and item != '.':
            list2.append(item.strip())
    
    sum1 = 0
    sum2 = 0
    #print(a.split("."))
    #print(b.split("."))
    for senx in list1:
        rs=[]
        pairs = dict()
        for seny in list2:
            ef = sim2sen(senx,seny)
            rs.append(ef)
            pairs.update({(senx,seny): ef})
        sum1 = sum1 + max(rs)
        if max(rs) >= threshold:
            key_list = list(pairs.keys())
            val_list = list(pairs.values())
            getkey = key_list[val_list.index(max(rs))]
            goodpair.update({getkey : max(rs)})
    for senx in list2:
        rs1=[]
        pairs = dict()
        for seny in list1:
            ef = sim2sen(senx,seny)
            rs1.append(ef)
            pairs.update({(senx,seny): ef})
        sum2 = sum2 + max(rs1)
        if max(rs1) >= threshold:
            key_list = list(pairs.keys())
            val_list = list(pairs.values())
            getkey = key_list[val_list.index(max(rs1))]
            goodpair.update({getkey : max(rs1)})
    return (sum1/len(list1)+sum2/len(list2))/2
def sim2doc2(doc1,doc2):
    print(doc1)
    print(doc2)
    temp1=doc1.split(".")
    temp2=doc2.split(".")
    for item in temp1:
        if item != '':
            list1.append(item.strip())
    for item in temp2:
        if item != '':
            list2.append(item.strip())
    
    sum1 = 0
    sum2 = 0
    for senx in list1:
        rs=[]
        pairs = dict()
        for seny in list2:
            ef = sim2sen(senx,seny)
            rs.append(ef)
            pairs.update({(senx,seny): ef})
        sum1 = sum1 + max(rs)
        if max(rs) >= threshold:
            key_list = list(pairs.keys())
            val_list = list(pairs.values())
            getkey = key_list[val_list.index(max(rs))]
            goodpair.update({getkey : max(rs)})
    for senx in list2:
        rs1=[]
        pairs = dict()
        for seny in list1:
            ef = sim2sen(senx,seny)
            rs1.append(ef)
            pairs.update({(senx,seny): ef})
        sum2 = sum2 + max(rs1)
        if max(rs1) >= threshold:
            key_list = list(pairs.keys())
            val_list = list(pairs.values())
            getkey = key_list[val_list.index(max(rs1))]
            goodpair.update({getkey : max(rs1)})
    return (sum1/len(list1)+sum2/len(list2))/2
    
#ketqua = sim2doc("document1_fix.txt","document2_fix.txt")

ketqua = sim2doc(doc1,doc2)
n = round(ketqua,4)
newketqua = n*100
f3 = open("D:\\Toan\\ketqua.txt","w",encoding="utf-8")
f3.write(str(newketqua)+"%")
f3.close()
f4 = open("D:\\Toan\\phanloai.txt","w",encoding="utf-8")
#for k,v in goodpair:
    #f4.write(str(k)+"||"+str(v)+"||"+str(goodpair[k,v])+"\n")
for i in range(len(list1)):
    for j in range(len(list2)):
        if (list1[i],list2[j]) in goodpair:
            f4.write("A"+str(i)+",B"+str(j)+"|"+list1[i]+"|"+list2[j]+"|"+str(goodpair[list1[i],list2[j]])+"\n")
for i in range(len(list2)):
    for j in range(len(list1)):
        if (list2[i],list1[j]) in goodpair:
            f4.write("B"+str(i)+",A"+str(j)+"|"+list2[i]+"|"+list1[j]+"|"+str(goodpair[list2[i],list1[j]])+"\n")
f4.close()

