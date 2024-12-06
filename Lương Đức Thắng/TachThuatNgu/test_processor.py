def load_dictionary(dict_path):
    """
    Load từ điển từ file và trả về một dictionary.
    """
    dic = {}
    with open(dict_path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            dic[line.strip()] = i
    return dic


def replace_with_dict(text, dic, replated):
    """
    Thay thế các cụm từ trong text dựa trên từ điển.
    """
    ws = text.lower().split()
    d = ""
    t = 0
    n = len(ws)

    while t < n:
        k = 1
        tem = ws[t]
        if t < n - 4 and (ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2] + "_" + ws[t + 3] + "_" + ws[t + 4]) in dic:
            k = 5
            tem = ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2] + "_" + ws[t + 3] + "_" + ws[t + 4]
        elif t < n - 3 and (ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2] + "_" + ws[t + 3]) in dic:
            k = 4
            tem = ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2] + "_" + ws[t + 3]
        elif t < n - 2 and (ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2]) in dic:
            k = 3
            tem = ws[t] + "_" + ws[t + 1] + "_" + ws[t + 2]
        elif t < n - 1 and (ws[t] + "_" + ws[t + 1]) in dic:
            k = 2
            tem = ws[t] + "_" + ws[t + 1]

        if tem != ws[t]:
            if tem not in replated:
                replated.append(tem)

        t += k
        d = d + " " + tem if d else tem

    return d.strip()
