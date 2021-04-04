import numpy as np
import pandas as pd
from math import log2

# 计算熵的函数
def Ent(data):
    good_melon = 0
    bad_melon = 0
    sum_melon = 0
    for item in data:
        if item[0] != '编号':
            if item[9] == '是':
                good_melon += 1
            else:
                bad_melon += 1
            sum_melon += 1
    if(good_melon!=0 and bad_melon!=0):
        p_good = good_melon/sum_melon
        p_bad = bad_melon/sum_melon
        ent = -(p_good) * log2(p_good) - (p_bad) * log2(p_bad)
    else: ent = 0
    return ent

# 计算信息增益的函数
def Gain(indexs, list_data, dict_lists, label_data):
    flag = 0
    for index in indexs:
        if index != 7 and index != 8:
            label_dict = dict_lists[label_data[index]]
            new_list = []
            numb = np.zeros(len(label_dict.keys()))
            for i in range(len(label_dict.keys())):
                klist = []
                new_list.append(klist)
            for item in list_data:
                new_list[int(label_dict[item[index]])].append(item)
                numb[int(label_dict[item[index]])] += 1
            gain = Ent(list_data)
            for i in range(len(label_dict.keys())):
                gain -= (numb[i] / len(list_data)) * Ent(new_list[i])
            if flag == 0:
                max_gain = gain
                idx = index
                m = len(label_dict.keys())
                n = new_list
                flag += 1
            else:
                if gain > max_gain:
                    max_gain = gain
                    idx = index
                    m = len(label_dict.keys())
                    n = new_list
        elif index == 7:
            new_list = []
            for i in range(2):
                klist = []
                new_list.append(klist)
            for item in list_data:
                if float(item[index]) <= 0.381:
                    new_list[0].append(item)
                    numb[0] += 1
                else:
                    new_list[1].append(item)
                    numb[1] += 1
            gain = Ent(list_data)
            for i in range(2):
                gain -= (numb[i] / len(list_data)) * Ent(new_list[i])
            if flag == 0:
                max_gain = gain
                flag += 1
                idx = index
                m = len(label_dict.keys())
                n = new_list
            else:
                if gain > max_gain:
                    max_gain = gain
                    idx = index
                    m = 2
                    n = new_list
        elif index == 8:
            new_list = []
            for i in range(2):
                klist = []
                new_list.append(klist)
            for item in list_data:
                if float(item[index]) <= 0.126:
                    new_list[0].append(item)
                    numb[0] += 1
                else:
                    new_list[1].append(item)
                    numb[1] += 1
            gain = Ent(list_data)
            for i in range(2):
                gain -= (numb[i] / len(list_data)) * Ent(new_list[i])
            if flag == 0:
                max_gain = gain
                flag += 1
                idx = index
                m = len(label_dict.keys())
                n = new_list
            else:
                if gain > max_gain:
                    max_gain = gain
                    idx = index
                    m = 2
                    n = new_list
    return idx, m, n


def Decision_tree(indexs, data_, dict_lists_, root):
    if len(data_) == 0:
        return 0
    left, right = 0, 0
    for iters in data_:
        if iters[9] == '是':
            left += 1
        elif iters[9] == '否':
            right += 1

    if right == 0:
        print(root + '好瓜')
        return 0
    elif left == 0:
        print(root + '坏瓜')
        return 0
    indx, bb, cc = Gain(indexs.keys(), data_, dict_lists_, indexs)
    root += '(' + indexs[indx] + ':'
    del indexs[indx]
    if indx == 8:
        for i in range(bb):
            if float(cc[i][0][indx]) > 0.126:
                Root = root + ' ' + '>=0.126' + ') --> '
                Decision_tree(indexs, cc[i], dict_lists_, Root)
            else:
                Root = root + ' ' + '<=0.126' + ') --> '
                Decision_tree(indexs, cc[i], dict_lists_, Root)
    elif indx == 7:
        for i in range(bb):
            if float(cc[i][0][indx]) > 0.381:
                Root = root + ' ' + '>= 0.381' + ') --> '
                Decision_tree(indexs, cc[i], dict_lists_, Root)
            else:
                Root = root + ' ' + '< 0.381' + ') -- > '
                Decision_tree(indexs, cc[i], dict_lists_, Root)
    elif indx != 7 and indx != 8:
        for i in range(bb):
            Root = root + ' ' + cc[i][0][indx] + ') --> '
            Decision_tree(indexs, cc[i], dict_lists_, Root)

    return 0


dataset=pd.read_csv('C:/Users/zongyi xiang/Downloads/dataset/watermelon3.0.csv')

data = dataset.values.tolist()
label_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
dict_lists = {}
for i in range(len(data[0]) - 1):
    dict_set_list = set()
    flag = 0
    for line in data:
        if flag != 0:
            dict_set_list.add(line[i + 1])
        flag += 1
    key_data = 0
    dict_list = {}
    for set_data in dict_set_list:
        dict_list[set_data] = key_data
        key_data += 1
    dict_lists[data[0][i + 1]] = dict_list
for key in ["密度", "含糖率"]:
    midu = sorted([float(x) for x in list(dict_lists[key].keys())[:]])
    dict_list = {}
    key_data = 0
    for number in midu:
        if number not in dict_list.keys():
            dict_list[str(number)] = key_data
        key_data += 1
    dict_lists[key] = dict_list
root = '●-->'
indexasx = set([1, 2, 3, 4, 5, 6, 7, 8])
indexs = {}
for i in indexasx:
    indexs[i] = label_list[i - 1]
del(data[0])
Decision_tree(indexs, data, dict_lists, root)
