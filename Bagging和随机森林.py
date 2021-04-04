import numpy as np
from math import log2
import random
from sklearn.ensemble import RandomForestClassifier

# 西瓜数据集3.0a
dataset = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]]
data = np.array(dataset)


# 决策树作为基分类器
class DecisionTree(object):
        def split(self,data):
            # 密度的划分
            D_1 = data[:, 0]
            D_1_sort = D_1.argsort()
            T_1 = []
            for i in range(16):
                    j = D_1_sort[i]
                    k = D_1_sort[i + 1]
                    a = (D_1[j] + D_1[k]) / 2
                    T_1.append(a)
            # 含糖率的划分
            D_2 = data[:, 1]
            D_2_sort = D_2.argsort()
            T_2 = []
            for i in range(16):
                    j = D_2_sort[i]
                    k = D_2_sort[i + 1]
                    a = (D_2[j] + D_2[k]) / 2
                    T_2.append(a)
            return T_1, T_2
        def shannonent(self,data):
                len0 = 17
                good_melon = 0
                bad_melon = 0
                for i in range(len0):
                        if data[i,2] == 1:
                                good_melon=good_melon+1
                        if data[i,2] == 0:
                                bad_melon=bad_melon+1
                p_good = good_melon/len0
                p_bad = bad_melon/len0
                D_ent = -p_good*log2(p_good)-p_bad*log2(p_bad)
                # 计算两个属性的熵
                T_1,T_2 = DecisionTree().split(data)
                max1_Gain = 0
                max2_Gain = 0
                div_point1 = 0
                div_point2 = 0
                for i in range(16):
                        # 密度熵计算
                        z1 = []
                        div1 = T_1[i]  # 划分点
                        for j in range(len0):
                                if data[j,0] < div1:
                                        z1.append(data[j,:])
                                good1_melon = 0
                                bad1_melon = 0
                                len1 = len(z1)
                        for a in z1:

                                if a[2] == 0:
                                        bad1_melon = bad1_melon+1
                                if a[2] == 1:
                                        good1_melon = good1_melon+1

                        if len1 ==0 :
                                p1_good_N = 100000
                                p1_bad_N = 100000
                        elif len1!= 0:
                                p1_good_N = good1_melon / len1
                                p1_bad_N = bad1_melon / len1

                        p1_good_P = (good_melon-good1_melon)/(len0-len1)
                        p1_bad_P = (bad_melon-bad1_melon)/(len0-len1)
                        if p1_good_N != 0:
                                fN_good = log2(p1_good_N)
                        else:fN_good = -10000 #用极小的数表示负无穷
                        if p1_bad_N != 0:
                                fN_bad = log2(p1_bad_N)
                        else:fN_bad =-10000
                        if p1_good_P != 0:
                                fP_good = log2(p1_good_P)
                        else:fP_good = -10000
                        if p1_bad_P != 0:
                                fP_bad = log2(p1_bad_P)
                        else:fP_bad = -10000

                        D1_ent_N = -p1_good_N * fN_good - p1_bad_N * fN_bad
                        D1_ent_P = -p1_good_P * fP_good - p1_bad_P * fP_bad
                        Gain1 = D_ent - (len1/len0)*D1_ent_N-((len0-len1)/len0)*D1_ent_P
                        if max1_Gain < Gain1:
                                max1_Gain = Gain1
                                div_point1 = div1

                        # 含糖率熵计算
                        z2 = []
                        div2 = T_2[i]  # 划分点
                        for j in range(len0):
                                if data[j, 1] < div2:
                                        z2.append(data[j, :])
                                good2_melon = 0
                                bad2_melon = 0
                                len2 = len(z2)
                        for a in z2:

                                if a[2] == 0:
                                        bad2_melon = bad2_melon + 1
                                if a[2] == 1:
                                        good2_melon = good2_melon + 1

                        if len2!= 0:
                                p2_good_N = good2_melon / len2
                                p2_bad_N = bad2_melon / len2
                        elif len2 == 0:
                                p2_good_N = 100000
                                p2_bad_N = 100000

                        p2_good_P = (good_melon - good2_melon) / (len0 - len2)
                        p2_bad_P = (bad_melon - bad2_melon) / (len0 - len2)
                        if p2_good_N != 0:
                                fN_good = log2(p2_good_N)
                        else:fN_good = -10000 #用极小的数表示负无穷
                        if p2_bad_N != 0:
                                fN_bad = log2(p2_bad_N)
                        else:fN_bad =-10000
                        if p2_good_P != 0:
                                fP_good = log2(p2_good_P)
                        else:fP_good = -10000
                        if p2_bad_P != 0:
                                fP_bad = log2(p2_bad_P)
                        else:fP_bad = -10000
                        D2_ent_N = -p2_good_N * fN_good - p2_bad_N * fN_bad
                        D2_ent_P = -p2_good_P * fP_good - p2_bad_P * fP_bad
                        Gain2 = D_ent - (len2 / len0) * D2_ent_N - ((len0 - len2) / len0) * D2_ent_P
                        if max2_Gain < Gain2:
                                max2_Gain = Gain2
                                div_point2 = div2
                #print(div_point1,div_point2)
                #print(max1_Gain, max2_Gain)
                # 返回划分点
                return div_point1,div_point2,max1_Gain,max2_Gain

class bagging(object):
        # 选取T个随机样本进行训练
        def train(self,data,T):
                data1 = {}
                for t in range(T):
                        data2 = []
                        for i in range(17):
                                j = random.randint(0, 16)
                                data2.append(data[j,:])
                        data3 = np.array(data2)
                        div_point1, div_point2,max1_Gain,max2_Gain = DecisionTree().shannonent(data3)
                        data1[t] = div_point1, div_point2,max1_Gain,max2_Gain
                return data1
        def test(self,data1,T,data_test):
                vote = 0
                for t in range(T):
                        if data1[t][2] > data1[t][3]:
                                if data_test[0] >= data1[t][0]:
                                        if data_test[1]>=data1[t][1]:
                                            vote = vote + 1
                        elif data1[t][2]<=data1[t][3]:
                                if data_test[1]>=data1[t][1]:
                                        if data_test[0]>=data1[t][0]:
                                                vote = vote + 1
                label = -1
                if vote > T/2:
                        label = 1
                elif vote <=T/2:
                        label = 0

                return label

class RF(object):
        def randomforest(self,dataset):
                random.shuffle(dataset)
                data = np.array(dataset)
                X = data[:,0:2]
                y = data[:,2]
                forest = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
                forest.fit(X,y)
                # 选取5个数据进行测试
                data_test = data[0:5,:]
                X_test = data_test[:,0:2]
                pre = forest.predict(X_test)
                y_test = data_test[:,2]
                acc_num = 0
                for i in range(5):
                        if pre[i] == y_test[i]:
                                acc_num = acc_num + 1
                acc = acc_num/5
                return acc

T = eval(input("输入集成器个数T："))
num_acc = 1
num = 0
for i in range(17):
        data_test = data[i]
        data1 = bagging().train(data,T)
        label_pre = bagging().test(data1,T,data_test)
        if data_test[2] == label_pre:
            num = num+num_acc
acc = (num/17)*100
acc_RF = RF().randomforest(dataset)*100

print('Bagging的精确度为：%.2f%%' %acc)
print('随机森林的精确度为：%.2f%%' %acc_RF)













