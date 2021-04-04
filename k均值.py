import matplotlib.pyplot as plt
import random

dataset = [[0.697, 0.460],
           [0.774, 0.376],
           [0.634, 0.264],
           [0.608, 0.318],
           [0.556, 0.215],
           [0.403, 0.237],
           [0.481, 0.149],
           [0.437, 0.211],
           [0.666, 0.091],
           [0.243, 0.267],
           [0.245, 0.057],
           [0.343, 0.099],
           [0.639, 0.161],
           [0.657, 0.198],
           [0.360, 0.370],
           [0.593, 0.042],
           [0.719, 0.103],
           [0.359, 0.188],
           [0.339, 0.241],
           [0.282, 0.257],
           [0.748, 0.232],
           [0.714, 0.346],
           [0.483, 0.312],
           [0.478, 0.437],
           [0.525, 0.369],
           [0.751, 0.489],
           [0.532, 0.472],
           [0.473, 0.376],
           [0.725, 0.445],
           [0.446, 0.459]]
k = eval(input("输入K值："))
m = len(dataset)
num = []
for i in range(k):
    p = eval(input("输入选择第i个数据为初始均值点(i=1,2..,30):"))
    num.append(p)

def init_means(k,m,num):
    # 初始选择的均值点的集合
    U = []
    for i in range(k):
        u = dataset[num[i]-1]
        U.append(u)
    return U

def kmeans(k, m, U):
    # 将样本划入簇
    C = {}
    lamda = [] # 簇标记的集合
    for i in range(k):
        lamda.append(i)
        C.setdefault(i,[]).append(U[i])
    for j in range(m):
        x = dataset[j]
        dist = {}
        for i in range(k):
            u = dataset[num[i]-1]
            dist[lamda[i]] = ((x[0]-u[0])**2+(x[1]-u[1])**2)**1/2
            for key,value in dist.items():
                if value == min(dist.values()):
                    l = key # 返回相应的簇标记
        C.setdefault(l,[]).append(x)

    item = 0 #每轮更新均值向量时，若为更新，则item++，若item=k，结束
    # 更新均值向量
    for i in range(k):
        length_C = len(C[i])
        sum_x1 = 0
        sum_x2 = 0
        for j in range(length_C):
            sum_x1 = sum_x1 + C[i][j][0]
            sum_x2 = sum_x2 + C[i][j][1]
        u1 = sum_x1/length_C
        u2 = sum_x2/length_C
        u_ = [u1,u2]
        if u_ != U[i]:
            U[i] = u_
        elif u_ == U[i]:
            item = item + 1
    return C, U, item

n = 100
U_init = init_means(k, m, num)
X_init = []
Y_init = []
# 画起始均值点
for i in range(k):
    X_init.append(U_init[i][0])
    Y_init.append(U_init[i][1])
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel('密度')
plt.ylabel('含糖率')
plt.scatter(X_init, Y_init, marker='x',color='black',s=100)

U0 = U_init
item1 = 0
for i in range(n):
    C, U, item = kmeans(k, m, U0)
    U0 = U
    item1 = item1+1
    if item == k:
        print("提前跳出循环,循环了%d次" %item1)
        break

# 画分类的点
for i in range(k):
    X = []
    Y = []
    for j in range(len(C[i])):
        X.append(C[i][j][0])
        Y.append(C[i][j][1])
    color = input("输入所需颜色:")
    plt.scatter(X,Y,color=color)

# 画最终初始点
X_end = []
Y_end = []
for i in range(k):
    X_end.append(U[i][0])
    Y_end.append(U[i][1])
plt.scatter(X_end, Y_end, marker='+',c='black',s=100)
plt.show()