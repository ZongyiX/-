import numpy as np
import random

f_test = open('C:/Users/zongyi xiang/Downloads/dataset/Iris-test.txt')
f_train = open('C:/Users/zongyi xiang/Downloads/dataset/Iris-train.txt')

data_train = list(f_train)
data_test = list(f_test)

random.shuffle(data_train)
random.shuffle(data_test)

# 用于data_train和data_test一样长，所以公用一个长度
length = len(data_train)

# 生成输入到隐层，隐层到输出层的权值矩阵和偏执常数
V = np.random.rand(10, 4)
W = np.random.rand(3, 10)
b_1 = np.random.rand(10, 1)
b_2 = np.random.rand(3, 1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# i 表示第i个数据
def feedforward(data_train, i, V, W, b_1, b_2):
    data_train_ = data_train[i].split()

    # 由输入层到隐层
    x1 = float(data_train_[0])
    x2 = float(data_train_[1])
    x3 = float(data_train_[2])
    x4 = float(data_train_[3])
    t = int(data_train_[4])
    if t == 0:
        T = np.array([[1], [0], [0]])
    elif t == 1:
        T = np.array([[0], [1], [0]])
    elif t == 2:
        T = np.array([[0], [0], [1]])
    # x为输入矩阵
    x = np.array([[x1], [x2], [x3], [x4]])
    A = np.dot(V, x)+b_1
    B = sigmoid(A)

    # 由隐层到输出层
    C = np.dot(W, B) + b_2
    Y = sigmoid(C)
    return Y, T, B, x

def backpropagation(eta, Y, T, V, W, b_1, b_2, B,x):
    # eta 为学习率
    gj = Y*(1-Y)*(T-Y)
    # 从输出层到隐层
    W = W + eta * np.dot(gj, B.transpose())
    b_2 = b_2 - eta*gj

    # 从隐层到输入层
    eh = B*(1-B)*np.dot(W.transpose(), gj)
    V = V + eta*np.dot(eh, x.transpose())
    b_1 = b_1 - eta*eh
    return V, W, b_1, b_2

def train(eta, data_train, length, V, W, b_1, b_2):
    for j in range(1000):
        for i in range(75):
            Y, T, B, x = feedforward(data_train, i, V, W, b_1, b_2)
            V, W, b_1, b_2 = backpropagation(eta, Y, T, V, W, b_1, b_2, B, x)
    return V, W, b_1, b_2

def test(data_test,i, V, W, b_1, b_2):
    data_test_ = data_test[i].split()
    x1 = float(data_test_[0])
    x2 = float(data_test_[1])
    x3 = float(data_test_[2])
    x4 = float(data_test_[3])
    t = int(data_test_[4])
    x = np.array([[x1], [x2], [x3], [x4]])
    A = np.dot(V, x) + b_1
    B = sigmoid(A)
    C = np.dot(W, B) + b_2
    Y = sigmoid(C)
    return Y, t

def evaluate(length, data_test, V, W ,b_1, b_2):
    num_0 = 0
    num_1 = 0
    num_2 = 0
    for i in range(length):
       Y, t = test(data_test, i, V, W, b_1, b_2)
       if t == 0:
           if float(Y[0]) > 0.9 and float(Y[1]) < 0.3 and float(Y[2]) <0.3:
               num_0 = num_0 + 1
       elif t == 1:
           if float(Y[1]) > 0.9 and float(Y[0]) < 0.3 and float(Y[2]) <0.3:
               num_1 = num_1 + 1
       elif t == 2:
           if float(Y[2]) > 0.9 and float(Y[0]) < 0.3 and float(Y[1]) <0.3:
               num_2 = num_2 + 1
       pre_0 = num_0/(25)
       pre_1 = num_1/(25)
       pre_2 = num_2/(25)
    return (pre_0+pre_1+pre_2)/3

# 学习率取0.5
sum = 0
sum_1 = 0
pre = []
for j in range(10):
    V, W, b_1, b_2 = train(0.5, data_train, length, V, W, b_1, b_2)
    pre.append(evaluate(length, data_test, V, W, b_1, b_2))
    print("第%d次的准确率为：%.4f%% \n" % (j+1, pre[j]*100))

for i in range(10):
    sum = pre[i]+sum
ave = sum/10
print("平均准确率为：%.4f%%:" %(ave*100.0))
for j in range(10):
    sum_1 = (pre[j]-ave)**2+sum_1
std = (sum_1/10)**0.5
print('标准差为：%.4f%%' %(std*100))
