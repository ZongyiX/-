from svmutil import *

# 人工选择部分数据为训练集，部分为测试集
y_train = [1, 1, 1, 1, -1, -1, -1, -1, -1]
x_train = [{1:0.608,2:0.318},{1:0.556,2:0.215},{1:0.403,2:0.237}, {1:0.481,2:0.149},
           {1:0.666,2:0.091},{1:0.243,2:0.267},{1:0.245,2:0.057},{1:0.343,2:0.099}, {1:0.639,2:0.161}]

y_test = [1, 1, 1, 1, -1, -1, -1, -1]
x_test = [{1:0.697,2:0.46},{1:0.437,2:0.211},{1:0.774,2:0.376},{1:0.634,2:0.264},
          {1:0.719,2:0.103},{1:0.593,2:0.042},{1:0.36,2:0.37},{1:0.657,2:0.198}]

# -t 为0时为线性核
model_Linear = svm_train(y_train, x_train, '-t 0')
# -t 为2时为高斯核
model_Gauss = svm_train(y_train, x_train, '-t 2')

print('\n')
p1_label, p1_acc, p1_val = svm_predict(y_test, x_test, model_Linear)
p2_label, p2_acc, p2_val = svm_predict(y_test, x_test, model_Gauss)


