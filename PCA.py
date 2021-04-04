import numpy as np
from sklearn.decomposition import PCA
import cv2

data = []
for i in [str(a + 1).rjust(3, '0') for a in range(15)]:
    for j in [str(b + 1).rjust(2, '0') for b in range(11)]:
        img = cv2.imdecode(np.fromfile('D:/日常工作/2020.作业/2018011205030-向宗义-作业10（ML）/yalefaces/'+i+'/'+j+'.jpg',dtype=np.uint8),-1)
        data.append(img)

n_samples = len(data)
n_features = 147 * 137 * 3
data_2D = np.zeros([n_samples, n_features])

# 将原始数据排列到矩阵内
for i in range(n_samples):
    data_2D[i,:] = data[i].reshape(1,-1)

# 利用PCA训练，保留前20个特征数据
pca = PCA(n_components=20)

# 对data_2D进行降维，得到data1
data1 = pca.fit_transform(data_2D)

# 将降维的数据转化为原始数据
data2 = pca.inverse_transform(data1)

# 将处理后的图片放入文件夹result中
j = 0
for i in [str(a + 1).rjust(3, '0') for a in range(len(data2))]:
    cv2.imwrite("D:/python_data/result/" + i + ".jpg",data2[j].reshape(147, 137, 3))
    j = j + 1