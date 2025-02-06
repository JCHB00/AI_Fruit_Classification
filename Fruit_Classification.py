#Fruit_Classification_ver 1.0
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#1. Split training set

X = [] #Contain the name of picture
Y = [] #定义图像分类类标
Z = [] #Define the pixel of the picture 
Fruit = ['Apple','Banana','Bluebarry','Watermelon']
for i in range(0,4):
    for f in os.listdir(Fruit[i]):
        X.append(Fruit[i] + "//" + str(f))
        Y.append(Fruit[i])

X = np.array(X)
Y = np.array(Y)
#Unkown Step{
#随机率为100% 选取其中的30%作为测试集

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.5, random_state=1)
print(len(X_train), len(X_test), len(y_train), len(y_test))

#}

#2 Read the picture and convert to 像素直方图

#Training Set
XX_train = []
for i in X_train:
    image = cv2.imread(i)
    img = cv2.resize(image, (256,256),interpolation=cv2.INTER_CUBIC)
    #计算图像直方图并存储至X数组
    hist = cv2.calcHist([img],[0,1],None,[256,256],[0,255,0,255])
    XX_train.append(((hist/255).flatten()))

#Testing Set
XX_test = []
for i in X_test:
    #Read Picture    
    image = cv2.imread(i)
    img = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
    hist = cv2.calcHist([img],[0,1],None,[256,256],[0,255,0,255])
    XX_test.append(((hist/255).flatten()))


#3 基于朴素贝叶斯(Naive Bayes)的图像分类处理
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test)
print('预测结果:')
print(predictions_labels)
print('算法评价:')
print(classification_report(y_test, predictions_labels))
#输出前10张图片及预测结果
k = 0
while k<14:
    #读取图像
    print(X_test[k])
    image = cv2.imread(X_test[k])
    print(predictions_labels[k])
    #显示图像
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k = k + 1
#0 is apple
#1 is banana
#2 is bluebarry
#3 is watermelon

"""
预测结果:
[1 2 0 0 3 2 2 2 2 3 2]
算法评价:
              precision    recall  f1-score   support

        0       0.50      1.00      0.67         1
        1       1.00      1.00      1.00         1
        2       1.00      1.00      1.00         6
        3       1.00      0.67      0.80         3

accuracy                            0.91        11
macro avg       0.88      0.92      0.87        11
weighted avg    0.95      0.91      0.92        11
"""

