from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import os
from numpy import *
import hog
import roi
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import sys

def test(testpaths):
    trainpaths=['c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\open1\\train','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\open2\\train','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed1\\train','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed2\\train','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed3\\train']
    pathtrainlabels=[0,0,1,1,1]

    # testpaths=['c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\open1\\valid','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\open2\\valid','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed1\\valid','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed2\\valid','c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\closed3\\valid']
    # pathtestlabels=[0,0,1,1,1]

    # 0 for open and 1 for closed
    traindata=[]
    trainlabels=[]
    testdata=[]
    testlabels=[]

    print(" Extracting features from training data ...")
    for i in range(len(trainpaths)):
        outputpath=roi.create_output_folder(trainpaths[i])
        l=os.listdir(outputpath) 
        for file in l: 
            img = cv2.imread(outputpath + '\\' + file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 128))
            fd=hog.extract(img)
            traindata.append(np.array(fd).flatten())
            trainlabels.append(pathtrainlabels[i])

    print(" Extracting features from testing data ...")
    for i in range(len(testpaths)):
        outputpath=roi.create_output_folder(testpaths[i])
        l=os.listdir(outputpath) 
        for file in l: 
            img = cv2.imread(outputpath + '\\' + file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 128))
            fd=hog.extract(img)
            testdata.append(np.array(fd).flatten())
            # testlabels.append(pathtestlabels[i])

    le = LabelEncoder()
    trainlabels = le.fit_transform(trainlabels)

    traindata=np.array(traindata)
    trainlabels = np.array(trainlabels)

    print(" Training Linear SVM classifier...")
    model = LinearSVC()
    model.fit(traindata, trainlabels)

    print(" Evaluating classifier on test data ...")
    predictions = model.predict(np.array(testdata))

    print(predictions)
    # print(classification_report(testlabels, predictions))
    # print(confusion_matrix(testlabels, predictions))

    # fpr, tpr, thresholds = roc_curve(testlabels, predictions)
    # print(fpr, tpr, thresholds)
    # roc_auc = auc(fpr, tpr)
    # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # display.plot()
    # plt.show()

testpath = sys.argv[1]
test([testpath])