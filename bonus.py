from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import numpy as np
import cv2
import os
from numpy import *
from hog import extract
import time
import pygame
import roi
pygame.mixer.init()

# Function to play a sound
def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

# Function to pause the currently playing sound
def pause_sound():
    pygame.mixer.music.pause()


# Function to unpause the currently playing sound
def unpause_sound():
    pygame.mixer.music.unpause()

unpause_path = 'c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\unpause'
pause_path= 'c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\pause'
next_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\next'
one_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\one'
two_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\two'
three_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\three'
four_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\four'
five_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\five'

p1 = roi.create_output_folder(unpause_path)
p2 = roi.create_output_folder(pause_path)
p5 = roi.create_output_folder(next_path)
p6 = roi.create_output_folder(one_path)
p7 = roi.create_output_folder(two_path)
p8 = roi.create_output_folder(three_path)
p9 = roi.create_output_folder(four_path)
p10 = roi.create_output_folder(five_path)

data= []
labels = []
testdata=[]

for file in os.listdir(p6): #one
    img = cv2.imread(p6 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(1)

for file in os.listdir(p7): #two
    img = cv2.imread(p7 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(2)

for file in os.listdir(p8): #three
    img = cv2.imread(p8 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)    
    data.append(np.array(fd).flatten())
    labels.append(3)

for file in os.listdir(p9): #four
    img = cv2.imread(p9 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(4)

for file in os.listdir(p10): #five
    img = cv2.imread(p10 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(5)

for file in os.listdir(p2): #pause
    img = cv2.imread(p2 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(6)

for file in os.listdir(p1): #unpause
    img = cv2.imread(p1 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(7)


for file in os.listdir(p5): #next
    img = cv2.imread(p5 + '\\' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    fd=extract(img)
    data.append(np.array(fd).flatten())
    labels.append(8)


le = LabelEncoder()
labels = le.fit_transform(labels)

traindata1=np.array(data)
trainlabels1 = np.array(labels)

model = LinearSVC()
model.fit(traindata1, trainlabels1)

test_path='c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\test'
outputpath=roi.create_output_folder(test_path)
l=os.listdir(outputpath)
testdata=[]

for file in l:

        img = cv2.imread(outputpath + '\\' + file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))       
        fd=extract(img)
        testdata=[np.array(fd).flatten()]
        predictions = model.predict(np.array(testdata))
        pred=predictions[0]

        if pred==0:
            print("one")
            play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\1.mp3")
            cur=1
        elif pred==1:
            print("two")
            play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\2.mp3")
            cur=2
        elif pred==2:
            print("three")
            play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\3.mp3")
            cur=3
        elif pred==3:
            print("four")
            play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\4.mp3")
            cur=4
        elif pred==4:
            print("five")
            play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\5.mp3")
            cur=5
        elif pred==5:
            print("pause")
            pause_sound()
        elif pred==6:
            print("unpause")
            unpause_sound()
        elif pred==7:
            print("next")
            if cur==1:
                pause_sound()
                play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\2.mp3")
                cur=2
            elif cur==2:
                pause_sound()
                play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\3.mp3")
                cur=3
            elif cur==3:
                pause_sound()
                play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\4.mp3")
                cur=4
            elif cur==4:
                pause_sound()
                play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\5.mp3")
                cur=5
            elif cur==5:
                pause_sound()
                play_sound("c:\\Users\\HP\\Downloads\\COL780 A3\\COL780 A3\\dataset\\songs\\1.mp3")
                cur=1
        time.sleep(5)






