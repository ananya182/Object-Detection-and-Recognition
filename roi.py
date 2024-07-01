import os
import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, 
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

def increase_bbox(bbox, scale_factor):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h

def create_output_folder(dataset_folder):
    output_folder = os.path.join(dataset_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                i=0
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    landmark_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmark_points.append([x, y])

                
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor)
                    x=max(0,x)
                    y=max(0,y)
                    cropped_image = img[y:y+h, x:x+w]

                    output_path = os.path.join(output_folder, filename.rstrip(".jpg")+"_"+str(i)+".jpg")
                    cv2.imwrite(output_path, cropped_image)
                    i+=1
    return output_folder

