# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:13:29 2019

@author: Lakshay
"""

import cv2
import imutils.paths as paths

import face_recognition
import pickle
import os


dataset = "E:\\training\\dataset\\"# path of the data set 
module = "E:\\training\\dataset\\encoding1.pickle" # were u want to store the pickle file 

imagepaths = list(paths.list_images(dataset))
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagepaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    boxes = face_recognition.face_locations(rgb, model= "hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
       knownEncodings.append(encoding)
       knownNames.append(name)
       print("[INFO] serializing encodings...")
       data = {"encodings": knownEncodings, "names": knownNames}
       output = open(module, "wb") 
       pickle.dump(data, output)
       output.close()