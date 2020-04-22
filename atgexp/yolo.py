import numpy as np
import argparse
import time
import cv2
import os
from .apps import AtgexpConfig
from django.conf import settings

import urllib


def predict(image_name):
    image_path = os.path.join(settings.MEDIA_ROOT,str(image_name))
    argconfidence = 0.5
    threshold = 0.3    
    labelsPath = AtgexpConfig.labelsPath
#     LABELS = open(labelsPath).read().strip().split("\n")    
    LABELS = AtgexpConfig.LABELS
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    
#     weightsPath = AtgexpConfig.weightsPath
#     configPath = AtgexpConfig.configPath
#     
#     net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    net = AtgexpConfig.net

    print(image_path)
    image = image_path
    image = cv2.imread(image)
    image = cv2.resize(image, (600,400))
    (H, W) = image.shape[:2]
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > argconfidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, argconfidence,
    threshold)
    
    data = []
    if len(idxs) > 0:
    # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])
    
            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(text)
            data.append(LABELS[classIDs[i]])
            
    
    return data