# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:36:48 2025

@author: talon

This File is for training a custome YOLO 11 model from ultralytics that has functionality to classeify objects from the coco8 dataset from ultralytics.

"""
# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %%
#You may have to install ultralytics


from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    device="cpu",
    verbose=False,
    project='Final_Project',  # custom project directory
    name='my_model'
)

# Evaluate the model
# adjust cofidence
metrics = model.val(data="coco8.yaml",classes=[0],conf=.50,batch=16,verbose=False)


# Run inference on an image (optional)
results = model("C:/Users/talon/cs-420/city.jpg")
predicted_img = results[0].plot()
cv2.imshow("TEST", predicted_img)
cv2.waitKey()
cv2.destroyAllWindows()

####################################################################################################################################
'''
Bellow are some sources i used

Downloading and using a yollo11 model:
https://github.com/ultralytics/ultralytics?tab=readme-ov-file

information on database:
https://docs.ultralytics.com/datasets/detect/coco8/

Training model parameters:
https://docs.ultralytics.com/modes/train/
https://docs.ultralytics.com/guides/hyperparameter-tuning/

Valadation model parameters:
https://docs.ultralytics.com/modes/val/

Information on yoolo11:
https://docs.ultralytics.com/models/yolo11/

Saving model to use somewhere else:
https://github.com/ultralytics/ultralytics/issues/10297

'''