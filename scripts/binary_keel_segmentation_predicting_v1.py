
"""
Script Name: binary_keel_segmentation_prediciting_v1.py
Author:      Moh. Sallam
Description: [this script is to visualize 
              the original images+true and predicted keels
              N.B: this script uses 
              the functions in "segmentation_functions.py" and
              the trained models from "binary_keel_segmentation_learning_v1.py"

Date:        [Most of this script was created on Jan 2022 and updated on Mar 2024 ]
"""


# --------- import required functions
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import CustomObjectScope
import pandas as pd
import cv2
import numpy as np

from segmentation_functions import load_split_data, tf_dataset_show , dice_coef
from segmentation_functions import show_img_true_mask


# --------- set some parameters
batch = 8
n_splits = 5     # for splitting data
test_prc = 0.4   # test /(test + valid)
h = 256
w = 256
c = 1


# Specify the data (images and masks paths)
main_path = '../data/'
output_dir = os.path.join(main_path, "output")
show_dir   = os.path.join(output_dir, "show")


dataset_splits = load_split_data(path=main_path, n_splits=n_splits, test_prc=test_prc)

for fold, (x_train, y_train, x_valid, y_valid, x_test, y_test) in enumerate(dataset_splits):
    print(f"Fold {fold + 1}:")
    print("Training data:", len(x_train), len(y_train))
    print("Validation data:", len(x_valid), len(y_valid))
    print("Test data:", len(x_test), len(y_test))

    # Modify output directories
    output_dir_fold = os.path.join(output_dir, f"output_fold_{fold + 1}")
    show_dir_fold = os.path.join(output_dir_fold, "show")
    os.makedirs(output_dir_fold, exist_ok=True)
    os.makedirs(show_dir_fold, exist_ok=True)


    with CustomObjectScope({'dice_coef': dice_coef}):
        model = tf.keras.models.load_model(os.path.join(output_dir_fold, 'model.h5'))
     
    # use tf_dataset_show NOT tf_dataset
    test_dataset = tf_dataset_show(x_test, y_test, h, w, batch=batch)

    test_steps = (len(x_test) // batch)
    if len(x_test) % batch != 0:
        test_steps += 1

    result = model.evaluate(test_dataset, steps=test_steps)
    evaluation_dict = {
        "Fold": [fold + 1],
        "Training data": [len(x_train)],
        "Validation data": [len(x_valid)],
        "Test data": [len(x_test)],
        "Test Loss": [round(result[0], 2)],
        "Test Accuracy": [round(result[1], 2)],
        "Test Precision": [round(result[2], 2)],
        "Test Recall": [round(result[3], 2)],
        "Test Dice Coefficient": [round(result[4], 2)]
    }
    df = pd.DataFrame(evaluation_dict)
    df.to_excel(os.path.join(output_dir_fold, "test_metrics_prediciting_script.xlsx"), index=False)

    show_img_true_mask(dataset=test_dataset, model=model, path=show_dir_fold)
