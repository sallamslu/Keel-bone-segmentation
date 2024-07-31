
 
"""
Script Name: binary_keel_segmentation_learning_v1.py
Author:      Moh. Sallam
Description: [this script is to train specific U-net to segemnt keel bone
              from the whole-body x-ray images of laying hens,including:
              1) split the dataset into 5 random sets, each set is randolmly
                 splitted into 80% for training, 12% for validation, and
                 8% for testing.    
              1) evaluate training vs validation for loss and dice-coefficient.
              2) evaluate the model on the testing sets for the dice-coefficient.   
              
              3) you can also visualize the original images of testing sets, 
                 and their true and predicted keels, by runing the script
                 "binary_keel_segmentation_predicting_v1.py"
              
              N.B: this script uses the functions in the file 
                   "segmentation_functions.py" that also written by Moh Sallam ]

Date:        [Most of this script was created on Jan. 2022 and updated on Mar. 2024 ]
"""
# ---------- import required functions

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.utils import CustomObjectScope
import pandas as pd

from segmentation_functions import load_split_data, tf_dataset, dice_coef, build_unet
from segmentation_functions import load_image, generator
from segmentation_functions import plot_metrics


# ----------- set and print parameters
output = "output"   # it is already there "parent_directory/data/output"
lr = 0.0001
batch = 8
augment = "no"      # use "yes" or "no" to augment or no

epochs =  200
n_splits = 5     # for splitting data
test_prc = 0.4   # test /(test + valid)
h = 256
w = 256
c = 1
base = 8
classes = 1

loss = "binary_crossentropy"
metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef]


print('Parameters used: ')
print('Number of splits for cross-validation: ', n_splits)
print('Test dataset / Test dataset + Valid dataset : ', test_prc)
print('Augment: ', augment)
print('Batch: ', batch)
print('Learning Rate: ', lr)
print('Base: ', base)
print('Classes: ', classes)
print('Loss: ', loss)
print('Epochs: ',  epochs)

# ----------- specify the data (images and masks paths)
main_path = '../data/'
output_dir = os.path.join(main_path, output)
os.makedirs(output_dir, exist_ok=True)

# ----------- load and split the data
dataset_splits = load_split_data(path=main_path, n_splits=n_splits, test_prc=test_prc)

for fold, (x_train, y_train, x_valid, y_valid, x_test, y_test) in enumerate(dataset_splits):
    print(f"Fold {fold + 1}:")
    print("Training data:", len(x_train), len(y_train))
    print("Validation data:", len(x_valid), len(y_valid))
    print("Test data:", len(x_test), len(y_test))

    # output directories per folds
    output_dir_fold = os.path.join(output_dir, f"output_fold_{fold + 1}")
    show_dir_fold = os.path.join(output_dir_fold, "show")
    os.makedirs(output_dir_fold, exist_ok=True)
    os.makedirs(show_dir_fold, exist_ok=True)

    # -------- if augment no or yes
    if augment == "no":
        print("Not augmenting data.")
        train_dataset = tf_dataset(x_train, y_train, h, w, batch=batch)
        valid_dataset = tf_dataset(x_valid, y_valid, h, w, batch=batch)

    if augment == "yes":
        print("Augmenting data...")
        x_train = load_image(x_train, h, w, mask=False)
        y_train = load_image(y_train, h, w, mask=True)
        x_valid = load_image(x_valid, h, w, mask=False)
        y_valid = load_image(y_valid, h, w, mask=True)

        train_generator = generator(x_train, y_train, batch_size=batch)
        valid_generator = generator(x_valid, y_valid, batch_size=batch)

    # ---------- call the model
    model = build_unet(h, w, c, base, classes, dropout=True, batchnorm=True)

    # ---------- compile the model
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=metrics)

    # ---------- set callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(output_dir_fold, 'model.h5')),
        CSVLogger(os.path.join(output_dir_fold, 'CSVLogger.csv')),
        TensorBoard(output_dir_fold) ,
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),  # reduce lr when a metric stopped improving
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]

    # ---------- fit the model
    train_steps = len(x_train) // batch
    valid_steps = len(x_valid) // batch
    if len(x_train) % batch != 0:
        train_steps += 1
    if len(x_valid) % batch != 0:
        valid_steps += 1

    if augment == "no":
        history = model.fit(train_dataset,
                            validation_data=valid_dataset,
                            epochs=epochs,
                            steps_per_epoch=train_steps,
                            validation_steps=valid_steps,
                            callbacks=callbacks)
    if augment == "yes":
        history = model.fit(train_generator,
                            validation_data=valid_generator,
                            epochs=epochs,
                            steps_per_epoch=train_steps,
                            validation_steps=valid_steps,
                            callbacks=callbacks)


    # ------- plot training and validation metrics
    plot_metrics(history, save_dir=output_dir_fold)

    # -------- evaluate the model on the testing sets (images never seen by the model)
    with CustomObjectScope({'dice_coef': dice_coef}):
        model = tf.keras.models.load_model(os.path.join(output_dir_fold, 'model.h5'))

    test_steps = (len(x_test) // batch)
    if len(x_test) % batch != 0:
        test_steps += 1

    test_dataset = tf_dataset(x_test, y_test, h, w, batch=batch)
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
    df.to_excel(os.path.join(output_dir_fold, "test_metrics_learning_script.xlsx"), index=False)

