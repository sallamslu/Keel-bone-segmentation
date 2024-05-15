from tensorflow.keras.layers import *
from glob import glob
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np


# ------ load_split_data
def load_split_data(path, n_splits=5, test_prc=0.3 ,rand=42):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masked/*")))

    print("Length of images: ", len(images))
    print("Length of masks: ", len(masks))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rand) 
    splits = list(kf.split(images, masks))
    dataset_splits = []

    for fold, (train_index, test_index) in enumerate(splits):
        x_train, x_test = [images[i] for i in train_index], [images[i] for i in test_index]
        y_train, y_test = [masks[i] for i in train_index], [masks[i] for i in test_index]

        #valid_size = int(len(train_index) * 0.3)  # 30% of the training data for validation
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=test_prc, random_state=rand)

        dataset_splits.append((x_train, y_train, x_valid, y_valid, x_test, y_test))

    return dataset_splits


# --------------- some functions required for tf_dataset and tf_dataset_show 
## ---------read_image read and normalize works for both image and masks as both are grayscaled
def read_image(path, h, w):
     path = path.decode()  # convert forward slashes into the correct kind of slash for the current OS
     x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # read image as matrix
     x = cv2.resize(x, (h, w))   # resize
     x = x/255.0                     # normalization
     x = np.expand_dims(x, axis=-1)
     return x

## ---------tf_parse to parse a single image and mask path and set the shapes required.
def tf_parse(x, y, h, w):
    def _parse(x, y):
        x = read_image(x, h, w)
        y = read_image(y, h, w)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([h, w, 1])
    y.set_shape([h, w, 1])
    return x, y

## ---------tf_dataset to create a tf.data pipeline .. takes a list of images, masks paths and the batch size.
def tf_dataset(x, y, h, w,batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, h, w))  # to do transformation required
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

## ---------tf_dataset_show
def tf_dataset_show(x, y, h, w,batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, h, w))  # to do transformation required
    dataset = dataset.batch(batch)
    #dataset = dataset.repeat()
    return dataset


# ----------------- some functions when augmentaion is yes, load_image and generator 
# ------------load_image by Mehdi from KTH >> mask labels is 0 and 1
def load_image(data_list, h, w,  mask=False):
    image_data = np.zeros((len(data_list), h, w, 1), dtype='float32')
    for i in range(len(data_list)):
        img = cv2.imread(data_list[i], 0)
        img = cv2.resize(img[:, :], (h, w))
        img = img.reshape(h, w) / 255.  # then 255 which is white will be equal 1
        if mask:
            img[img > 0]  = 1   # label 1  for white
            img[img != 1] = 0   # label 0  for black
        image_data[i, :, :, 0] = img
    return image_data

# ------------generator by Mehdi from KTH
def generator(x_train, y_train, batch_size):
    n_train_sample = len(x_train)
    #backgound_value = x_train.min()
    data_gen_args = dict(rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         # cval = backgound_value,
                         zoom_range=0.2,
                         horizontal_flip=True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    while True:
        for ind in (range(0, n_train_sample, batch_size)):
            batch_img = x_train[ind:ind + batch_size]
            batch_label = y_train[ind:ind + batch_size]

            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            length = len(batch_img)
            if length == batch_size:
                pass
            else:
                for tmp in range(batch_size - length):
                    batch_img = np.append(batch_img, np.expand_dims(batch_img[-1], axis=0), axis=0)
                    batch_label = np.append(batch_label, np.expand_dims(batch_label[-1], axis=0), axis=0)

            image_generator = image_datagen.flow(batch_img, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)

            mask_generator = mask_datagen.flow(batch_label, shuffle=False,
                                               batch_size=batch_size,
                                               seed=1)

            image = image_generator.next()
            label = mask_generator.next()
            # binarizer = Binarizer().fit(label)
            # label = binarizer.transform(label)                    #(label, copy=False) # threshold default is 0.0
            label = np.where(label > 0, 1, 0).astype(np.float32)  # to binarize the mask
            yield image, label

# -------------dice and dice_loss 
def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, 'float64')
    y_pred = K.cast(y_pred, 'float64')
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union        = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice         = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred):
  return 1.-dice_coef(y_true, y_pred)



# ----------build_unet  
class CONV:
    def conv_block(data, base, batchnorm=False):
        # --------------- 1st block
        x = Conv2D(filters=base, kernel_size=(3, 3), padding='same')(data)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # --------------- 2nd block
        x = Conv2D(filters=base, kernel_size=(3, 3), padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

def build_unet(h, w, c, base, classes, dropout=True, batchnorm=True):
    num_filters = [base, base * 2, base * 4, base * 8]
    inputs = Input((h, w, c))

    skip_x = []
    x = inputs

    # ----------------- Encoder
    for f in num_filters:
        x = CONV.conv_block(x, f, batchnorm=True)
        skip_x.append(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        if dropout:
            x = Dropout(0.2)(x)

            # ----------------- Bridge
    x = CONV.conv_block(x, num_filters[-1], batchnorm=True)
    # x = MaxPooling2D(pool_size=(2, 2), padding="same") (x)  # no need to this layer
    if dropout:
        x = Dropout(0.2)(x)

    num_filters.reverse()
    skip_x.reverse()

    # ----------------- Decoder
    for i, f in enumerate(num_filters):
        #x = UpSampling2D((2, 2))(x)  # Conv2DTranspose(f, (2,2)) (x)
        x = Conv2DTranspose(filters=f, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        xs = skip_x[i]
        #height, width = determine_crop(xs, x)
        #crop_xs = Cropping2D(cropping=(height, width))(xs)
        x = Concatenate()([x, xs])
        if dropout:
            x = Dropout(0.2)(x)
        x = CONV.conv_block(x, f, batchnorm=True)

    # ----------------- Output
    out = Conv2D(classes, (1, 1), padding="same", activation='sigmoid')(x)

    # ----------------- input and output
    model = Model(inputs=inputs, outputs=out)
    model.summary()
    return model



# ----------------- some functions to plot_metrics of tarining and validation
## ----------find_metric_key
def find_metric_key(history_dict, metric_name):
    for key in history_dict.keys():
        if metric_name in key:
            return key
    return None

## ---------plot_metrics
def plot_metrics(history, save_dir):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print("Available metrics:", list(history.history.keys()))

    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'auc', 'dice_coef']

    for metric_name in metrics_to_plot:
        metric_key = find_metric_key(history.history, metric_name)
        val_metric_key = find_metric_key(history.history, f'val_{metric_name}')

        if metric_key and val_metric_key:
            # Plot training & validation metric values
            plt.plot(history.history[metric_key])
            plt.plot(history.history[val_metric_key])
            plt.title(f'Model {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.capitalize())
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(save_dir, f'model_{metric_name.lower()}.jpg'))
            plt.close()



#------------- 2 functions to show the original and true and predicted keels for testing data
## -----------display  
#def display(display_list):
#    plt.figure(figsize=(15, 5))
#    title = ['Input Image', 'True Mask', 'Predicted Mask']
#    #title = ['Input Image', 'Predicted Mask']
#    for i in range(len(display_list)):
#        plt.subplot(1, len(display_list), i + 1)
#        plt.title(title[i])
#        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#        plt.axis('off')
#        #plt.show()

## ------------show_img_true_mask
#def show_img_true_mask(dataset=None, model=None, path=None):
#    for i, (image, mask) in enumerate(dataset):
#        pred_mask = model.predict(image)
#        pred_mask *= 255.0
#        #print(pred_mask.min())
#        #print(pred_mask.max())
#        #print(np.unique(pred_mask, return_counts=True))
#        display([image[0], mask[0], pred_mask[0]])
#        #display([image[0], pred_mask[0]])
#        plt.savefig(os.path.join(path, f'img_true_mask{i}.png'), dpi=800)
#        plt.close()


#------------- 2 functions to show the original and true and predicted keels for testing data
## -----------display 
def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ['A. Input Image', 'B. True Mask', 'C. Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

## ------------show_img_true_mask
def show_img_true_mask(dataset=None, model=None, path=None):
    if not os.path.exists(path):
        os.makedirs(path)

    for batch_index, (images, masks) in enumerate(dataset):
        pred_masks = model.predict(images)
        pred_masks *= 255.0  # Scale the predicted masks
        
        for i in range(images.shape[0]):
            display([images[i], masks[i], pred_masks[i]])
            plt.savefig(os.path.join(path, f'img_true_mask_batch{batch_index}_img{i}.png'), dpi=800)
            plt.close()


