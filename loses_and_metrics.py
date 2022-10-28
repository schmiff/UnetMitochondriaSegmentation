# taken from
# https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/Computer_Vision/Unet-Brain-MRI-Segmentation-Tensorflow-Keras/utils.py
from keras import backend as K
# Global Parameter
smooth = 100
# 2 * Area of Overlap / Total Pixel of images
def dice_coefficient(y_true , y_pred, smooth=smooth):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten, ) + K.sum(y_pred_flatten)

    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=smooth):
    return -dice_coefficient(y_true, y_pred, smooth)

# Area of Overlap / Area of Union
def iou(y_true, y_pred, smooth=smooth):
    intersection = K.sum(y_true*y_pred)
    sum = K.sum(y_true+y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou
#
def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

