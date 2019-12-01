from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy


vgg19 = VGG19(input_shape=(128,128,3), include_top=False)

def vgg_loss(y_true, y_pred):
    return mean_squared_error(vgg19(y_true), vgg19(y_pred))

def inverted_binary_crossentropy(y_true, y_pred):
    return 1 - binary_crossentropy(y_true, y_pred)


