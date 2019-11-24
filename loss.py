from tensorflow.keras.applications.vgg19 import VGG19

vgg = VGG19(
    include_top = False,
    input_shape = (128,128,3)
)

def generator_loss(yp, yt):
    vyp = vgg.predict(yp)
    vyt = vgg.predict(yt)

