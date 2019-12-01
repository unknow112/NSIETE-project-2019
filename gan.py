import tensorflow.keras  as keras
from generator import Generator
from discriminator import Discriminator
from loss import vgg_loss

class Gan(keras.Model):
    def __init__(self, residual_block_count=16):
        super(Gan, self).__init__()

        self.g = Generator()
        self.d = Discriminator()

    def compile(self, **kwargs):
        opt = keras.optimizers.Adam(0.001, 0.5)

        self.g.compile(optimizer=opt,loss=vgg_loss)
        self.d.compile(optimizer=opt,loss="binary_crossentropy")
        super().compile(loss=[vgg_loss, 'binary_crossentropy'], optimizer=opt)
       

    def call(self, x):
        y = self.g(x)
        rating = self.d(y)
        return y, rating











