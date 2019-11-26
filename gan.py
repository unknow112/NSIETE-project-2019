import tensorflow.keras  as keras
from generator import Generator
from discriminator import Discriminator
import tensorflow_gan as tfgan
import loss

class Gan(keras.Model):
    def __init__(self, residual_block_count=16):
        super(Gan, self).__init__()

        self.g = Generator()
        self.d = Discriminator()

    def compile(self, **kwargs):
        opt = keras.optimizers.Adam(0.001, 0.5)

        #self.d.compile(optimizer=opt,loss=loss.content_loss)
        self.d.compile(optimizer=opt,loss=tfgan.losses.wargs.wasserstein_discriminator_loss)
        super().compile(loss=loss.adversarial_loss, optimizer=opt)
       


    def call(self, x):
        y = self.g(x)
        rating = self.d(y)
        return rating











