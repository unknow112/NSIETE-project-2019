import tensorflow.keras  as keras
from generator import Generator
from discriminator import Discriminator
import tensorflow_gan as tfgan

class Gan(keras.Model):
    def __init__(self, residual_block_count=16):
        super(Gan, self).__init__()

        self.g = Generator()
        self.d = Discriminator()
        self.d.trainable = False

    def compile(self, **kwargs):
        opt = keras.optimizers.Adam(0.001, 0.5)

        gen_loss = tfgan.losses.wasserstein_generator_loss
        dis_loss = tfgan.losses.wasserstein_discriminator_loss

        self.g.compile(optimizer=opt,loss=gen_loss)
        self.d.compile(optimizer=opt,loss=dis_loss)
        super().compile(loss=[gen_loss, dis_loss], optimizer=opt)

    def call(self, x):
        y = self.g(x)
        rating = self.d(y)
        return y, rating











