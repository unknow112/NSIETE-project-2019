import tensorflow.keras  as keras
from generator import Generator
from discriminator import Discriminator
import tensorflow_gan as tfgan


def gen_loss():
    pass

def dis_loss():
    pass






class Gan(keras.Model):
    def __init__(self, residual_block_count=16):
        super(Gan, self).__init__()

        self.g = Generator()
        self.d = Discriminator()



    #def fit(self, **kwargs):
    #    pass
        










