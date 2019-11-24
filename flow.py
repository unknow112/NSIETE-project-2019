from time import time
from skimage.io import imread, imsave
from skimage.color import gray2rgb
import numpy as np
from os import scandir
from gan import Gan
import gc
from parallel_loader import ParallelLoader
import json
        
    
def is_img(x):
    return x.name.split('.')[-1] in {'png','jpg'}
    

LR_DIR='inputdata'
HR_DIR='outputdata'

LR_IMAGES= sorted(filter(is_img ,scandir(path=LR_DIR)), key=lambda x: x.name)
HR_IMAGES= sorted(filter(is_img ,scandir(path=HR_DIR)), key=lambda x: x.name)
assert len(LR_IMAGES) == len(HR_IMAGES)
assert all(map(
    lambda X: X[0].name == X[1].name, 
    zip(LR_IMAGES, HR_IMAGES)
))


LR_IMAGES = list(map(lambda x: x.path, LR_IMAGES))
HR_IMAGES = list(map(lambda x: x.path, HR_IMAGES))





def load_and_normalize(image):
    res = imread(image)
    res = (res - 127.5) / 127.5
    if len(res.shape) == 2:
        res = gray2rgb(res)
    return res


def train(*, epoch_count, batch_size, hr_images, lr_images):
    """if you set batch size to 0, it will mean that there will be only one batch"""

    gan = Gan()
    gan.compile()

    sequencer = ParallelLoader(
        x_template = lr_images, 
        y_template = hr_images, 
        batch_size = batch_size, 
        epoch_count = epoch_count,
        loader_f = load_and_normalize
    )

    for epoch, bnumber, blr, bhr in sequencer:
        start = time()
        gctime = time()
            
        bsr = gan.g.predict(blr)

        gan.d.trainable=True
        loss_d_fake = gan.d.train_on_batch(
            bsr,
            np.full((len(bsr),1), -1)
        )
        
        loss_d_real = gan.d.train_on_batch(
            bhr,
            np.ones((len(bhr),1))
        )
        gan.d.trainable=False

        loss_gan = gan.train_on_batch(blr,  [bhr, np.ones((len(bhr),1))])

        total = time() - start
        print(json.dumps({
            'epoch_no': epoch, 
            'batch_no': bnumber, 
            'loss_d_fake': loss_d_fake, 
            'loss_d_real': loss_d_real, 
            'loss_gan': loss_gan, 
            'time': "%.2fs"%total 
        }))
        if (time() - gctime) > 420: 
            gctime = time()
            gc.collect()
    
    sequencer.join()
    return gan




        


        



