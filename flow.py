from time import time
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


LR_IMAGES = np.array(list(map(lambda x: x.path, LR_IMAGES)))
HR_IMAGES = np.array(list(map(lambda x: x.path, HR_IMAGES)))

def mkmodel():
    gan = Gan()
    gan.compile()
    return gan

def train(*,gan,  epoch_count, batch_size, hr_images, lr_images):
    """if you set batch size to 0, it will mean that there will be only one batch"""


    sequencer = ParallelLoader(
        x_template = lr_images, 
        y_template = hr_images, 
        batch_size = batch_size, 
        epoch_count = epoch_count,
    )

    gctime = time()
    for epoch, bnumber, blr, bhr in sequencer:
        start = time()

        bsr = gan.g.predict(blr)

        gan.d.trainable=True
        gan.d.compile()
        loss_d = gan.d.train_on_batch(
            np.concatenate([bsr, bhr]),
            np.concatenate([
                np.full((len(bsr), 1), -1),
                np.ones((len(bhr),1))
            ])
        )
        gan.d.trainable=False
        gan.d.compile()

        loss_gan = gan.train_on_batch(blr,  [bhr, np.ones((len(bhr),1))])

        total = time() - start
        print(json.dumps({
            'epoch_no': epoch, 
            'batch_no': bnumber, 
            'loss_d': str(loss_d),
            'loss_gan': str(loss_gan),
            'time': "%.2fs"%total 
        }), flush=True)
        if (time() - gctime) > 420: 
            gctime = time()
            gc.collect()
    
    sequencer.join()
    return gan




        


        



