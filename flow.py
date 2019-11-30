from time import time
from loss import GeneratorLoss
import numpy as np
from os import scandir
from generator import Generator
from discriminator import Discriminator
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



def train(*, epoch_count, batch_size, hr_images, lr_images):
    """if you set batch size to 0, the batch size will be of whole training dataset"""

    dis = Discriminator()
    dis.compile(loss="binary_crossentropy")

    gen = Generator()
    gen.compile(loss=GeneratorLoss(dis))

    sequencer = ParallelLoader(
        x_template = lr_images,
        y_template = hr_images,
        batch_size = batch_size,
        epoch_count = epoch_count,
    )

    gctime = time()
    for epoch, bnumber, blr, bhr in sequencer:
        start = time()

        bsr = gen.predict(blr)

        loss_d = dis.train_on_batch(
            np.concatenate([bsr, bhr]),
            np.concatenate([
                np.full((len(bsr), 1), 0),
                np.full((len(bhr), 1), 1)
            ])
        )

        loss_g = gen.train_on_batch(blr, bhr)

        total = time() - start
        print(json.dumps({
            'epoch_no': epoch, 
            'batch_no': bnumber, 
            'loss_d': str(loss_d),
            'loss_g': str(loss_g),
            'time': "%.2fs"%total 
        }))
        if (time() - gctime) > 420: 
            gctime = time()
            gc.collect()
    
    sequencer.join()
    return dis, gen




        


        



