from time import time
from skimage.io import imread, imsave
import numpy as np
from os import scandir,path
from gan import Gan
import gc
from utils import generator

def to_batch(a, bsize):
    if bsize == 0:
        return [a.copy()]
    else:
        return np.array_split(a, len(a) // bsize)
    

def shuffle(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


LR_DIR='inputdata'
HR_DIR='outputdata'

LR_IMAGES= sorted(map(lambda x: x.name,filter(lambda x: 'png' in x.name ,scandir(path=LR_DIR))))
HR_IMAGES= sorted(map(lambda x: x.name,filter(lambda x: 'png' in x.name ,scandir(path=HR_DIR))))
assert LR_IMAGES == HR_IMAGES


def load_and_normalize(image):
    res = imread(image)
    res = (res - 127.5) / 127.5
    return res

LR_IMAGES = np.array(list(map(
    lambda x: load_and_normalize(path.join(LR_DIR,x)),
    LR_IMAGES
)))

HR_IMAGES = np.array(list(map(
    lambda x: load_and_normalize(path.join(HR_DIR,x)),
    HR_IMAGES
)))



def train(*, epoch_count, batch_size, hr_images, lr_images):
    """if you set batch size to 0, it will mean that there will be only one batch"""

    gan = Gan()
    gan.compile()

    # for epoch in range(epoch_count):
    #     print("doing epoch no %d " % epoch, end='',flush=True)
    #     start = time()

    gan.fit_generator(generator(lr_images,hr_images,batch_size), epochs=epoch_count, steps_per_epoch=len(lr_images))

        # lr , hr = shuffle(lr_images, hr_images)

        # lr_batches = to_batch(lr, batch_size)
        # hr_batches = to_batch(hr, batch_size)


        # for blr, bhr in zip(lr_batches,hr_batches):
        #     print('!', end='', flush=True)
        #     bsr = gan.g.predict(blr)

        #     loss_d_fake = gan.d.train_on_batch(
        #         bsr,
        #         np.full((len(bsr),1), -1)
        #     )
            
        #     loss_d_real = gan.d.train_on_batch(
        #         bhr,
        #         np.ones((len(bhr),1))
        #     )

        #     loss_gan = gan.train_on_batch(blr,  [bhr, np.ones((len(bhr),1))])
        # total = time() - start
        # print(" took %.2fs" % total)
        # print({'epoch_no': epoch, 'loss_d_fake': loss_d_fake, 'loss_d_real': loss_d_real, 'loss_gan': loss_gan, 'time': "%.2fs"%total })
        # gctime = time()
        # if (time() - gctime) > 420: 
        #     gctime = time()
        #     gc.collect()
    return gan




        


        



