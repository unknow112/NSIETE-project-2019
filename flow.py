
from skimage.io import imread, imsave
import numpy as np
from os import scandir,path
from gan import dis_loss, gen_loss

def to_batch(a, bsize):
    if len(a) % bsize == 0:
        return np.array_split(a, len(a) // bsize)
    else:
        remain = len(a) % bsize
        return np.array_split(a[:-remain], len(a) // bsize) + [a[-remain:]]


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


EPOCH_COUNT = 10
BATCH_SIZE = 4
SAMPLE_SIZE = len(LR_IMAGES)


#gan = Gan() #not implemented


"""


for _ in range(EPOCH_COUNT):
    lr , hr = shuffle(LR_IMAGES, HR_IMAGES)

    lr_batches = to_batch(lr, BATCH_SIZE)
    hr_batches = to_batch(hr, BATCH_SIZE)


    for blr, bhr in zip(lr_batches,hr_batches):
        bsr = gan.g.call(blr)

        whole_batch_x = np.concatenate((bsr, bhr))
        whole_batch_y = np.concatenate((np.zeros(bsr.shape[0],dtype=int),np.ones(bhr.shape[0], dtype=int)))

        whole_batch_yp = gan.d.call(whole_batch_x)




        


        




"""