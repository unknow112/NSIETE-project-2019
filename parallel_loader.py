import multiprocessing as mp
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb


def load_and_normalize(image):
    res = imread(image)
    res = (res - 127.5) / 127.5
    if len(res.shape) == 2:
        res = gray2rgb(res)
    return res


def to_batch(a, bsize):
    if bsize == 0:
        return [a.copy()]
    bcount = len(a) // bsize
    if len(a) % bsize == 0:
        return np.array_split(a, bcount)
    else:
        wholepart = bcount * bsize
        return np.array_split(a[:wholepart], bcount) + [a[wholepart:]]


def shuffle(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class EndItem():
    pass


class ParallelLoader():
    def __init__(self, *,x_template, y_template, batch_size, prepare=10, epoch_count):
        self.x = x_template
        self.y = y_template
        self.bsize = batch_size
        self.epoch_count = epoch_count
        self.mp_manager = mp.Manager()
        self.prepared = self.mp_manager.Queue(prepare)
        self.task = mp.Process(target=ParallelLoader.prepare, args=(self,))
        self.task.start()
    
    def join(self):
        self.task.join()

    def __iter__(self):
        return self

    def __next__(self):
        next = self.prepared.get()
        if next == EndItem:
            raise StopIteration
        else:
            return next

    def prepare(self):
        for epoch in range(self.epoch_count):
            self.x , self.y = shuffle(self.x, self.y)

            x_batches_t = to_batch(self.x, self.bsize)
            y_batches_t = to_batch(self.y, self.bsize)

            for batch,(xbt, ybt) in enumerate(zip(x_batches_t, y_batches_t)):
                xb = np.array(list(map(load_and_normalize, xbt)))
                yb = np.array(list(map(load_and_normalize, ybt)))
                self.prepared.put((epoch,batch,xb,yb))
        
        self.prepared.put(EndItem)
