import multiprocessing as mp
import numpy as np


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
    def __init__(self, *,x_template, y_template, batch_size, prepare=10, epoch_count ,loader_f):
        self.x = x_template
        self.y = y_template
        self.bsize = batch_size
        self.prepared = mp.Queue(prepare)
        self.epoch_count = epoch_count
        self.loader = loader_f
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

            for batch,(xbt, ybt) in enumerate(zip(x_batches, y_batches)):
                xb = np.array(list(map(self.loader, xbt)))
                yb = np.array(list(map(self.loader, ybt)))
                self.prepared.put((epoch,batch,xb,yb))
        
        self.prepared.put(EndItem)

        



