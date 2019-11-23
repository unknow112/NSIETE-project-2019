"""
take Images iterator and output folders, crop (find bigest centered square) 
and resize images

use function `workflow(...)`
Output: 
    images inside outputdirs
"""

from multiprocessing import Pool
from resizer import square,resize
from os import path
from PIL import Image
from collections import namedtuple
from itertools import chain

Dir = namedtuple('Dir', ['name','path','prefix'])

class FunctorWrapper():
    def __init__(self, of):
        self.output_folders = {**of} 
    def __call__(self,image):
        try:
            opened_image = Image.open(image.path)
            squared_image = square(opened_image)

        except OSError:
            return image.path

        for resolution in self.output_folders:
            resized_image = resize(resolution, squared_image)
            save_path = self.output_folders[resolution]
            save_name = '_'.join([image.prefix,image.name])
            print("saving image %s in resolution %dpx" %(save_name,resolution))
            resized_image.save(path.join(save_path, save_name))

def workflow(image_iterators, output_folders):
    """
    image_iterators = [
        ...
        (iterator, str::name_prefix),
        ...
    ]
    output_folders = {
        ...
        int::squared_output_resolution : str::full_output_folder
        ...
    }
    """

    for key in output_folders:
        assert output_folders[key][0] == '/'


    concatenated_iterators = list(chain(*list(map(
        lambda TUPLE: list(map(
            lambda x: Dir(x.name, x.path, TUPLE[1]),
            TUPLE[0]
        )),
        image_iterators
    ))))

    with Pool(4) as p:
        status = p.map(FunctorWrapper(output_folders), concatenated_iterators)

    with open("bad_files.log", "w") as f:
        f.writelines(list(filter(lambda X: type(X) == str, status)))
