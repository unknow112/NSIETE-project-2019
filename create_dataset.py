"""
take Images iterator and output folders, crop (find bigest centered square) 
and resize images

use function `workflow(image_iterator, output_folders)`

Input: 
    image_iterator: iterator of objects where `path` member is full file 
        path, `name` member is filename and every object represents valid 
        image. See `os.scandir`

    output_folders: dictionary like:
    ```
    {
        int squared_output_resolution : str  full_output_folder
    }
    ```

Output: 
    images inside outputdirs
"""
from multiprocessing import Pool
from resizer import square,resize
from os import path
from PIL import Image
from collections import namedtuple
Dir = namedtuple('Dir', ['name','path'])

def open_images(iterator):
    return map(
        lambda x: { 
            'path': x.path,
            'name': x.name,
            'img_obj': Image.open(x.path)
        },
        iterator
    )

class FunctorWrapper():
    def __init__(self, of):
        self.output_folders = {**of} 
    def __call__(self,image):
        opened_image = open_images([image])
        squared_image = list(map(lambda x: { **x , 'img_obj': square(x['img_obj']) }, opened_image ))
        for resolution in self.output_folders:
            image = list(map(
                lambda img: {**img, 'img_obj':resize(resolution, img['img_obj'])} , 
                squared_image
            ))[0]
            print("saving image %s in resolution %dpx" %(image['name'],resolution))
            image['img_obj'].save(path.join(self.output_folders[resolution],image['name']))

def workflow(image_iterator, output_folders):
    for key in output_folders:
        assert output_folders[key][0] == '/'

    image_iterator = list(map(lambda x: Dir(x.name, x.path), image_iterator))

    with Pool(4) as p:
        p.map(FunctorWrapper(output_folders), image_iterator)
