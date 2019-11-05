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

from resizer import square,resize
from os import path
from PIL import Image



def open_images(iterator):
    return map(
        lambda x: { 
            'path': x.path,
            'name': x.name,
            'img_obj': Image.open(x.path)
        },
        iterator
    )


def workflow(image_iterator, output_folders):
    for key in output_folders:
        assert output_folders[key][0] == '/'


    opened_images = open_images(image_iterator)

    print("squaring....")
    squared_images = list(map(lambda x: { **x , 'img_obj': square(x['img_obj']) }, opened_images ))

    for resolution in output_folders:
        resized_images = map(lambda img: {**img, 'img_obj':resize(resolution, img['img_obj'])} , squared_images)

        for image in resized_images:
            print("saving image %s in resolution %dpx" %(image['name'],resolution))
            image['img_obj'].save(path.join(output_folders[resolution],image['name']))