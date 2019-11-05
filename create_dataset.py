"""
requirements

take dataset iterator and output folder , crop (find bigest centered square) and resize images

Input: 
iterator of DirEntry objects path member is full path and DirEntries are all images!
dictionary like:
{
    int squared_output_resolution : str  full_output_folder
}
Output: 
images inside outputdir
"""

def get_crop_area(width, height):
    #if width > height:
    pass

def square(img):
    width, height = img.size
    if width == height:
        return img

    new_area = get_crop_area(*img.size)
    return img.crop(new_area)

def resize():
    pass

def workflow():
    pass

def main():
    pass


