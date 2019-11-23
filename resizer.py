from PIL import Image

def resize(new_size, image):
    w,h = image.size
    assert w == h
    return image.resize((new_size,new_size), Image.ANTIALIAS)


def get_crop_area(width, height):
    assert height != width
    shift = abs(width - height) // 2
    square = min(width, height)

    rest = shift + square

    if width > height:
        return shift, 0, rest, square

    if height > width:
        return 0, shift, square, rest


def square(img):
    width, height = img.size
    if width == height:
        return img

    new_area = get_crop_area(*img.size)
    return img.crop(new_area)
