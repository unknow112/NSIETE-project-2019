def resize(new_size, image):
    w,h = image.size
    assert w == h
    return image.resize((new_size,new_size), Image.ANTIALIAS)


def get_crop_area(width, height):
    assert height != width
    shift = abs((width - height)) // 2
 
    if width > height:
        rest = width - shift
        return shift, 0, rest, height

    if height > width:
        rest = height - shift
        return 0, shift, width, rest


def square(img):
    width, height = img.size
    if width == height:
        return img

    new_area = get_crop_area(*img.size)
    return img.crop(new_area)