from os import scandir
from os import path
from PIL import Image

"""
1. nacitaj obrazky
2. vypocitaj nove rozlisenie
3. resize and save
"""

"""deprecated
files = scandir(path=ORIG)
filePaths = map(lambda x: x.path, files)
imageObjects = list(map(  lambda x: { 'img_obj' : Image.open(x), "img_name": x.split('/')[-1] } , filePaths   ))
"""

def resize(ratio, image):
    width, height = image.size
    new_width, new_height = int(width*ratio), int(height*ratio)
    return image.resize((new_width,new_height), Image.ANTIALIAS)


"""deprecated
for percentage in (80,50,20):
    resizedImages = map(lambda img: {**img, 'img_obj':resize(percentage/100, img['img_obj'])} , imageObjects)
    for image in resizedImages:
        print("saving image %s in ratio %.2f" %(image['img_name'],percentage/100))
        image['img_obj'].save(path.join(DIRS[percentage],image['img_name']))
"""

""" 
basewidth = 300
#img = Image.open('somepic.jpg')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('sompic.jpg') 
"""
