from PIL import Image
import os
from resizeimage import resizeimage

count = 0

def resize_image(path_dir:str):
    for f in os.listdir(path_dir):
        try:
            with Image.open(f) as image:
                count +=1
                cover = resizeimage.resize_cover(image, [800,800])
                cover.save('pgm-1_{}.jpg'.format(count),image.format)
                print(count)
                os.remove(f)
        except(OSError) as e:
            print('Bad Image {}{}'.format(f,count))