from PIL import Image
import numpy as np
import random
import os

path = r'/data0/datasets/MSCOCO-2017/train/'
images = os.listdir(os.path.join(path, 'images/'))
for image in images:
    im = Image.open(os.path.join(path, 'images/', image))

    parts = []
    height, width = im.size
    for i in range(0, 3):
        for j in range(0, 3):
            box = (i*height//3, j*width//3, (i+1)*height//3, (j+1)*width//3)
            parts.append(im.crop(box))

    neg_img = im.copy()
    while (np.asarray(neg_img) == np.asarray(im)).all():
        random.shuffle(parts)
    
        for i, part in enumerate(parts):
            offset = ((i%3)*height//3, (i//3)*width//3)
            neg_img.paste(part, offset)
    
    neg_img.save(os.path.join(path, 'permuted_images/', image.split('/')[-1]))
