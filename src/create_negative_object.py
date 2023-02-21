import os
import csv
import random
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
args = parser.parse_args()


df = pd.read_csv(args.train_data)
obj_to_index = defaultdict(list)

for j in tqdm(range(len(df))):
    objs = list(set([x[0] for x in eval(df.iloc[j]['objects'])]))
    for obj in objs:
        obj_to_index[obj].append(j)
print(len(obj_to_index.keys()))
neg_images = []
for index in tqdm(range(len(df))):
    objs = list(set([x[0] for x in eval(df.iloc[index]['objects'])]))
    if len(objs) == 1:
        ni_indices = obj_to_index[objs[0]]
        ni_indices.remove(index)
        if len(ni_indices) == 0:
            neg_images.append("")
        else:
            neg_index = random.choice(ni_indices)
            image_location = df.iloc[neg_index]['image']
            neg_images.append(image_location)
    elif len(objs) > 1:
        image_location = ""
        random.shuffle(objs)
        for x in range(len(objs) - 1):
            for y in range(x + 1, len(objs)):
                ni_indices_1 = obj_to_index[objs[x]]
                ni_indices_2 = obj_to_index[objs[y]]
                ni_indices = list(set(ni_indices_1) and set(ni_indices_2))
                ni_indices.remove(index)
                if len(ni_indices):
                    neg_index = random.choice(ni_indices)
                    image_location = df.iloc[neg_index]['image']
                    break
        if image_location:
            neg_images.append(image_location)
        else:
            obj = random.choice(objs)
            ni_indices = obj_to_index[obj]
            ni_indices.remove(index)
            if len(ni_indices):
                neg_index = random.choice(ni_indices)
                image_location = df.iloc[neg_index]['image']
                neg_images.append(image_location)
            else:
                neg_images.append("")
    else:
        neg_images.append("")

data = {'image':df['image'].tolist(), 'caption': df['caption'].tolist(), 'neg_image': neg_images, 'objects': df['objects'].tolist()}
dframe = pd.DataFrame(data)
dframe.to_csv('train_w_neg_image.csv')
