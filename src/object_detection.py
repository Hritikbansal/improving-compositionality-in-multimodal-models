import os
import csv
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()

parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
parser.add_argument("--save_train_data", type = str, default = None, help = "file name for the save train data with detected objects")
parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")

args = parser.parse_args()


root = os.path.dirname(args.train_data)
df = pd.read_csv(f'{args.train_data}')
images = df['image'].tolist()
captions = df['caption'].tolist()

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model = model.to(0)

transform = T.Compose([T.ToTensor()])

filename = os.path.join(root, args.save_train_data)

i = 0
while i < len(images):
    subset_images = images[i : i + args.batch_size]
    subset_captions = captions[i : i + args.batch_size]
    run_images = list(map(lambda x: os.path.join(root, x), subset_images))
    results = model(run_images).pandas().xyxy
    for index in tqdm(range(len(results))):
        result = results[index]
        confidence = result['confidence'].tolist()
        names  = result['name'].tolist()
        zipped = list(zip(names, confidence))
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([str(subset_images[index]), subset_captions[index], str(zipped)])
    i += args.batch_size