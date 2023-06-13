import argparse
import torch
import pandas as pd

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from pkgs.openai.clip import load as load_model

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
parser.add_argument("--rotate", default = False, action = "store_true", help = "whether to rotate embeddings or not")
parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

options = parser.parse_args()
device = 'cuda:4'

model, processor = load_model(name = 'ViT-B/32', pretrained = options.pretrained, keep_positional = True, rotate = options.rotate)
model.to(device)

def get_inputs(image, caption):
    captions = processor.process_text(caption)
    pixel_values = processor.process_image(image.convert("RGB"))
    return captions['input_ids'].to(device), captions['attention_mask'].to(device), pixel_values.to(device).unsqueeze(0)

def clipscore(model, output):
    return (model.logit_scale.exp() * output.image_embeds @ output.text_embeds.t()).item()

if not options.pretrained:
    state_dict = torch.load(options.checkpoint, map_location = device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

model.eval()

# df = pd.read_csv('mscoco_order_neg.csv', sep = '\t')
df = pd.read_csv('flickr_order_neg.csv', sep = '\t')

# path = '/data0/datasets/MSCOCO/test/'
path = '/data0/datasets/FLICKR30K/test/'
count = 0
for index, example in tqdm(df.iterrows()):
    image = Image.open(path + example['image'])
    clip_score_c0, clip_score_c1, clip_score_c2, clip_score_c3 = 0, 0, 0, 0
    
    input_c0 = get_inputs(image, example["original_caption"])
    output_c0 = model(input_ids = input_c0[0], attention_mask = input_c0[1], pixel_values = input_c0[2])
    clip_score_c0 = clipscore(model, output_c0)

    if pd.notna(example["shuffle_noun_and_adjectives"]):
        input_c1 = get_inputs(image, example["shuffle_noun_and_adjectives"])
        output_c1 = model(input_ids = input_c1[0], attention_mask = input_c1[1], pixel_values = input_c1[2])
        clip_score_c1 = clipscore(model, output_c1)

    if pd.notna(example["shuffle_trigrams"]):
        input_c2 = get_inputs(image, example["shuffle_trigrams"])
        output_c2 = model(input_ids = input_c2[0], attention_mask = input_c2[1], pixel_values = input_c2[2])
        clip_score_c2 = clipscore(model, output_c2)

    if pd.notna(example["shuffle_words_within_trigrams"]):
        input_c3 = get_inputs(image, example["shuffle_words_within_trigrams"])
        output_c3 = model(input_ids = input_c3[0], attention_mask = input_c3[1], pixel_values = input_c3[2])
        clip_score_c3 = clipscore(model, output_c3)

    if clip_score_c0 == max([clip_score_c0, clip_score_c1, clip_score_c2, clip_score_c3]):
        count += 1

print("Order score:", count/len(df))

# from visual_genome.visual_genome import VG_Order_Test
# from pkgs.openai.clip import load as load_model

# import torch
# import os
# import numpy as np
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# FLICKR_ROOT = '/data0/datasets/FLICKR30K/test/'

# device = 'cuda:6'

# model, image_processor = load_model(name = 'ViT-B/32', pretrained = False, keep_positional = True, rotate = False)

# checkpoint = '/home/diptisahu11/clip/logs/clip400m-shuffle-images-and-captions/checkpoints/epoch_5.pt'

# state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
# if(next(iter(state_dict.items()))[0].startswith("module")):
#     state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

# model.load_state_dict(state_dict, strict=False)
# model = model.to(device)
# model.eval()

# dataset = VG_Order_Test(image_processor.process_image, root_dir=FLICKR_ROOT)
# joint_loader = DataLoader(dataset, batch_size=32)

# @torch.no_grad()
# def get_retrieval_scores_batched(model, joint_loader):
#     scores = []
#     for batch in tqdm(joint_loader):
#         image_options = []
#         for i_option in batch["image_options"]:
#             image_embeddings = model.get_image_features(i_option.to(device)).cpu().numpy() # B x D
#             image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
#             image_options.append(np.expand_dims(image_embeddings, axis=1))

#         caption_options = []
#         for c_option in batch["caption_options"]:
#             caption_tokenized = torch.cat([image_processor.process_text(c)['input_ids'] for c in c_option])
#             caption_embeddings = model.get_text_features(caption_tokenized.to(device)).cpu().numpy() # B x D
#             caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
#             caption_options.append(np.expand_dims(caption_embeddings, axis=1))

#         image_options = np.concatenate(image_options, axis=1) # B x K x D
#         caption_options = np.concatenate(caption_options, axis=1) # B x L x D
#         batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
#         scores.append(batch_scores)

#     all_scores = np.concatenate(scores, axis=0) # N x K x L
#     return all_scores

# scores = get_retrieval_scores_batched(model, joint_loader)

# def macroacc_evaluation(scores, dataset):

#     metrics = {"Accuracy": None}
#     preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
#     correct_mask = (preds == 1)
#     metrics["Accuracy"] = np.mean(correct_mask)

#     all_attributes = np.array(dataset.all_attributes)
#     # Log the accuracy of all relations
#     for attribute in np.unique(all_attributes):
#         attribute_mask = (all_attributes == attribute)
#         if attribute_mask.sum() == 0:
#             continue
#         metrics[f"{attribute}-Acc"] = correct_mask[attribute_mask].mean()

#     return metrics

# metrics = macroacc_evaluation(scores, dataset)

# all_accs = []
# for k,v in metrics.items():
#     if "-Acc" in k:
#         all_accs.append(v)

# print(np.mean(all_accs))