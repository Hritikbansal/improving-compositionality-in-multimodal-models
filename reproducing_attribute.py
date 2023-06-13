from visual_genome.visual_genome import VG_Attribution_Test
from pkgs.openai.clip import load as load_model

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

VG_IMAGE_DIR = '/data0/datasets/VisualGenome/images/'

device = 'cuda:4'

model, image_processor = load_model(name = 'ViT-B/32', pretrained = False, keep_positional = True, rotate = False)

checkpoint = '/data0/ckpts/diptisahu/finetuned/clip500k/shuffle-images-and-captions/checkpoints/epoch_5.pt'

state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
if(next(iter(state_dict.items()))[0].startswith("module")):
    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

dataset = VG_Attribution_Test(image_processor.process_image, root_dir=VG_IMAGE_DIR)
joint_loader = DataLoader(dataset, batch_size=32)

@torch.no_grad()
def get_retrieval_scores_batched(model, joint_loader):
    scores = []
    for batch in tqdm(joint_loader):
        image_options = []
        for i_option in batch["image_options"]:
            image_embeddings = model.get_image_features(i_option.to(device)).cpu().numpy() # B x D
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
            image_options.append(np.expand_dims(image_embeddings, axis=1))

        caption_options = []
        for c_option in batch["caption_options"]:
            caption_tokenized = torch.cat([image_processor.process_text(c)['input_ids'] for c in c_option])
            caption_embeddings = model.get_text_features(caption_tokenized.to(device)).cpu().numpy() # B x D
            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
            caption_options.append(np.expand_dims(caption_embeddings, axis=1))

        image_options = np.concatenate(image_options, axis=1) # B x K x D
        caption_options = np.concatenate(caption_options, axis=1) # B x L x D
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
        scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0) # N x K x L
    return all_scores

scores = get_retrieval_scores_batched(model, joint_loader)

def macroacc_evaluation(scores, dataset):

    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_attributes = np.array(dataset.all_attributes)
    # Log the accuracy of all relations
    for attribute in np.unique(all_attributes):
        attribute_mask = (all_attributes == attribute)
        if attribute_mask.sum() == 0:
            continue
        metrics[f"{attribute}-Acc"] = correct_mask[attribute_mask].mean()

    return metrics

metrics = macroacc_evaluation(scores, dataset)

all_accs = []
for k,v in metrics.items():
    if "-Acc" in k:
        all_accs.append(v)

print(np.mean(all_accs))

