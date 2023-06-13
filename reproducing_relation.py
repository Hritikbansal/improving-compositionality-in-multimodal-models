from visual_genome.visual_genome import VG_Relation_Test
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

dataset = VG_Relation_Test(image_processor.process_image, root_dir=VG_IMAGE_DIR)
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

drop_relations = ['adjusting',
 'attached to',
 'between',
 'bigger than',
 'biting',
 'boarding',
 'brushing',
 'chewing',
 'cleaning',
 'climbing',
 'close to',
 'coming from',
 'coming out of',
 'contain',
 'crossing',
 'dragging',
 'draped over',
 'drinking',
 'drinking from',
 'driving',
 'driving down',
 'driving on',
 'eating from',
 'eating in',
 'enclosing',
 'exiting',
 'facing',
 'filled with',
 'floating in',
 'floating on',
 'flying',
 'flying above',
 'flying in',
 'flying over',
 'flying through',
 'full of',
 'going down',
 'going into',
 'going through',
 'grazing in',
 'growing in',
 'growing on',
 'guiding',
 'hanging from',
 'hanging in',
 'hanging off',
 'hanging over',
 'higher than',
 'holding onto',
 'hugging',
 'in between',
 'jumping off',
 'jumping on',
 'jumping over',
 'kept in',
 'larger than',
 'leading',
 'leaning over',
 'leaving',
 'licking',
 'longer than',
 'looking in',
 'looking into',
 'looking out',
 'looking over',
 'looking through',
 'lying next to',
 'lying on top of',
 'making',
 'mixed with',
 'mounted on',
 'moving',
 'on the back of',
 'on the edge of',
 'on the front of',
 'on the other side of',
 'opening',
 'painted on',
 'parked at',
 'parked beside',
 'parked by',
 'parked in',
 'parked in front of',
 'parked near',
 'parked next to',
 'perched on',
 'petting',
 'piled on',
 'playing',
 'playing in',
 'playing on',
 'playing with',
 'pouring',
 'reaching for',
 'reading',
 'reflected on',
 'riding on',
 'running in',
 'running on',
 'running through',
 'seen through',
 'sitting behind',
 'sitting beside',
 'sitting by',
 'sitting in front of',
 'sitting near',
 'sitting next to',
 'sitting under',
 'skiing down',
 'skiing on',
 'sleeping in',
 'sleeping on',
 'smiling at',
 'sniffing',
 'splashing',
 'sprinkled on',
 'stacked on',
 'standing against',
 'standing around',
 'standing behind',
 'standing beside',
 'standing in front of',
 'standing near',
 'standing next to',
 'staring at',
 'stuck in',
 'surrounding',
 'swimming in',
 'swinging',
 'talking to',
 'topped with',
 'touching',
 'traveling down',
 'traveling on',
 'tying',
 'typing on',
 'underneath',
 'wading in',
 'waiting for',
 'walking across',
 'walking by',
 'walking down',
 'walking next to',
 'walking through',
 'working in',
 'working on',
 'worn on',
 'wrapped around',
 'wrapped in',
 "by",
 "of",
 "near", "next to",
 "with",
 "beside",
 "on the side of",
 "around"]

def macroacc_evaluation(scores, dataset, drop_relations=drop_relations):

    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)

    all_relations = np.array(dataset.all_relations)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        if relation in drop_relations:
            continue
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()

    return metrics

metrics = macroacc_evaluation(scores, dataset)

all_accs = []
for k,v in metrics.items():
    if "-Acc" in k:
        all_accs.append(v)

print(np.mean(all_accs))

