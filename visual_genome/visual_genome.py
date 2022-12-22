import json, os
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from easydict import EasyDict as edict

VG_IMAGE_DIR = '/data0/datasets/VisualGenome/images/'

class VG_Relation_Test(Dataset):
    def __init__(self, image_preprocess, root_dir=VG_IMAGE_DIR):
        '''
        '''
        with open("./visual_genome/visual_genome_relation.json", "r") as f:
            self.dataset = json.load(f)
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(root_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
        self.image_preprocess = image_preprocess
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):    
        rel = self.dataset[index]
        image = Image.open(rel["image_path"]).convert('RGB')    
        # Get the bounding box that contains the relation
        image = image.crop((rel["bbox_x"], rel["bbox_y"], rel["bbox_x"]+rel["bbox_w"], rel["bbox_y"]+rel["bbox_h"]))
        
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)  
            
        caption = rel["true_caption"]
        reversed_caption = rel["false_caption"]   
        item = edict({"image_options": [image], "caption_options": [reversed_caption, caption], "relation": rel["relation_name"]})
        return item


class VG_Attribution_Test(Dataset):
    def __init__(self, image_preprocess, root_dir=VG_IMAGE_DIR):
        '''
        '''
        with open("visual_genome_attribution.json", "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item["image_path"] = os.path.join(root_dir, item["image_path"])
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scene = self.dataset[index]
        image = Image.open(scene["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation
        image = image.crop((scene["bbox_x"], scene["bbox_y"], scene["bbox_x"] + scene["bbox_w"], scene["bbox_y"] + scene["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        true_caption = scene["true_caption"]
        false_caption = scene["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption],
                      "relation": "Attribution"})
        return item


def visual_genome_relation_evaluation(scores, dataset: VG_Relation_Test):
    """
    Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
    """  

    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)
    
    all_relations = np.array(dataset.all_relations)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Count"] = relation_mask.sum()
        metrics[f"{relation}-Accuracy"] = correct_mask[relation_mask].mean()
    
    return metrics
