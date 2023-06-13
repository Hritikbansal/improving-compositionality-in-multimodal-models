import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from pkgs.openai.clip import load as load_model

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")

parser.add_argument("--rotate", default = False, action = "store_true", help = "whether to rotate embeddings or not")

parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

options = parser.parse_args()

device = 'cuda:4'

token = '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'
winoground = load_dataset("facebook/winoground", use_auth_token=token)["test"]

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

#vanilla_winoground = [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39,
#43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 79, 80, 85, 87, 89, 90, 91,
#92, 94, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123,
#124, 125, 126, 127, 129, 137, 139, 140, 141, 142, 145, 146, 147, 151, 153, 154, 157, 158,
#160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184,
#185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 225, 231, 236, 240, 243, 244,
#248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 278, 279, 283, 285,
#288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 328, 337, 341, 349, 357, 360,
#366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397]

winoground_clip_scores = []
for example in tqdm(winoground):
    # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
    # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
    # example = winoground[index]
    
    input_c0_i0 = get_inputs(example["image_0"], example["caption_0"])
    input_c1_i0 = get_inputs(example["image_0"], example["caption_1"])
    input_c0_i1 = get_inputs(example["image_1"], example["caption_0"])
    input_c1_i1 = get_inputs(example["image_1"], example["caption_1"])

    output_c0_i0 = model(input_ids = input_c0_i0[0], attention_mask = input_c0_i0[1], pixel_values = input_c0_i0[2])
    output_c1_i0 = model(input_ids = input_c1_i0[0], attention_mask = input_c1_i0[1], pixel_values = input_c1_i0[2])
    output_c0_i1 = model(input_ids = input_c0_i1[0], attention_mask = input_c0_i1[1], pixel_values = input_c0_i1[2])
    output_c1_i1 = model(input_ids = input_c1_i1[0], attention_mask = input_c1_i1[1], pixel_values = input_c1_i1[2])

    clip_score_c0_i0 = clipscore(model, output_c0_i0)
    clip_score_c1_i0 = clipscore(model, output_c1_i0)
    clip_score_c0_i1 = clipscore(model, output_c0_i1)
    clip_score_c1_i1 = clipscore(model, output_c1_i1)

    winoground_clip_scores.append({"id" : example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1})

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

text_correct_count = 0
image_correct_count = 0
group_correct_count = 0
for result in winoground_clip_scores:
    text_correct_count += 1 if text_correct(result) else 0
    image_correct_count += 1 if image_correct(result) else 0
    group_correct_count += 1 if group_correct(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)
print("image score:", image_correct_count/denominator)
print("group score:", group_correct_count/denominator)
