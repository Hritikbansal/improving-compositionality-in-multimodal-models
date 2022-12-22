import torch

from tqdm import tqdm
from pkgs.openai.clip import load as load_model

device = 'cuda'

## pretrained = True loads the original OpenAI CLIP model trained on 400M image-text pairs
clip_model, clip_processor = load_model(name = 'RN50', pretrained = False)

# Replace with the location of the checkpoint
# The link for checkpoints -- https://drive.google.com/drive/u/0/folders/1K0kPJZ3MA4KAdx3Fpq25dgW59wIf7M-x

checkpoint = '/data0/ckpts/hbansal/winoground/clip500k-without-rotary-with-pe/best.pt'

state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
if(next(iter(state_dict.items()))[0].startswith("module")):
    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

clip_model.load_state_dict(state_dict, strict=False)
clip_model.eval()

print(clip_model)

