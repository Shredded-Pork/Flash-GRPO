import torch
from hpsv3 import HPSv3RewardInferencer

def load_hpsv3score():
    inferencer = HPSv3RewardInferencer(device='cuda', checkpoint_path='HPSv3/HPSv3.safetensors')
    torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_hpsv3score(images, prompts):
        rewards = inferencer.reward(images, prompts)

        return [reward[0].item() for reward in rewards]

    return compute_hpsv3score