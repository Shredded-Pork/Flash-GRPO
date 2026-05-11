<div align="center" style="font-family: charter;">

<h1>🦕 Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization</h1>


<a href="" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Flash-GRPO-red?logo=arxiv" height="20" /></a>
<a href="https://shredded-pork.github.io/Flash-GRPO.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/💻_Project-Flash--GRPO-blue.svg" height="20" /></a>
</div>

**Flash-GRPO**, a single-step training framework that outperforms full trajectory training in alignment quality under low computational budgets while substantially improving training efficiency.

<div style="text-align: center;">
    <img src="asset/teaser.png" alt="LOGO">
</div>

<div style="text-align: center;">
    <img src="asset/method.png" alt="LOGO">
</div>

## 🗺️ Roadmap for Flash-GRPO
> Flash-GRPO, a single-step training framework that outperforms full trajectory trainingin alignment quality under low computational budgets while substantially improving training efficiency. Flash-GRPO addresses two critical challenges: iso-temporal grouping eliminates timestep-confounded variance by enforcing prompt-wise temporal consistency, decoupling policy performance
from timestep difficulty; temporal gradient rectification neutralizes the time-dependent scaling factor that causes vastly inconsistent gradient magnitudes across timesteps. Experiments on 1.3B to 14B parameter models validate Flash-GRPO’s effectiveness, demonstrating substantial training acceleration with consistent stability and state-of-the-art alignment qualit
> 
> Welcome Ideas and Contributions. Stay tuned!

## 📕 Training & Evaluation
### Preparation
1. First you need to download the reward model (we support clip-based pickscore, vlm-based hpsv3, ...) and base model (SD3.5-M, FLUX.1-dev).
2. Then you need to modify the noise level in [sd3_pipeline_with_logprob_perstep](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob_perstep.py) and [sd3_pipeline_with_logprob](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py).
3. Finally, you need to modify the [config](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/config/dgx.py). We suggest you using 24 groups and 48 num groups.

**Note that we use branch=4, per branch exploration=6. You can modify them in our code. We will release a neat code verision in next few days.**

### Training
#### About Group Strategy
1. Seed Group:
<img width="1011" height="45" alt="image" src="https://github.com/user-attachments/assets/e865d9db-3f8f-45a7-a497-2298b0013a42" />
<img width="894" height="41" alt="image" src="https://github.com/user-attachments/assets/9953b278-3c91-4b27-8ac5-f67c0199b5f4" />
<img width="892" height="119" alt="image" src="https://github.com/user-attachments/assets/06771eb4-6974-4ddf-a02e-2ec29b3382c4" />

2. Prompt Group: notes the seed group

3. Batch Group: global_std=True
   
#### SD3.5-M
```bash
# Flow-GRPO
bash scripts/multi_node/main.sh
# TempFlow-GRPO
bash scripts/multi_node/train_sd3_pr.sh
```
#### FLUX.1-dev
```bash
# Flow-GRPO
bash scripts/multi_node/train_flux.sh
# TempFlow-GRPO
bash scripts/multi_node/train_flux_pr.sh
```
#### QwenImage
```bash
# Flow-GRPO
bash scripts/multi_node/train_qwenimage.sh
# TempFlow-GRPO
bash scripts/multi_node/train_qwenimage_pr.sh
```


## 📊 Experimental Performance
<img src="asset/performance.png" alt="Performance" width="800"/>

## 📺 Visualization
<img src="asset/appendix_vis4.png" alt="FLUX.1-dev" width="1024"/> 

- For more details please read our paper.

# Acknowledgements
[Flow-GRPO](https://github.com/yifan123/flow_grpo): The first method integrating online reinforcement learning (RL) into flow matching models.
