import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def wan2_1_flash():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/video")

    config.pretrained.model = "Wan2.1-T2V-1.3B-Diffusers" 
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=4.5
    
    config.height = 480
    config.width = 832
    config.frames = 81
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 4 # 实际是2 每张卡上的prompt是一样的
    config.sample.num_batches_per_epoch = 2
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.51
    # kl loss
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=True
    config.train.ema=True
    config.mixed_precision = "bf16"
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 300
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = 'logs'
    config.resume_from = None
    config.reward_fn = {
        "videohpsv3": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.run_name = ""
    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
