import os
import torch
import torch.nn as nn
import numpy as np
import time
import random
import copy
import pandas as pd
import cv2
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

from models.ct_adapter import CTAdapterPlus
from models.u2net.u2net_segmentation import U2NetSegmentationProcessor


def main():
    device = 'cuda'
    fg_ref_path = '../assets/videostudio-img/fg.png'
    bg_ref_path = '../assets/videostudio-img/bg.png'
    scene_prompt = 'a girl walking in a beach'
    input_fg_scale = 0.50
    input_bg_scale = 0.33
    scene_height = 512
    scene_width = 512
    seed = 100

    pretrained_model_name_or_path='../weights/SG161222/Realistic_Vision_V4.0_noVAE'
    vae_model_path='../weights/stabilityai/sd-vae-ft-mse'
    image_encoder_path='../weights/FireCRT/VideoStudio/videostudio-img-encoder'
    ip_ckpt = '../weights/FireCRT/VideoStudio/videostudio-img-combine.bin'
    u2net_pth = '../weights/FireCRT/VideoStudio/u2net.pth'

    # videostudio-img initialization
    ct_noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                                clip_sample=False, set_alpha_to_one=False, steps_offset=1,)
    ct_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, scheduler=ct_noise_scheduler, 
                                               vae=ct_vae, feature_extractor=None, safety_checker=None)
    ct_model = CTAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    u2net_processor = U2NetSegmentationProcessor(u2net_pth, device=device)

    # load image
    fg_ref_img = Image.open(fg_ref_path)
    bg_ref_img = Image.open(bg_ref_path)
    fg_masked_pil_image, _ = u2net_processor.obtain_fg_bg_pil(np.array(fg_ref_img))
    _, bg_masked_pil_image = u2net_processor.obtain_fg_bg_pil(np.array(bg_ref_img))
    fg_masked_pil_image.resize((256, 256))
    bg_masked_pil_image.resize((256, 256))
    image_list = [fg_masked_pil_image, bg_masked_pil_image]
    combine_image = ct_model.generate(pil_image=image_list, num_samples=1, num_inference_steps=60, 
                                       seed=seed, prompt=scene_prompt, scale=input_fg_scale, scale_bg=input_bg_scale, 
                                       height=scene_height, width=scene_width)[0]
    combine_image.save('../assets/videostudio-img/combine.png')     


if __name__ == '__main__':
    main()




