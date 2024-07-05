import os
import cv2
import argparse
import random
import numpy as np
import torch
import spacy   

from spacy.matcher import Matcher
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModel, AutoProcessor
from torchvision import transforms

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

from models.unet_3d_condition_xl import UNet3DConditionXLModel
from pipelines.pipeline_stable_diffusion_xl import PseudoVideoStableDiffusionXLPipeline, tensor2vid
from util.util import export_to_video, obtain_image_feature_from_vision_clip, obtain_action_phrase_from_prompt, obtain_action_prob_condition



def load_action_prototype(file_path):
    with open(file_path, 'rb') as fr:
        action_prototypes = np.load(fr)
    return action_prototypes   


def load_first_frame(image_path, image_transform):
    image_data = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image_data) # C H W
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(2)
    return image_tensor


def main():

    video_height = 320
    video_width = 512
    num_inference_steps=70
    guidance_scale=12.0
    num_frames = 16

    image_path = '../assets/videostudio-vid/rider.png'
    video_outpath = '../assets/videostudio-vid/rider.mp4'
    input_prompt = 'The motorcyclist riding on the road.'

    device = 'cuda'
    default_pretrained_model_path = '../weights/FireCRT/VideoStudio/videostudio-vid'
    clip_path = '../weights/laion/CLIP-ViT-H-14-laion2B-s32B-b79K'

    # videostudio-vid initialization
    text_encoder = CLIPTextModel.from_pretrained(default_pretrained_model_path, subfolder="text_encoder", revision=None, torch_dtype=torch.float16, variant="fp16")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(default_pretrained_model_path, subfolder="text_encoder_2", revision=None, torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained(default_pretrained_model_path, subfolder="vae", revision=None, torch_dtype=torch.float16, variant="fp16")
    tokenizer = CLIPTokenizer.from_pretrained(default_pretrained_model_path, subfolder="tokenizer", revision=None)
    tokenizer_2 = CLIPTokenizer.from_pretrained(default_pretrained_model_path, subfolder="tokenizer_2", revision=None)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(default_pretrained_model_path, subfolder="scheduler", revision=None)
    unet = UNet3DConditionXLModel.from_pretrained(default_pretrained_model_path, subfolder="unet", revision=None, ignore_mismatched_sizes=False, torch_dtype=torch.float16)

    img_clip = CLIPVisionModel.from_pretrained(clip_path)
    img_processor = AutoProcessor.from_pretrained(clip_path)

    text_encoder.to(device)
    text_encoder_2.to(device)
    vae.to(device)
    img_clip.to(device)
    pipe = PseudoVideoStableDiffusionXLPipeline(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler)
    pipe.to(device)
    print('Have prepared the videostudio-vid pipeline.')

    action_prototypes = load_action_prototype('./data/action_prototypes.npy')
    nlp_analyser = spacy.load('en_core_web_sm') 
    pattern = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    matcher = Matcher(nlp_analyser.vocab)
    matcher.add("Verb phrase", [pattern])
    print('Have prepared the action prototype.')

    latent_frame_transform = transforms.Compose([transforms.Resize(size=(video_height, video_width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.5], std=[0.5]),])
    image_transform = transforms.Compose([transforms.Resize(size=video_width, antialias=True),
                                               transforms.CenterCrop(size=(video_width, video_width)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.5], std=[0.5]),])

    
    # video generation
    latent_image = load_first_frame(image_path, latent_frame_transform)
    mid_frame = load_first_frame(image_path, image_transform)
    midf_clip_token_fea, midf_clip_pool_fea = obtain_image_feature_from_vision_clip(img_clip, img_processor, mid_frame)
    midf_clip_pool_fea.to(device, torch.float16)
    midf_clip_token_fea.to(device, torch.float16)

    action_phrase_list = obtain_action_phrase_from_prompt(nlp_analyser, matcher, input_prompt)
    action_prob_fea = obtain_action_prob_condition(text_encoder_2, tokenizer_2, action_phrase_list, action_prototypes)
    action_prob_fea = action_prob_fea.to(device, pipe.unet.dtype)
    print('Sample prompts: {}, action phrase: {}'.format(input_prompt, action_phrase_list))    

    generator = torch.Generator(device=device).manual_seed(100)
    video_frames = pipe(prompt=input_prompt,
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale, 
                        height=video_height, width=video_width, num_frames=num_frames, 
                        video_latent=True, 
                        midf_clip_pool_fea_guidance=True,
                        midf_clip_pool_fea=midf_clip_pool_fea,
                        midf_clip_token_cross_attn_guidance=True,
                        midf_clip_token_fea=midf_clip_token_fea,
                        aes_score_value=6.0,
                        negative_aes_score_value=6.0,
                        generator=generator,
                        as_latent_frame=latent_image,
                        concate_in_middle=False,
                        action_prob_fea=action_prob_fea)[0]
    outpath = export_to_video(video_frames, video_outpath, save_to_gif=False, use_cv2=False, fps=8)
    print('Output videos: {}'.format(outpath))


if __name__ == '__main__':
    main()
    

