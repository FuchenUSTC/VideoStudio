import torch
import torch.nn as nn
import numpy as np
import time
import random
import copy
import pandas as pd
import cv2
import tensorflow as tf
from torchvision import transforms

from models.videostudio_llm import VideoStudioLLM


def main():
    input_prompt = "A beautiful girl is walking on the beach"
    llm_model_path = '../weights/THUDM/chatglm3-6b'
    llm_planer = VideoStudioLLM(model_path=llm_model_path)
    print('Have loded LLM for video planer')    

    with torch.no_grad():        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            s_time = time.time()
            llm_output = llm_planer(input_prompt)
            e_time = time.time()
            duration = e_time - s_time
    print('LLM duration: {} seconds'.format(duration))     

    ref_prompt_list = llm_output['entity_prompt']
    scene_prompt_list = llm_output['scene_prompt']
    foreground_index = llm_output['scene_foreground']
    background_index = llm_output['scene_background']

    print('Reference entity image prompt list: {}'.format(ref_prompt_list))
    print('Scene prompt list: {}'.format(scene_prompt_list))
    print('Foreground entity index: {}'.format(foreground_index))
    print('Background entity index: {}'.format(background_index))


if __name__ == '__main__':
    main()


