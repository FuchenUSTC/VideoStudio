import torch
from transformers import AutoTokenizer, AutoModel

import socket
import json
import heapq
import re
import sys

from .gen_planner import base_prompt as p1, base_history as h1


def parse_multi_scene_output(response):
    lines = response.split('\n')
    out_lines = []
    for i in range(len(lines)):
        if lines[i].startswith(f'[step_{i+1}: ') and lines[i].endswith(']'):
            out_lines.append(lines[i][len(f'[step_{i+1}: '):-1])
        else:
            # print(out_lines)
            return False, None
    return True, out_lines


def check_contain_chinese(check_str):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(check_str)
    return match is not None


class VideoStudioLLM():
    def __init__(self, model_path='../weights/THUDM/chatglm3-6b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cpu')
        self.model = self.model.half().cuda()
        self.model = self.model.eval()
        
        self.top_k = 1
        self.max_try = 5
    
    def __call__(self, input_txt):
        counter = 0
        success_flag = True
        
        while True:
            counter += 1
            print(f'Calling times: {counter}')
            
            if counter > self.max_try:
                success_flag = False
                break
                
            # step1: gen multi-scene prompt simple, extract common objects, extract common backgrounds
            response_planner, _ = self.model.chat(self.tokenizer, p1 + input_txt, history=h1.copy())
            torch.cuda.empty_cache()
            # print(response_planner)
            success, planner = parse_multi_scene_output(response_planner)
            if counter <= 3:
                success = success and (not 'camera' in response_planner)
            success = success and (len(planner)==4)
            if not success:
                continue
            
            multi_scene_prompt = []
            common_objects = []
            common_backgrounds = []
            
            num_scenes = len(planner)
            success = True
            for planner_i in planner:
                planner_i = planner_i[1:-1]
                if not '; entities: ' in planner_i:
                    success = False
                    break
                ss1 = planner_i.split('; entities: ')
                if len(ss1) != 2:
                    success = False
                    break
                multi_scene_prompt.append(ss1[0])
                
                if not '; background: ' in planner_i:
                    success = False
                    break
                ss2 = ss1[1].split('; background: ')
                if len(ss2) != 2:
                    success = False
                    break
                common_objects.append('[' + ss2[0] + ']')
                common_backgrounds.append(ss2[1])
                
            if not success:
                continue
                
            # step4: extract key IP
            object_ips = {}
            for i in range(num_scenes):
                ss = common_objects[i][1:-1].split(', ')
                for si in ss:
                    if not si in object_ips:
                        object_ips[si] = [i]
                    else:
                        object_ips[si].append(i)
            object_names = list(object_ips.keys())
            object_count = []
            for si in object_names:
                object_count.append(len(object_ips[si]))
            
            idx = heapq.nlargest(self.top_k, range(len(object_count)), object_count.__getitem__)
            for i in range(len(object_names)):
                if not i in idx or len(object_ips[object_names[i]]) < 2:
                    del object_ips[object_names[i]]
            # print('object ips:')
            # print(object_ips)
            
            background_ips = {}
            for i in range(num_scenes):
                si = common_backgrounds[i]
                if not si in background_ips:
                    background_ips[si] = [i]
                else:
                    background_ips[si].append(i)
            background_names = list(background_ips.keys())
            
            for i in range(len(background_names)):
                if len(background_ips[background_names[i]]) < 2:
                    del background_ips[background_names[i]]
            # print('background ips:')
            # print(background_ips)
               
            # step6: Generate IP prompts
            ips = {}
            ips.update(object_ips)
            ips.update(background_ips)
            ips_name = list(ips.keys())
            # print('ips prompt:')
            ips_prompt = {}
            success = True
            for name in ips_name:
                # print(name)
                # print(ips[name])
                
                history = []
                query = f'''
Give some aspects that should be considered when describing a photo of {name} in detail.
'''
                response_aspects1, history = self.model.chat(self.tokenizer, query, history=history)
                
                query = f'''
As a professional photographer, give more aspects that should be considered when describing a photo of {name} in detail, e.g., theme, composition, focal length and depth of field, details and texture, technology and post-processing, rendering technology, camera brand and model used, film type and characteristics, location and characteristics of light sources, reference to the master's work, etc.
'''
                response_aspects2, history = self.model.chat(self.tokenizer, query, history=history)
                if name in object_ips:
                    query = response_aspects1 + '\n' + response_aspects2 + '\n' + f'''
Considering the above mentioned aspects, given you a sentence of video: "{input_txt}", give a description (single paragraph without segmentation) for a photo of {name} in this video in detail.

You must follow these instructions:
1. The description provided should be concise and detailed.
2. Prohibition of artistic appreciation and personal emotions.
3. While retaining the author's meaning, clearly supplement all aspects of the professional photography description just mentioned.
4. It is prohibited to include vague descriptions such as "may" and "may".
5. The description is in English.
'''
                else:
                    query = response_aspects1 + '\n' + response_aspects2 + '\n' + f'''
Considering the above mentioned aspects, give a description (single paragraph without segmentation) for a photo of {name} in this video in detail.

You must follow these instructions:
1. The description provided should be concise and detailed.
2. Prohibition of artistic appreciation and personal emotions.
3. While retaining the author's meaning, clearly supplement all aspects of the professional photography description just mentioned.
4. It is prohibited to include vague descriptions such as "may" and "may".
5. The description is in English.
'''
                response, _ = self.model.chat(self.tokenizer, query, history=[])
                # print(response)
                if '\n' in response:
                    success = False
                    break
                    
                # print(response)
                
                ips_prompt[name] = response
                
            if not success:
                continue
            
            # step7: Translator
            for i in range(num_scenes):
                if check_contain_chinese(multi_scene_prompt[i]):
                    query = f'''
The outputs contain only English words.
Translate the following content into English (single paragraph without segmentation):
{multi_scene_prompt[i]}'''

                    response, history = self.model.chat(self.tokenizer, query, history=[])
                    multi_scene_prompt[i] = response
                    
                # print(i)
                # print(multi_scene_prompt[i])            
            
            # print('ips prompt:')
            
            for name in ips_name:
                if check_contain_chinese(ips_prompt[name]):
                    query = f'''
The outputs contain only English words.
Translate the following content into English (single paragraph without segmentation):
{ips_prompt[name]}'''

                    response, history = self.model.chat(self.tokenizer, query, history=[])
                    ips_prompt[name] = response
            break
        
        if success_flag:
            multi_scene_script = {}
            multi_scene_script['scene_prompt'] = []
            multi_scene_script['scene_foreground'] = [-1] * num_scenes
            multi_scene_script['scene_background'] = [-1] * num_scenes
            multi_scene_script['entity_prompt'] = []
            
            for i in range(num_scenes):
                multi_scene_script['scene_prompt'].append(multi_scene_prompt[i].replace('\n', ' '))
                
            for idx, name in enumerate(ips_name):
                multi_scene_script['entity_prompt'].append(ips_prompt[name].replace('\n', ' '))
                
                for i in range(num_scenes):
                    if i in ips[name]:
                        if name in object_ips:
                            if multi_scene_script['scene_foreground'][i] == -1:
                                multi_scene_script['scene_foreground'][i] = idx
                        else:
                            if multi_scene_script['scene_background'][i] == -1:
                                multi_scene_script['scene_background'][i] = idx
            return multi_scene_script
        else:
            return None
        