import importlib
import os
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2
import numpy as np
import imageio
import torchvision
import PIL
import safetensors.torch

from typing import Optional
from collections import abc
from einops import rearrange
from functools import partial
from typing import List
from huggingface_hub import HfFolder, Repository, create_repo, whoami

import multiprocessing as mp
from threading import Thread
from queue import Queue
from collections import defaultdict


from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import BaseOutput, logging
from spacy.util import filter_spans
from skimage.metrics import structural_similarity

logger = logging.get_logger('video-diffusion') 


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = video.mul_(std).add_(mean)  # unnormalize back to [0,1]
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> (i f) h w c')
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype('uint8') for image in images]  # f h w c
    return images


def renorm(sample, mean, std):
    if len(sample.shape) == 4:
        mean = torch.tensor(mean, device=sample.device).reshape(1, -1, 1, 1)  # nchw
        std = torch.tensor(std, device=sample.device).reshape(1, -1, 1, 1)  # nchw    
    else:
        mean = torch.tensor(mean, device=sample.device).reshape(1, -1, 1, 1, 1)  # ncfhw
        std = torch.tensor(std, device=sample.device).reshape(1, -1, 1, 1, 1)  # ncfhw        
    res = sample.mul_(std).add_(mean) 
    res.clamp_(0, 1)
    res = res.cpu() * 255
    return res


def savetensor2vid(video_tensor, output_video_path, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], fps=8):
    video = tensor2vid(video_tensor, mean, std)
    h, w, c = video[0].shape
    video_stack = np.stack(video, axis=0)
    video_tensor = torch.from_numpy(video_stack)
    torchvision.io.write_video(output_video_path, video_tensor, fps=8)
    return output_video_path


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('./data/fonts/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)
        
    
def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters())
    return params_to_string(params_num)


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, save_to_gif=False, use_cv2=True, fps=8) -> str:
    h, w, c = video_frames[0].shape
    if save_to_gif:
        image_lst = []
        if output_video_path.endswith('mp4'):
            output_video_path = output_video_path[:-3] + 'gif'
        for i in range(len(video_frames)):
            image_lst.append(video_frames[i])
        imageio.mimsave(output_video_path, image_lst, fps=fps)     
        return output_video_path
    else:
        if use_cv2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
            for i in range(len(video_frames)):
                img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
                video_writer.write(img)
            video_writer.release()
        else:
            duration = math.ceil(len(video_frames) / fps)
            append_num = duration * fps - len(video_frames)
            for k in range(append_num): video_frames.append(video_frames[-1])
            video_stack = np.stack(video_frames, axis=0)
            video_tensor = torch.from_numpy(video_stack)
            torchvision.io.write_video(output_video_path, video_tensor, fps=fps, options={"crf": "17"})
        return output_video_path


def export_pil_to_video(pil_video_frames: List[PIL.Image.Image], output_video_path: str = None, fps=8) -> str:
    video_frames = []
    for i in range(len(pil_video_frames)): video_frames.append(np.array(pil_video_frames[i]))
    duration = math.ceil(len(video_frames) / fps)
    append_num = duration * fps - len(video_frames)
    for k in range(append_num): video_frames.append(video_frames[-1])
    video_stack = np.stack(video_frames, axis=0)
    video_tensor = torch.from_numpy(video_stack)
    torchvision.io.write_video(output_video_path, video_tensor, fps=fps, options={"crf": "17"})
    return output_video_path


def load_additional_temporal_weights_v2(unet, temporal_weights_path):
    state_dict = torch.load(temporal_weights_path, map_location="cpu")
    match_param_num, mis_match_param_num = 0, 0
    with torch.no_grad():
        for param_name, param in unet.named_parameters():
            if 'temp' in param_name or 'transformer_in' in param_name:
                if 'transformer_in' in param_name:
                    state_key = param_name.replace('temp_transformer_in','transformer_in')
                elif 'up_blocks.0' in param_name:
                    state_key = param_name.replace('up_blocks.0','up_blocks.1')
                elif 'up_blocks.1' in param_name:
                    state_key = param_name.replace('up_blocks.1','up_blocks.2')
                else:
                    state_key = param_name
                
                if state_key in state_dict and (param.shape == state_dict[state_key].shape): 
                    param.copy_(state_dict[state_key])
                    match_param_num += 1
                    #logger.info('Match the temporal key: {}'.format(param_name))
                else:
                    mis_match_param_num += 1
                    #logger.info('Miss match: {}'.format(param_name))
                #logger.info('The unet temp weights: {}, shape: {}'.format(param_name, param.shape))
        #for param_name in state_dict:
        #    if 'temp' in param_name or 'transformer_in' in param_name:
        #        logger.info('The ckpt weights: {}, shape: {}'.format(param_name, state_dict[param_name].shape))
    logger.info('Load the temporal weights from: {}, matched number: {}, miss matched number: {}'.format(temporal_weights_path, match_param_num, mis_match_param_num))


def load_additional_temporal_weights(unet, temporal_weights_path):
    match_num = 0
    if temporal_weights_path.endswith('safetensors'):
        with open(temporal_weights_path, 'rb') as f:
            data = f.read()
        state_dict = safetensors.torch.load(data)        
    else:
        state_dict = torch.load(temporal_weights_path, map_location="cpu")
    with torch.no_grad():
        for param_name, param in unet.named_parameters():
            if 'temp' in param_name or 'transformer_in' in param_name:
                param.copy_(state_dict[param_name])
                match_num += 1
    logger.info('Load the temporal weights from: {}, matched number: {}'.format(temporal_weights_path, match_num))


def load_additional_spatial_weights(unet, spatial_weights_path):
    match_num = 0
    miss_match_num = 0
    if spatial_weights_path.endswith('safetensors'):
        with open(spatial_weights_path, 'rb') as f:
            data = f.read()
        state_dict = safetensors.torch.load(data)
    else:
        state_dict = torch.load(spatial_weights_path, map_location="cpu")
    with torch.no_grad():
        for param_name, param in unet.named_parameters():
            if ('temp' not in param_name) and ('transformer_in' not in param_name):
                if param_name in state_dict:
                    param.copy_(state_dict[param_name])
                    match_num += 1
                else:
                    miss_match_num += 1
    logger.info('Load the spatial weights from: {}, matched number: {}, miss match (aes): {}'.format(spatial_weights_path, match_num, miss_match_num))


def copy_conv_in_weights(unet):
    with torch.no_grad():
        zero_matrix = torch.zeros_like(unet.conv_in.weight).to(unet.dtype).to(unet.device)
        unet.concate_temp_conv_in.weight[:,:4,:,:].copy_(unet.conv_in.weight)
        unet.concate_temp_conv_in.weight[:,4:,:,:].copy_(zero_matrix)
        unet.concate_temp_conv_in.bias.copy_(unet.conv_in.bias)


def copy_conv_in_mask_weights(unet):
    with torch.no_grad():
        torch.nn.init.zeros_(unet.concate_mask_conv_in.weight)
        #zero_matrix = torch.zeros_like(unet.conv_in.weight).to(unet.dtype).to(unet.device)
        unet.concate_mask_conv_in.weight[:,:4,:,:].copy_(unet.conv_in.weight)
        #unet.concate_mask_conv_in.weight[:,4:,:,:].copy_(zero_matrix)
        unet.concate_mask_conv_in.bias.copy_(unet.conv_in.bias)


def obtain_video_mixed_noise(latents, alpha=1.0):
    if len(latents.shape) == 4:
        return torch.randn_like(latents)
    else:
        alpha = torch.tensor(alpha)
        effi_1 = torch.sqrt(alpha**2 / (1. + alpha**2))
        effi_2 = torch.sqrt(1./(1. + alpha**2))
        N,C,T,H,W = latents.shape
        noise_share = torch.randn((N,C,1,H,W), dtype=latents.dtype).to(latents.device).repeat(1,1,T,1,1)
        assert torch.equal(noise_share[:,:,0,:,:], noise_share[:,:,1,:,:])

        noise_res = torch.randn_like(latents)
        noise = effi_1 * noise_share + effi_2 * noise_res
        noise = noise.type(latents.dtype)
        return noise


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


# compute shot detection
def calculate_ssim(frames_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=frames_tensor.device).reshape(-1, 1, 1, 1)  # C T H W
    std = torch.tensor([0.5, 0.5, 0.5], device=frames_tensor.device).reshape(-1, 1, 1, 1)  # C T H W
    frames = frames_tensor.mul_(std).add_(mean)  # unnormalize back to [0,1]
    frames.clamp_(0, 1)
    
    round_all_ssim_scores = []
    frames_numpy = frames.permute(1,2,3,0).detach().cpu().numpy() * 255 # C T H W -> T H W C
    frames_numpy = frames_numpy.astype('uint8')
    T, h, w, c = frames_numpy.shape
    
    all_ssim_scores = []
    for i in range(T-1):
        ssim_score = structural_similarity(frames_numpy[i], frames_numpy[i+1], channel_axis=2)
        all_ssim_scores.append(ssim_score)
        round_all_ssim_scores.append(round(ssim_score, 3))
    new_all_ssim_scores = abs(np.round(all_ssim_scores - sum(all_ssim_scores)/len(all_ssim_scores), 3)).tolist()
    return round_all_ssim_scores, new_all_ssim_scores


def detect_smooth_video(frames_tensor, sudden_change_thres=0.40):
    score_list, score_diff_list = calculate_ssim(frames_tensor)
    lowest_ssim = min(score_list)
    idx = score_list.index(lowest_ssim)
    if lowest_ssim < sudden_change_thres or lowest_ssim > 0.990: #
        return False, lowest_ssim
    else:
        return True, lowest_ssim


# obtain image CLIP pooler feature
def obtain_image_feature_from_vision_clip(clip_vision_model, processor, images):
    # convert tensor to pil image
    images_np = tensor2vid(images)
    inputs = processor(images=images_np, return_tensors="pt") # N C H W
    inputs['pixel_values'] = inputs['pixel_values'].to(clip_vision_model.device, dtype=clip_vision_model.dtype)
    outputs = clip_vision_model(**inputs)
    last_hidden_state = outputs.last_hidden_state # N S C 
    pooler_output = outputs.pooler_output # N C
    return last_hidden_state, pooler_output


def obtain_image_feature_from_projection_vision_clip(clip_vision_model, processor, images):
    # convert tensor to pil image
    images_np = tensor2vid(images)
    inputs = processor(images=images_np, return_tensors="pt") # N C H W
    inputs['pixel_values'] = inputs['pixel_values'].to(clip_vision_model.device, dtype=clip_vision_model.dtype)
    outputs = clip_vision_model(**inputs)
    img_projection_embeds = outputs.image_embeds # N C 
    return img_projection_embeds


# stage-wise dynamic video noise generation
def obtain_stagewise_dynamic_video_noise(latents, timestep, middle_frame_lambda=0.5, add_noise_range=[500, 800]):
    if len(latents.shape) == 4:
        return torch.randn_like(latents)
    else:
        condition = (timestep > add_noise_range[0]) & (timestep <= add_noise_range[1])
        vid_index = condition.nonzero().reshape(-1).to(latents.device)
        middle_frame_main_lambda = torch.sqrt(torch.tensor(middle_frame_lambda))
        middle_frame_residue_lambda = torch.sqrt(torch.tensor(1-middle_frame_lambda))
        N,C,T,H,W = latents.shape
        video_noise = torch.randn_like(latents) # to be replace by other noise
        noise_main = torch.randn((vid_index.numel(), C, H, W), dtype=latents.dtype).to(latents.device)
        # the noise in middle frame
        video_noise[vid_index,:,T//2,:,:] = noise_main
        # the noise in other frames
        noise_residue = torch.randn((N, C, T, H, W), dtype=latents.dtype).to(latents.device)
        video_noise[vid_index,:,:T//2,:,:] = middle_frame_main_lambda * noise_main.unsqueeze(2).repeat(1,1,T//2,1,1) + middle_frame_residue_lambda * noise_residue[vid_index,:,:T//2,:,:]
        video_noise[vid_index,:,T//2+1:,:,:] = middle_frame_main_lambda * noise_main.unsqueeze(2).repeat(1,1,T-(T//2+1),1,1) + middle_frame_residue_lambda * noise_residue[vid_index,:,T//2+1:,:,:]
        video_noise = video_noise.type(latents.dtype)
        assert torch.equal(video_noise[vid_index,:,T//2,:,:], noise_main)
        return video_noise


# copy conv_in weights when use the warping concate
def copy_warping_conv_in_weights(unet):
    with torch.no_grad():
        zero_matrix = torch.zeros_like(unet.conv_in.weight).to(unet.dtype).to(unet.device)
        unet.warping_conv_in.weight[:,:4,:,:].copy_(unet.conv_in.weight)
        unet.warping_conv_in.weight[:,4:,:,:].copy_(zero_matrix)
        unet.warping_conv_in.bias.copy_(unet.conv_in.bias)
        

# obtain videomae classfication scores 
def obtain_action_prob_from_vidmae(videomae, processor, images):
    # convert tensor to pil image
    N,_,_,_,_ = images.shape
    images_np = tensor2vid(images) # (n,t) c h w
    images_np = np.array(images_np)
    image_list = np.split(images_np, N) # N [t h w c)]
    image_list = [image.transpose(0,3,1,2) for image in image_list]
    inputs_array,logits_array = [], []
    for image in image_list:
        inputs_array.append(processor(images=list(image), return_tensors="pt").to('cuda')) # N C T H W
    with torch.no_grad():
      for inputs in inputs_array:
        outputs = videomae(**inputs)
        logits_array.append(outputs.logits)
    out_logits_array = []
    for logits in logits_array:
        logits = F.softmax(logits, dim=1) # 400
        max_prob = logits.max()
        logits *= (max_prob > 0.5)
        out_logits_array.append(logits)
    out_logits_tensor= torch.stack(out_logits_array).reshape(N, 400)
    return out_logits_tensor


# obtain action phrase list from prompt
def obtain_action_phrase_from_prompt(nlp, matcher, prompt):
    doc = nlp(prompt) 
    # call the matcher to find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    action_phrase_list = filter_spans(spans)
    return action_phrase_list    


def compute_cosine_similarity(text_encoder, tokenizer, action_phrase, action_prototypes):
    text_inputs = tokenizer(
        action_phrase,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(
        text_input_ids.to('cuda'),
        output_hidden_states=True,
    )
    pooled_prompt_embeds = prompt_embeds[0]
    cos_score = F.cosine_similarity(pooled_prompt_embeds, action_prototypes)
    max_prob = cos_score.max()
    max_label = cos_score.argmax(-1).item()
    if max_prob < 0.2: max_prob = 0
    return max_prob, max_label


# obtain action condition
def obtain_action_prob_condition(text_encoder, tokenizer, action_phrase_list, action_prototypes):
    action_prototypes = torch.tensor(action_prototypes).to('cuda')
    action_prob_fea = torch.zeros(1, 400).to('cuda')
    if len(action_phrase_list) == 0: 
        return action_prob_fea
    else:
        #output_list = ""
        action_prob, cos_sum = {}, 0
        for action_phrase in action_phrase_list:
            action_phrase = str(action_phrase)
            max_prob, max_label = compute_cosine_similarity(text_encoder, tokenizer, action_phrase, action_prototypes)
            action_prob[max_label] = max_prob
            cos_sum += max_prob
            #output_list += ' {}: {}'.format(action_phrase, max_prob)
        if cos_sum == 0: cos_sum = 1.0
        for key in action_prob:
            action_prob_fea[0, key] = action_prob[key] / cos_sum * 1.0
    #logger.info('action phrase: {}, action prob fea {}'.format(action_phrase_list, output_list))        
    return action_prob_fea


# sampler in svd
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


# image encoder
def encode_image(image_list, image_encoder, image_processor, feature_extractor, device, weight_dtype):
    image_embeddings_list = []
    for image in image_list:
        image = image_processor.pil_to_numpy(image)
        image = image_processor.numpy_to_pt(image)
        # We normalize the image before resizing to match with the original implementation.
        # Then we unnormalize it after resizing.
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        # Normalize the image with for CLIP input
        image = feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        image = image.to(device, dtype=weight_dtype)
        image_embeddings = image_encoder(image).image_embeds 
        image_embeddings = image_embeddings # L C
        image_embeddings_list.append(image_embeddings)
    image_embedding_tensor = torch.stack(image_embeddings_list, dim=0).to(device=device) # N L C
    return image_embedding_tensor


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


# motion score
def calculate_latent_motion_score(latents):
    if latents.shape[2] == 1: 
        diff = torch.abs(latents-latents)
    else:
        diff=torch.abs(latents[:,:,1:]-latents[:,:,:-1]) # N C T H W
    motion_score = torch.sum(torch.mean(diff, dim=[2,3,4]), dim=1) * 10
    return motion_score


def remove_scheduler_noise(
    scheduler,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    removed = (original_samples - sqrt_one_minus_alpha_prod * noise)/sqrt_alpha_prod
    return removed


# get motion area mask
# frames, list of numpy, 0 - 255
def get_moved_area_mask(frames, move_th=5, th=-1):
    ref_frame = frames[0] 
    # Convert the reference frame to gray
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray
    # Initialize the total accumulated motion mask
    total_mask = np.zeros_like(ref_gray)

    # Iterate through the video frames
    for i in range(1, len(frames)):
        frame = frames[i]
        # Convert the frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute the absolute difference between the reference frame and the current frame
        diff = cv2.absdiff(ref_gray, gray)
        #diff += cv2.absdiff(prev_gray, gray)
        # Apply a threshold to obtain a binary image
        ret, mask = cv2.threshold(diff, move_th, 255, cv2.THRESH_BINARY)
        # Accumulate the mask
        total_mask = cv2.bitwise_or(total_mask, mask)
        # Update the reference frame
        prev_gray = gray

    contours, _ = cv2.findContours(total_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    ref_mask = np.zeros_like(ref_gray)
    ref_mask = cv2.drawContours(ref_mask, contours, -1, (255, 255, 255), -1)
    for cnt in contours:
        cur_rec = cv2.boundingRect(cnt)
        rects.append(cur_rec) 

    #rects = merge_overlapping_rectangles(rects)
    mask = np.zeros_like(ref_gray)
    if th < 0:
        h, w = mask.shape
        th = int(h*w*0.005)
    for rect in rects:
        x, y, w, h = rect
        if w*h < th:
            continue
        #ref_frame = cv2.rectangle(ref_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask[y:y+h, x:x+w] = 255
    return mask

