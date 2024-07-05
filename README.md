# [ECCV 2024] VideoStudio: Generating Consistent-Content and Multi-Scene Videos
<a target="_blank" href="https://VidStudio.github.io">
<img src='https://img.shields.io/badge/Project-Page-green' alt="Project page"/>
</a>
<a target="_blank" href="https://huggingface.co/FireCRT/VideoStudio/tree/main">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Open in HugginFace"/>
</a>


VideoStudio: Novel framework by leveraging LLM for consistent and multi-scene video generation. 

## Update
- [ ] Release the training code of VideoStuio-Img and VideoStudio-Vid.
- [ ] Release the complete inference pipeline for long video generation.
- [x] `2024/07/05` Release inference code of independent componets in VideoStudio: LLM instructions for video script generation, the code and weights of VideoStudio-Img and VideoStudio-Vid.

## Setup

### Prepare Environment
Please install the python packages listed in the [requirements.txt](./requirements.txt)

### Download Checkpoints
Please download the huggingface models listed in [weights/put-the-huggingface-models-in-this-folder](./weights/put-the-huggingface-models-in-this-folder) and put them in to the `weights` folder.

The folder organization is
```
└── weights
    ├── FireCRT/VideoStudio
    │   ├── videostudio-img-encoder
    │   ├── videostudio-vid
    │   ├── videostudio-img-combine.bin
    │   ├── ...    
    ├── SG161222/Realistic_Vision_V4.0_noVAE
    │   ├── ...
    ├── stabilityai/sd-vae-ft-mse
    │   ├── ...
    ├── laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    │   ├── ...
    ├── THUDM/chatglm3-6b
    │   ├── ...
    └────
```

## Inference

### VideoStudio-Img
```
cd videostudio_img
bash infer_videostudio_img.sh
```

Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Foreground Image</p></td>
  <td><p style="text-align: center">Background Image</p></td>
  <td><p style="text-align: center">Combined Image</p></td>
<tr>
<tr>
  <td>
  <img src='./assets/videostudio-img/fg.png' width=256>
 </td>
  <td>
  <img src='./assets/videostudio-img/bg.png' width=256>
 </td>
  <td>
  <img src='./assets/videostudio-img/combine.png' width=256>
 </td>
<tr>
</table>



### VideoStudio-Vid
```
cd videostudio_vid
bash infer_videostudio_vid.sh
```

Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Input Image</p></td>
  <td><p style="text-align: center">Output Video</p></td>
<tr>
<tr>
  <td>
  <img src='./assets/videostudio-vid/rider.png' width=400>
 </td>
  <td>
  <img src='./assets/videostudio-vid/rider.gif' width=400>
 </td>
<tr>
</table>


### Video Script Generation
```
cd script_generation
bash script_generation.sh
```


## License
Please check [Apache-2.0 license](./LICENSE) for details.

## Acknowledgements
The code is built upon [IPAdapter](https://github.com/tencent-ailab/IP-Adapter), [U-2-Net](https://github.com/xuebinqin/U-2-Net) and video generaion pipeline in [Diffusers](https://github.com/huggingface/diffusers) .

## Citation

If you use these models in your research, please cite:

    @inproceedings{Long:ECCV24,
      title={VideoStudio: Generating Consistent-Content and Multi-Scene Videos},
      author={Fuchen Long, Zhaofan Qiu, Ting Yao and Tao Mei},
      booktitle={ECCV},
      year={2024}
    }



