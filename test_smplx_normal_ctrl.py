import gc
import subprocess
import time
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

from mvdiffusion.pipelines.pipeline_mvdiffusion_controlnet import MVDiffusionImagePipeline
from mvdiffusion.models.controlnet import ControlNetModel

from mvdiffusion.data.thuman_dataset import ObjaverseDataset

import torch

import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from packaging import version
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

# from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset

# from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange
from rembg import remove
import pdb

weight_dtype = torch.float16


@dataclass
class TestConfig:
    controlnet_path: str
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool


def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def log_validation_joint(dataloader, pipeline, cfg: TestConfig, save_dir:str, controlnet):
    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)

    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    images_control = []
    for i, batch in enumerate(dataloader):
        filename = batch['subject_id']
        # (B, Nv, 3, H, W)
        imgs_in, colors_out, normals_out = batch['imgs_in'], batch['imgs_out'], batch['normals_out']
        # use smplx normals as additional control
        normals_ctrl = batch['normals_smplx']

        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([imgs_in, imgs_in], dim=0)
        imgs_out = torch.cat([normals_out, colors_out], dim=0)
        normals_ctrl = torch.cat([normals_ctrl] * 2, dim=0)

        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out,
                                                                                        "B Nv C H W -> (B Nv) C H W")
        normals_ctrl = rearrange(normals_ctrl, "B Nv C H W -> (B Nv) C H W")

        # (B*Nv, Nce)
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)
        images_gt.append(imgs_out)
        images_control.append(normals_ctrl)
        num_views = len(VIEWS)

        sam_predictor = sam_init(pipeline)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_task_embeddings, control_img=normals_ctrl, generator=generator,
                    guidance_scale=guidance_scale,
                    output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images  # BxNv C H W

                bsz = out.shape[0] // 2
                normals_pred = out[:bsz]
                images_pred = out[bsz:]

                # cur_dir = os.path.join(save_dir, "cropsize-cfg1.0")
                cur_dir = "/media/pawting/SN640/hello_worlds/Wonder3D/outputs/cropsize-cfg1.0"

                for i in range(bsz // num_views):
                    scene = filename[i]
                    scene_dir = os.path.join(cur_dir, scene)
                    normal_dir = os.path.join(scene_dir, "normals")
                    masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                    os.makedirs(normal_dir, exist_ok=True)
                    os.makedirs(masked_colors_dir, exist_ok=True)
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i * num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]

                        normal_filename = f"normals_000_{view}.png"
                        rgb_filename = f"rgb_000_{view}.png"
                        normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                        color = save_image(color, os.path.join(scene_dir, rgb_filename))
                        # background removal
                        # rm_normal = remove(normal)
                        # rm_color = remove(color)
                        rm_normal = bg_removal_sam(normal, sam_predictor)
                        rm_color = bg_removal_sam(color, sam_predictor)

                        rm_normal.save(os.path.join(scene_dir, normal_filename))
                        rm_color.save(os.path.join(masked_colors_dir, rgb_filename))
                        # save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                        # save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))

    # same memory
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def bg_removal_sam(img, predictor):
    img = Image.fromarray(img)
    image_rem = img.convert('RGBA')
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:, :, -1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    rm_img = sam_segment(predictor, img.convert('RGB'), x_min, y_min, x_max, y_max)
    return rm_img


def sam_init(pipeline):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=pipeline.device)
    predictor = SamPredictor(sam)
    return predictor


def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2: center - h // 2 + h, center - w // 2: center - w // 2 + w] = image_arr[y: y + h,
                                                                                                   x: x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)


def load_wonder3d_pipeline(cfg):
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_path, torch_dtype=torch.float16)
    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    # speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipeline.to('cuda:0')

    return pipeline


def check_xformers(cfg, pipeline):
    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()
            print("use xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


def process_3d(data_dir, guidance_scale, crop_size):
    dir = None
    global scene

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = f"../{data_dir}/cropsize-{int(crop_size)}-cfg{guidance_scale:.1f}/"
    cmd = (f'cd instant-nsr-pl && python launch.py '
           f'--config configs/neuralangelo-ortho-wmask.yaml '
           f'--gpu 0 '
           f'--train dataset.root_dir={exp_dir} dataset.scene={scene} && cd ..')
    subprocess.run(cmd, shell=True)
    import glob

    obj_files = glob.glob(f'{cur_dir}/instant-nsr-pl/exp/{scene}/*/save/*.obj', recursive=True)
    print(f"Mesh reconstruction done, saved to {obj_files}")
    # if obj_files:
    #     dir = obj_files[0]
    # return dir


def main(
        cfg: TestConfig
):
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    # 1. Load pipeline for inference
    pipeline = load_wonder3d_pipeline(cfg)

    check_xformers(cfg, pipeline=pipeline)
    # 2. Get dataset for testing
    # validation_dataset = MVDiffusionDataset(
    #     **cfg.validation_dataset
    # )

    validation_dataset = ObjaverseDataset(
        **cfg.validation_dataset
    )

    # 3. DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    # 4. Perform inference
    if cfg.pred_type == 'joint':
        log_validation_joint(
            validation_dataloader,
            pipeline,
            cfg,
            weight_dtype,
            cfg.save_dir
        )
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    from utils.misc import load_config

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    # cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    if cfg.num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    elif cfg.num_views == 4:
        VIEWS = ['front', 'right', 'back', 'left']
    main(cfg)
