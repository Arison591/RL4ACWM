import os, random, math
import sys
import gc
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Any, Dict, List
import argparse

from datetime import datetime, timedelta
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging
import cv2



# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func
from utils.get_ray_maps import get_ray_maps


def load_config(config_file):
    cd = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**cd)
    return args

def prepare_model(args, dtype=torch.bfloat16, device="cuda:0"):

    ### Load Tokenizer
    tokenizer_class = import_custom_class(
        args.tokenizer_class, getattr(args, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        args.textenc_class, getattr(args, "textenc_class_path", "transformers")
    )
    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        args.pretrained_model_name_or_path if not hasattr(args, "tokenizer_pretrained_model_name_or_path") else args.tokenizer_pretrained_model_name_or_path,
        load_weights=args.load_weights
    )
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype).eval()

    ### Load VAE
    vae_class = import_custom_class(
        args.vae_class, getattr(args, "vae_class_path", "transformers")
    )
    if getattr(args, 'vae_path', False):
        vae = load_vae_models(vae_class, args.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
    if isinstance(vae.latents_mean, List):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, List):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if vae is not None:
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

    ### Load Diffusion Model
    diffusion_model_class = import_custom_class(
        args.diffusion_model_class, getattr(args, "diffusion_model_class_path", "transformers")
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=args.diffusion_model['model_path'],
        load_weights=args.load_weights and getattr(args, "load_diffusion_model_weights", True),
        **args.diffusion_model['config']
    ).to(device, dtype=dtype)
    total_params = count_model_parameters(diffusion_model)
    print(f'Total parameters for transfomer model:{total_params}')


    ### Load Diffuser Scheduler
    diffusion_scheduler_class = import_custom_class(
        args.diffusion_scheduler_class, getattr(args, "diffusion_scheduler_class_path", "diffusers")
    )

    if hasattr(diffusion_scheduler_class, "from_pretrained") and os.path.exists(os.path.join(args.pretrained_model_name_or_path, "scheduler")):
        scheduler = diffusion_scheduler_class.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'scheduler'))
    else:
        if hasattr(args, "diffusion_scheduler_args"):
            scheduler = diffusion_scheduler_class(**args.diffusion_scheduler_args)
        else:
            scheduler = diffusion_scheduler_class()



    # scheduler.config.final_sigmas_type = "sigma_min"

    ### Import Inference Pipeline Class
    pipeline_class = import_custom_class(
        args.pipeline_class, getattr(args, "pipeline_class_path", "diffusers")
    )

    pipe = pipeline_class(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=diffusion_model
    )

    return tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe


def load_images(args, image_root, valid_cams, size=(256,192)):
    n_mem = args.data["train"]["n_previous"]
    mv_images = []
    ori_sizes = []
    for cam in valid_cams:
        images = []
        for i in range(n_mem):
            img = cv2.imread(os.path.join(image_root, cam, str(i)+".png"))[:,:,::-1]
            ori_sizes.append(img.shape, )
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            img = torch.from_numpy(np.transpose(img, (2,0,1)))
            images.append(img)
        ### c,t,h,w
        images = torch.stack(images, dim=1)
        mv_images.append(images)
    ### v,c,t,h,w
    mv_images = torch.stack(mv_images, dim=0)
    return mv_images, ori_sizes


def load_cam_infos(extrinsic_root, intrinsic_root, valid_cams, orisize=None, size=(192,256)):
    extrinsics = []
    intrinsics = []
    for cam in valid_cams:
        extrinsics.append(np.load(os.path.join(extrinsic_root, f"extrinsic_{cam}.npy")))
        intrinsics.append(np.load(os.path.join(intrinsic_root, f"intrinsic_{cam}.npy")))
    ### v,t,4,4
    extrinsics = np.stack(extrinsics, axis=0)
    ### v,3,3
    intrinsics = np.stack(intrinsics, axis=0)

    intrinsics[:,0,0] = intrinsics[:,0,0] * size[1] / orisize[0][1]
    intrinsics[:,0,2] = intrinsics[:,0,2] * size[1] / orisize[0][1]
    intrinsics[:,1,1] = intrinsics[:,1,1] * size[0] / orisize[0][0]
    intrinsics[:,1,2] = intrinsics[:,1,2] * size[0] / orisize[0][0]

    return extrinsics, intrinsics



def infer(
    config_file, image_root, extrinsic_root, intrinsic_root, action_path, prompt, save_path,
    seed=None, device="cuda", default_fps=30, split_views=False, denoise_interval=0,
    save_trajectory=False, trajectory_path=None,
):

    args = load_config(config_file)

    if "action_chunk" in args.data["train"]:
        args.data['train']['chunk']
        video_fps = default_fps // (args.data['train']['action_chunk'] // args.data['train']['chunk'])
    else:
        video_fps = default_fps

    tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe = prepare_model(args, device=device)

    valid_cams = [_+"_color" for _ in args.data["train"]["valid_cam"]]

    obs, ori_sizes = load_images(args, image_root, valid_cams, size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0]))

    v,c,t,h,w = obs.shape

    SPATIAL_DOWN_RATIO = vae.spatial_compression_ratio
    TEMPORAL_DOWN_RATIO = vae.temporal_compression_ratio

    ### extrinsics: v,t,4,4
    ### intrinsics: v,3,3
    extrinsics, intrinsics = load_cam_infos(extrinsic_root, intrinsic_root, args.data["train"]["valid_cam"], orisize=ori_sizes, size=(args.data["train"]["sample_size"]))
    ### actions   : t,c
    actions = np.load(action_path)

    extrinsics = torch.FloatTensor(extrinsics)
    intrinsics = torch.FloatTensor(intrinsics)
    actions = torch.FloatTensor(actions)

    os.makedirs(save_path, exist_ok=True)

    
    trajs = get_traj_maps(
        actions, torch.linalg.inv(extrinsics), extrinsics, intrinsics, args.data["train"]["sample_size"], radius_gen_func=simple_radius_gen_func
    ) # trajs: c,v,t,h,w

    trajs = trajs * 2 - 1
    ori_trajs = trajs.clone()

    # save_video(
    #     rearrange(trajs, 'c v t h w -> c t h (v w)', v=v),
    #     os.path.join(save_path, "trajs.mp4"),
    #     fps=video_fps
    # )

    rays_o, rays_d = get_ray_maps(
        intrinsics.unsqueeze(dim=1).repeat(1,extrinsics.shape[1],1,1).reshape(-1,3,3), extrinsics.reshape(-1,4,4), args.data["train"]["sample_size"][0], args.data["train"]["sample_size"][1]
    )
    rays = torch.cat((rays_o, rays_d), dim=-1).reshape(trajs.shape[1], trajs.shape[2], rays_o.shape[1], rays_o.shape[2], -1)
    rays = rays.permute(4,0,1,2,3) # rays: c,v,t,h,w

    # c,v,t,h,w
    cond_to_concat = torch.cat((trajs, rays), dim=0)


    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


    nall = trajs.shape[2]
    nchunk = int(np.ceil((nall-args.data['train']['n_previous'])/args.data['train']['chunk']))
    
    videos = obs.clone()
    mem_idxes = list(range(args.data['train']['n_previous']))

    ### init conditions
    ichunk_cond_to_concat = torch.cat((
        cond_to_concat[:,:,:args.data['train']['n_previous']],
        cond_to_concat[:,:,args.data['train']['n_previous']:args.data['train']['n_previous']+args.data['train']['chunk']]
    ), dim=2)

    trajs = ichunk_cond_to_concat[:3].clone()

    # seed=None 时不固定随机种子（沿用当前全局 RNG 行为）；seed 给定则复现逐 chunk 采样
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # 去噪中间检查点：每 denoise_interval 步用 callback 捕获当前 latents（CPU 小张量），
    # 全部 chunk 跑完后统一解码成帧，写入 {save_path}/denoise_<step>/{cam}.mp4。
    # 延迟到末尾解码（而非每 chunk 立即 decode），避开去噪循环期间的显存峰值。
    cam_names = args.data["train"]["valid_cam"]
    denoise_steps = set()
    if denoise_interval > 0:
        denoise_steps = set(range(denoise_interval, args.num_inference_step + 1, denoise_interval))
        denoise_steps.add(1)
        # 最终去噪步始终保存，即使 num_inference_step 不是 interval 的整数倍。
        denoise_steps.add(args.num_inference_step)
    checkpoint_latents = {}  # (chunk_idx, step) -> latents (cpu clone)
    captured = {}  # step -> latents (cpu clone)，仅当前 chunk 有效
    trajectory_chunks = []
    current_trajectory = None

    # checkpoint 解码发生在 pipe.infer() 外部；pipe.infer() 自身虽然有
    # @torch.no_grad()，但这里不会自动继承该上下文。必须把整个 VAE
    # forward 包在 inference_mode 中，否则 VAE 会为每个中间视频构建
    # 完整 autograd graph，导致解码峰值显存远高于正常生成路径。
    def _decode_latents(lat):
        with torch.inference_mode():
            lat = lat.to(device=device, dtype=vae.dtype)
            lat_mean = torch.tensor(
                vae.config.latents_mean,
                device=lat.device,
                dtype=lat.dtype,
            ).view(1, vae.z_dim, 1, 1, 1)
            lat_std = torch.tensor(
                vae.config.latents_std,
                device=lat.device,
                dtype=lat.dtype,
            ).view(1, vae.z_dim, 1, 1, 1)
            return vae.decode(
                lat * lat_std / scheduler.config.sigma_data + lat_mean,
                return_dict=False,
            )[0]

    def _cb(pipe_, i, t, ck):
        step = i + 1
        if current_trajectory is not None:
            row = {
                "step": step,
                "timestep": float(t.item()) if torch.is_tensor(t) else float(t),
                "latents": ck["latents"].detach().cpu().clone(),
            }
            if "denoised" in ck:
                row["denoised"] = ck["denoised"].detach().cpu().clone()
            current_trajectory.append(row)
        if step in denoise_steps:
            # For a denoising preview, decode the model's clean estimate x_0,
            # not the still-noisy state x_t used by CoCA trajectory analysis.
            captured[step] = ck.get("denoised", ck["latents"]).detach().cpu().clone()
        return ck

    def _cb_start(pipe_, i, t, ck):
        if current_trajectory is not None and not current_trajectory:
            current_trajectory.append({
                "step": 0,
                "timestep": float(t.item()) if torch.is_tensor(t) else float(t),
                "latents": ck["latents"].detach().cpu().clone(),
            })

    for ichunk in range(nchunk):

        current_trajectory = [] if save_trajectory else None

        preds = pipe.infer(
            video=obs.permute(0,2,1,3,4).to(device), # -> v, t, c, h, w
            cond_to_concat=rearrange(ichunk_cond_to_concat, "c v t h w -> v c t h w"), 
            prompt=[prompt, ],
            negative_prompt=negative_prompt,
            height=h, width=w, n_view=v,
            num_frames=args.data['train']['chunk'],
            num_inference_steps=args.num_inference_step,
            # decode_timestep=0.03,
            # decode_noise_scale=0.025,
            n_prev=args.data['train']['n_previous'],
            guidance_scale=1.0,
            merge_view_into_width=False,
            generator=generator,
            callback_on_step_end=_cb if (denoise_interval > 0 or save_trajectory) else None,
            callback_on_step_end_tensor_inputs=["latents", "denoised"],
            callback_on_step_start=_cb_start if save_trajectory else None,
            output_type="pt",
            postprocess_video=False,
        )['frames'] # preds: v c t h w , range -1 to 1 (could exceed range)

        if save_trajectory:
            trajectory_chunks.append(current_trajectory)
            current_trajectory = None

        videos = torch.cat((videos, preds.data.cpu()), dim=2) # v c t h w

        videos = torch.clamp(videos, min=-1, max=1)

        if denoise_interval > 0 and captured:
            for step, lat in captured.items():
                checkpoint_latents[(ichunk, step)] = lat
            captured.clear()

        if ichunk < nchunk-1:
            ### update memories and conditions
            ncur = videos.shape[2]
            mem_idxes = list(np.linspace(0, ncur-1, args.data['train']['n_previous']).astype(np.int16))

            obs = videos[:,:,mem_idxes].clone()
            ichunk_cond_to_concat = torch.cat((
                cond_to_concat[:,:,mem_idxes],
                cond_to_concat[:,:,args.data['train']['n_previous']+(ichunk+1)*args.data['train']['chunk']:args.data['train']['n_previous']+(ichunk+2)*args.data['train']['chunk']]
            ),dim=2)

            if ichunk_cond_to_concat.shape[2]<args.data['train']['chunk']+args.data['train']['n_previous']:
                ichunk_cond_to_concat = torch.cat([ichunk_cond_to_concat,] + [ichunk_cond_to_concat[:,:,-1:],]*(args.data['train']['chunk']-ichunk_cond_to_concat.shape[2]-args.data['train']['n_previous']), dim=2)

    n_total = ori_trajs.shape[2]
    video_to_save = torch.cat((rearrange(videos[:,:,:n_total], 'v c t h w -> c t h (v w)', v=v), rearrange(ori_trajs, 'c v t h w -> c t h (v w)', v=v),), dim=2)

    save_video(
        video_to_save,
        os.path.join(save_path, "video.mp4"),
        fps=video_fps
    )

    if save_trajectory or trajectory_path:
        trajectory_path = trajectory_path or os.path.join(save_path, "rollout", "trajectory.pt")
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
        torch.save({
            "chunks": trajectory_chunks,
            "num_chunks": nchunk,
            "num_inference_steps": args.num_inference_step,
            "valid_cams": cam_names,
            "seed": seed,
            "config_file": config_file,
            "includes_initial_latent": True,
            "includes_predicted_x0": True,
        }, trajectory_path)
        with open(os.path.splitext(trajectory_path)[0] + "_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "num_chunks": nchunk,
                "num_inference_steps": args.num_inference_step,
                "valid_cams": cam_names,
                "seed": seed,
                "config_file": config_file,
                "includes_initial_latent": True,
                "includes_predicted_x0": True,
            }, f, indent=2, ensure_ascii=False)

    if split_views:
        for i, cam_name in enumerate(cam_names):
            view_video = videos[i, :, :n_total, :, :]
            save_video(
                view_video,
                os.path.join(save_path, f"{cam_name}.mp4"),
                fps=video_fps
            )

    # 主视频/单视角已保存。检查点解码只需要 VAE，把不再用的扩散模型、
    # 文本编码器与数据张量全部释放出显存后再逐个解码（共享 GPU 上剩余显存不足）。
    if checkpoint_latents:
        prefix = videos[:, :, :args.data['train']['n_previous']].detach().cpu()
        del video_to_save, videos, ori_trajs, cond_to_concat, ichunk_cond_to_concat
        del obs, trajs, preds, generator
        del diffusion_model, text_encoder, pipe, tokenizer
        gc.collect()  # 破除 apply_forward_hook 等 module↔hook 引用环，del 不触发回收
        torch.cuda.empty_cache()
        print(f"[INFO] 主生成释放后显存: allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB "
              f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GiB")

        # 逐 step 处理：把该 step 各 chunk 的 latents 解码拼成完整视频 -> 立刻保存 -> 释放。
        # 避免原实现「先解码全部 (step,chunk) 再统一保存」，峰值随步数×块数线性增长。
        for step in sorted({s for (_, s) in checkpoint_latents}):
            ckpt_dir = os.path.join(save_path, f"denoise_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = prefix.clone()  # v c t h w（n_previous 帧）
            ok = False
            for chunk_idx in sorted({c for (c, s) in checkpoint_latents if s == step}):
                torch.cuda.empty_cache()
                try:
                    if hasattr(vae, "clear_cache"):
                        vae.clear_cache()
                    # _decode_latents 输出 (b v) c t h w，b=1 时即 v c t h w
                    vid = _decode_latents(checkpoint_latents[(chunk_idx, step)]).detach().cpu().to(prefix.dtype)
                except Exception as e:
                    print(f"[WARN] denoise_{step}（chunk {chunk_idx}）解码失败，跳过该 chunk: {e}")
                    continue
                ckpt = torch.cat((ckpt, vid), dim=2)
                ok = True
                del vid
            if not ok:
                continue
            ckpt = torch.clamp(ckpt[:, :, :n_total], min=-1, max=1)
            for j, cam_name in enumerate(cam_names):
                save_video(ckpt[j], os.path.join(ckpt_dir, f"{cam_name}.mp4"), fps=video_fps)
            del ckpt
            torch.cuda.empty_cache()


def args_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to observation images')
    parser.add_argument('--extrinsic_root', type=str, required=True, help='Path to extrinsics')
    parser.add_argument('--intrinsic_root', type=str, required=True, help='Path to intrinsics')
    parser.add_argument('--action_path', type=str, required=True, help='Path to actions')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--prompt', type=str, default="best quality, consistent and smooth motion, realistic, clear and distinct.")
    parser.add_argument('--split_views', action='store_true', help='Also save each camera view as a separate mp4 file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Noise seed for reproducibility; None = do not fix seed (default behavior, non-deterministic)')
    parser.add_argument('--denoise-interval', type=int, default=0,
                        help='每 N 个去噪步保存一个中间视频到 {output_path}/denoise_<step>/；0 = 关闭（默认）')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    ### For simplicity, this script directly load extrinsics, intrinsics and actions from .npy files.
    ### We also provide a conversion script `gesim_video_gen_examples/get_example_gesim_inputs.py` to demonstrate how to generate these .npy files.

    args = args_parser()
    print(args)

    infer(
        args.config_file, args.image_root, args.extrinsic_root, args.intrinsic_root, args.action_path, args.prompt, args.output_path,
        seed=args.seed, split_views=args.split_views, denoise_interval=args.denoise_interval
    )
