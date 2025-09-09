# scripts/inference.py
import argparse
import math
import os
import sys
import yaml
import torch
import numpy as np
from easydict import EasyDict
import glob

sys.path.append(os.getcwd())

def add_diffuse_gesture_path():
    """Encuentra las rutas necesarias de DiffuseStyleGesture y las devuelve."""
    possible_roots = [
        '/workspace/DiffuseStyleGesture/BEAT-TWH-main',
        '/root/DiffuseStyleGesture/BEAT-TWH-main',
        os.path.abspath(os.path.join(os.getcwd(), 'DiffuseStyleGesture', 'BEAT-TWH-main'))
    ]
    
    diffuse_root = None
    for path in possible_roots:
        if os.path.exists(path):
            diffuse_root = path
            break
            
    if not diffuse_root:
        raise ImportError("No se pudo encontrar el repositorio DiffuseStyleGesture/BEAT-TWH-main.")

    print(f"Dependencias de DiffuseStyleGesture encontradas en: {diffuse_root}")
    
    sys.path.extend([
        diffuse_root,
        os.path.join(diffuse_root, 'process'),
        os.path.join(diffuse_root, 'utils'),
        os.path.join(diffuse_root, 'model'),
    ])
    
    return diffuse_root



 
import pdb
import subprocess
from datetime import datetime
import copy
import librosa
from pprint import pprint
import torch.nn.functional as F
import re

# Mapeo de modelos a sus configuraciones
MODEL_CONFIGS = {
    'Ref-Basic': {'audio_feature_dim': 1583},
    'Basic-Whisper': {'audio_feature_dim': 1583},
    'Multi-Fusion': {'audio_feature_dim': 4207},
    'Multi-Dual': {'audio_feature_dim': 4207},
    'Text-Only': {'audio_feature_dim': 4207},
    'Multi-Whisper': {'audio_feature_dim': 4355},
    'Multi-DiT': {'audio_feature_dim': 4207},
    'Text-DiT': {'audio_feature_dim': 4207},
}

def create_model_and_diffusion(args, model_name, device_name):
    from models.mdm import MDM
    from scripts.model_util import create_gaussian_diffusion, load_model_wo_clip
    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim,
                audio_feat_dim_latent=args.audio_feat_dim_latent, model_name=model_name)
      
    diffusion = create_gaussian_diffusion()
    
    return model, diffusion

def inference(args, save_dir, prefix, textaudio, sample_fn, model, mydevice, config, data_paths, n_frames=0, smoothing=False, skip_timesteps=0, style=None, seed=123456):
    from process_TWH_bvh import pose2bvh as pose2bvh_twh
    torch.manual_seed(seed)
    speaker = np.where(style == np.max(style))[0][0]

    if n_frames == 0:
        n_frames = textaudio.shape[0]
    else:
        textaudio = textaudio[:n_frames]

    real_n_frames = n_frames
    stride_poses = args.n_poses - args.n_seed
    num_subdivision = math.ceil(n_frames / stride_poses) if stride_poses > 0 else 1
    n_frames = num_subdivision * stride_poses if stride_poses > 0 else n_frames
    print(f'real_n_frames: {real_n_frames}, num_subdivision: {num_subdivision}, stride_poses: {stride_poses}, n_frames: {n_frames}, speaker_id: {speaker}')

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim]).to(mydevice)
    textaudio = torch.cat((textaudio, textaudio_pad), 0)
    audio_reshape = textaudio.reshape(num_subdivision, stride_poses, args.audio_feature_dim).transpose(0, 1)

    # --- CAMBIO 2: Cargar datos desde las rutas absolutas proporcionadas ---
    data_mean = np.load(data_paths['mean'])
    data_std = np.load(data_paths['std'])
    seed_gesture_path = data_paths['seed']
    pipeline_path = data_paths['pipeline']

    shape_ = (1, model.njoints, model.nfeats, args.n_poses)
    out_list = []

    for i in range(num_subdivision):
        print(f"{i+1}/{num_subdivision}")
        model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1].transpose(0,1)

        if i == 0:
            seed_gesture = np.load(seed_gesture_path)[:args.n_seed + 2]
            # ... (el resto de la lógica de la inferencia no cambia)
            seed_gesture = (seed_gesture - data_mean) / data_std
            seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
            seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
            seed_gesture_ = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1)
            seed_gesture_ = torch.from_numpy(seed_gesture_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
            model_kwargs_['y']['seed'] = seed_gesture_.unsqueeze(2)
        else:
            model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed:].to(mydevice)

        sample = sample_fn(
            model, shape_, clip_denoised=False, model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps, progress=True
        )
        if len(out_list) > 0 and args.n_seed != 0:
            last_poses = out_list[-1][..., -args.n_seed:]
            out_list[-1] = out_list[-1][..., :-args.n_seed]
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                next = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
        out_list.append(sample)

    motion_feature_division = 3 if "v0" in config.version else 1
    out_list = [i.detach().data.cpu().numpy()[:, :args.njoints // motion_feature_division] for i in out_list]
    if len(out_list) > 1:
        sampled_seq_1 = np.vstack(out_list[:-1]).squeeze(2).transpose(0, 2, 1).reshape(1, -1, model.njoints // motion_feature_division)
        sampled_seq_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, sampled_seq_2), axis=1)
    else:
        sampled_seq = np.array(out_list[0]).squeeze(2).transpose(0, 2, 1)

    if args.n_seed > 0:
        sampled_seq = sampled_seq[:, args.n_seed:]
    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    out_poses = out_poses[:real_n_frames]
    
    pose2bvh_twh(out_poses, save_dir, prefix, pipeline_path=pipeline_path)


def main(args, config, diffuse_root):
    from scripts.model_util import load_model_wo_clip
    from process_TWH_bvh import load_metadata

    # Construir la nueva ruta de guardado
    model_checkpoint_name = os.path.basename(args.model_path).split('.')[0]
    folder_name = f"{args.model_name}_{model_checkpoint_name}"
    args.save_dir = os.path.join("bvh_generated", folder_name)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    print(f"--- The files will be saved in: {os.path.abspath(args.save_dir)} ---")
        
    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model name '{args.model_name}' no es válido.")

    config.audio_feature_dim = MODEL_CONFIGS[args.model_name]['audio_feature_dim']
    
    device_name = f'cuda:{args.gpu}'
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(config, args.model_name, device_name)
    
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    
    model.to(mydevice)
    model.eval()
    
    sample_fn = diffusion.p_sample_loop

    data_paths = {
        'mean': os.path.join(diffuse_root, 'process', 'gesture_TWH_mean_v0.npy'),
        'std': os.path.join(diffuse_root, 'process', 'gesture_TWH_std_v0.npy'),
        'seed': 'data/val_2023_v0_014_main-agent.npy',
        'pipeline': os.path.join(diffuse_root, 'process', 'resource', 'pipeline_rotmat_62.sav')
    }

    combined_files = glob.glob(os.path.join(args.txt_path, '*.npy'))
    print(f'Total number of combined embedding files: {len(combined_files)}')

    metadatapath = os.path.join(args.metadata_path, "metadata.csv")
    _, metadict_byfname, _ = load_metadata(metadatapath, "main-agent")

    for combined_file in combined_files:
        filename = os.path.basename(combined_file).replace('_text_audio.npy', '').replace('.npy', '')
        print(f"Processing: {filename}")
        
        textaudio = np.load(combined_file)
        textaudio = torch.FloatTensor(textaudio).to(mydevice)
        _, speaker_id = metadict_byfname.get(filename, (None, 0))
        
        speaker = np.zeros([17])
        speaker[speaker_id] = 1
        print(f"File: {filename}, Speaker ID: {speaker_id}")
        
        inference(config, args.save_dir, filename, textaudio, sample_fn, model, mydevice, config, data_paths, n_frames=args.max_len, smoothing=True, skip_timesteps=args.skip_timesteps, style=speaker, seed=123456)


if __name__ == '__main__':
    diffuse_root = add_diffuse_gesture_path()
    
    possible_config_paths = [
        '/root/DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/configs/DiffuseStyleGesture.yml',
        '/workspace/DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/configs/DiffuseStyleGesture.yml',
        'DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/configs/DiffuseStyleGesture.yml'
    ]
    
    default_config_path = None
    for path in possible_config_paths:
        if os.path.exists(path):
            default_config_path = path
            break

    if default_config_path is None:
        raise FileNotFoundError("No se pudo encontrar 'DiffuseStyleGesture.yml' en las rutas esperadas.")

    parser = argparse.ArgumentParser(description='Unified Gesture Generation')
    parser.add_argument('--config', default=default_config_path)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, default='data/tst/')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--model_name', type=str, required=True, help=f"Elige entre: {list(MODEL_CONFIGS.keys())}")
    
    args = parser.parse_args()
    
    print(f"Cargando configuración desde: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = EasyDict(config)
    
    config.njoints = 2232
    config.latent_dim = 512
    config.audio_feat_dim_latent = 128
    config.style_dim = 17
    config.cond_mode = 'cross_local_attention4_style1_sample'
    config.version = 'v0' 
    config.n_poses = 150 
    config.n_seed = 30 
    config.audio_feat = 'wavlm'
    config.cond_mask_prob = 0.1 

    for k, v in vars(args).items():
        config[k] = v

    main(args, config, diffuse_root)