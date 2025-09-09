# scripts/model_util.py
import sys
import os

sys.path.append(os.getcwd())

def add_diffuse_gesture_path():
    """Encuentra y añade las rutas necesarias de DiffuseStyleGesture."""
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
        
    sys.path.extend([
        diffuse_root,
        os.path.join(diffuse_root, 'utils'),
        # os.path.join(diffuse_root, 'model'),
    ])

add_diffuse_gesture_path()

import pdb
from models.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model_wo_clip(model, state_dict):
    """
    Carga los pesos de un state_dict en un modelo, ignorando las claves relacionadas con CLIP.
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Claves faltantes en el modelo: {missing_keys}")
    print(f"Claves inesperadas en el state_dict: {unexpected_keys}")
    assert len(unexpected_keys) == 0
    # Asegura que las únicas claves que faltan son las de clip_model, que no se usan en la inferencia.
    if missing_keys:
        assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion(args, data):
    """
    Crea el modelo y el proceso de difusión para el entrenamiento/evaluación.
    """
    # NOTA: Para usar esto con el modelo unificado, se debe añadir 'model_name' a los argumentos.
    model_args = get_model_args(args, data)
    model = MDM(**model_args)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_model_args(args, data):
    """
    Construye el diccionario de argumentos para el modelo MDM a partir de args y datos.
    """
    # Lógica para determinar cond_mode
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'

    # Parámetros por defecto para TWH (ajustados según tu caso de uso)
    data_rep = 'rot6d'
    njoints = 2232  # TWH-specific
    nfeats = 1

    # Sobrescribir para otros datasets si es necesario
    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251

    return {
        'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
        'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
        'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
        'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch, 'dataset': args.dataset,
        # AÑADIR ESTA LÍNEA CUANDO ENTRENES:
        'model_name': args.model_name
    }

def create_gaussian_diffusion():
    # Parámetros de difusión predeterminados, como en tu script original
    steps = 1000
    noise_schedule = 'cosine'
    predict_xstart = True
    learn_sigma = False
    sigma_small = True
    timestep_respacing = ''
    
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    loss_type = gd.LossType.MSE
    
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.START_X if predict_xstart else gd.ModelMeanType.EPSILON),
        model_var_type=((gd.ModelVarType.FIXED_SMALL if sigma_small else gd.ModelVarType.FIXED_LARGE) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
    )