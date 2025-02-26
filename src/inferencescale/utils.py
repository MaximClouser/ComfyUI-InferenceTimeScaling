import gc
import torch
import torch.nn.functional as F

import time
import numpy as np
import comfy.model_management as model_management


def clear_memory(model_name):
    """
    Unloads the specified model from memory (if loaded) and clears PyTorch cache.
    """
    print("Unloading Model:")
    loaded_models = model_management.loaded_models()
    
    if model_name in loaded_models:
        print(" - Model found in memory, unloading...")
        loaded_models.remove(model_name)
    
    model_management.free_memory(1e30, model_management.get_torch_device(), loaded_models)
    model_management.soft_empty_cache(True)
    
    # Clear Python and CUDA caches
    try:
        print("Clearing Cache...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        print("Unable to clear cache")
    time.sleep(2)


def generate_neighbors(x, threshold=0.95, num_neighbors=4):
    """Generates neighbors on the same sphere with cosine similarity = threshold."""
    # Function adopted from: https://github.com/sayakpaul/tt-scale-flux
    rng = np.random.Generator(np.random.PCG64())
    x_f = x.flatten(1)
    x_norm = torch.linalg.norm(x_f, dim=-1, keepdim=True, dtype=torch.float64).unsqueeze(-2)
    u = x_f.unsqueeze(-2) / x_norm.clamp_min(1e-12)
    v = torch.from_numpy(rng.standard_normal(size=(u.shape[0], num_neighbors, u.shape[-1]), dtype=np.float64)).to(u.device)
    w = F.normalize(v - (v @ u.transpose(-2, -1)) * u, dim=-1)
    return (x_norm * (threshold * u + np.sqrt(1 - threshold**2) * w)).reshape(x.shape[0], num_neighbors, *x.shape[1:]).to(x.dtype)
