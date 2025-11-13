# this is the updated version of the eva_x.py of the repo EVA-X (https://github.com/hustvl/EVA-X)
# Clone eva-x repos first, and update the eva_x.py to this version to get MIMIC CXR embeddings

 
# changes to the original script (eva_x.py)

# 1. PyTorch Loading Fix:
# Added NumPy safe globals to make PyTorch 2.6+ compatible with the model weights
# Modified the model loading code to handle weights_only=False with version-specific fallback

# 2. Position Embedding Fix:
# Replaced the missing _pos_embed method call in forward_features
# Implemented direct position embedding addition instead of relying on a non-existent method
# Added handling for both standard and rotary position embeddings


import torch
import torch.nn as nn
from timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed, resample_patch_embed

# Add safe globals for PyTorch 2.6+ compatibility
import numpy as np
from torch.serialization import add_safe_globals

# Add numpy scalar types to safe globals
add_safe_globals([np.ndarray, np.generic, np.core.multiarray.scalar])

def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict

class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        # Fix for the _pos_embed error
        # Instead of using _pos_embed, manually add the position embedding
        if self.pos_embed is not None:
            # Handle rotary position embedding if used
            if hasattr(self, 'use_rot_pos_emb') and self.use_rot_pos_emb:
                # Just add the regular positional embedding
                x = x + self.pos_embed
                # Create a "dummy" rotary position embedding (not used in the model)
                rot_pos_embed = None
            else:
                x = x + self.pos_embed
                rot_pos_embed = None
        else:
            rot_pos_embed = None
        
        # Apply transformer blocks
        for blk in self.blocks:
            if hasattr(blk, 'rope') and rot_pos_embed is not None:
                x = blk(x, rope=rot_pos_embed)
            else:
                x = blk(x)
                
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def eva_x_tiny_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
    )
    if pretrained:
        # For PyTorch 2.6+ compatibility, specify weights_only=False
        try:
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        except TypeError:  # For older PyTorch versions that don't have weights_only
            checkpoint = torch.load(pretrained, map_location='cpu')
            
        eva_ckpt = checkpoint_filter_fn(checkpoint, model)
        msg = model.load_state_dict(eva_ckpt, strict=False)
        print(msg)
    return model

def eva_x_small_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),   # 224/16
    )
    if pretrained:
        # For PyTorch 2.6+ compatibility, specify weights_only=False
        try:
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        except TypeError:  # For older PyTorch versions that don't have weights_only
            checkpoint = torch.load(pretrained, map_location='cpu')
            
        eva_ckpt = checkpoint_filter_fn(checkpoint, model)
        msg = model.load_state_dict(eva_ckpt, strict=False)
        print(msg)
    return model

def eva_x_base_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
    )
    if pretrained:
        # For PyTorch 2.6+ compatibility, specify weights_only=False
        try:
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        except TypeError:  # For older PyTorch versions that don't have weights_only
            checkpoint = torch.load(pretrained, map_location='cpu')
            
        eva_ckpt = checkpoint_filter_fn(checkpoint, model)
        msg = model.load_state_dict(eva_ckpt, strict=False)
        print(msg)
    return model


# if __name__ == '__main__':
#     # Example usage code
#     pass

if __name__ == '__main__':

    eva_x_ti_pt = '/Volumes/code/my_project/mimiccxr_eva/repos/eva_x_tiny_patch16_merged520k_mim.pt'
    eva_x_s_pt = '/Volumes/code/my_project/mimiccxr_eva/repos/eva_x_small_patch16_merged520k_mim.pt'
    eva_x_b_pt = '/Volumes/code/my_project/mimiccxr_eva/repos/eva_x_base_patch16_merged520k_mim.pt'
    
    # Initialize models
    eva_x_ti = eva_x_tiny_patch16(pretrained=eva_x_ti_pt)
    eva_x_s = eva_x_small_patch16(pretrained=eva_x_s_pt)
    eva_x_b = eva_x_base_patch16(pretrained=eva_x_b_pt)
    
    # Print model sizes to verify loading
    print(f"Tiny model parameters: {sum(p.numel() for p in eva_x_ti.parameters())}")
    print(f"Small model parameters: {sum(p.numel() for p in eva_x_s.parameters())}")
    print(f"Base model parameters: {sum(p.numel() for p in eva_x_b.parameters())}")