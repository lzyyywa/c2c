import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# [NEW] Import the custom hyperbolic math library
# Ensure codes/utils/hyperbolic_ops.py is created first
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("[Error] Could not import LorentzMath! Please create codes/utils/hyperbolic_ops.py first.")

_tokenizer = _Tokenizer()

# ==================================================================================
# 1. Basic Components (Unchanged)
# ==================================================================================

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj
        self.num_frames = cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)
        return out


# ==================================================================================
# 2. CustomCLIP (Core Modification Area: Hyperbolic Transformation)
# ==================================================================================

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        # Text Prompt Learners (Fine-grained)
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        # Encoders
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        # Independent Learning Modules
        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        # int(cfg.emb_dim) here refers to the Euclidean space dimension
        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        # [NEW] Placeholders: Store tokens for hierarchical texts (Parent & Composition)
        # These will be populated when set_hierarchy_prompts is called in train_models.py
        self.coarse_verb_tokens = None
        self.coarse_obj_tokens = None
        self.comp_tokens = None
        
        # Save reference to clip.tokenize for subsequent use
        self.clip_tokenize = clip.tokenize
        # Save clip's token embedding layer for encoding texts without prompt learner
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding

    def set_hierarchy_prompts(self, coarse_verbs, coarse_objs, pairs):
        """
        [NEW] Set hierarchical texts (extracted and passed by HierarchyHelper)
        """
        print(f"[CustomCLIP] Setting up hierarchy prompts...")
        
        # 1. Coarse Verbs & Objects
        # Templatized: "a photo of [Parent] something"
        self.coarse_verb_tokens = self.clip_tokenize([f"a photo of {v} something" for v in coarse_verbs])
        self.coarse_obj_tokens = self.clip_tokenize([f"a photo of something {o}" for o in coarse_objs])
        
        # 2. Composition (Pairs) - Used only for Loss constraints
        # Templatized: "a photo of [Verb] [Object]"
        pair_texts = [f"a photo of {v} {o}" for v, o in pairs]
        self.comp_tokens = self.clip_tokenize(pair_texts)
        
        print(f"[CustomCLIP] Hierarchy ready: {len(coarse_verbs)} coarse verbs, {len(coarse_objs)} coarse objs, {len(pairs)} composition pairs.")

    def _encode_plain_text(self, tokenized_prompts, device):
        """
        [NEW] Helper function: Encode plain texts without PromptLearner (e.g., Coarse/Comp)
        """
        # 1. Token Embeddings
        x = self.token_embedding(tokenized_prompts).type(self.text_encoder.dtype)
        
        # 2. Add Positional Embeddings
        x = x + self.positional_embedding.type(self.text_encoder.dtype)
        
        # 3. Pass through Transformer
        return self.text_encoder(x, tokenized_prompts)

    def forward(self, video, pairs=None):
        device = video.device

        # =================================================
        # 1. Text Encoding -> Euclidean Space
        # =================================================
        
        # (A) Fine-grained Verb
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features) # [N_verb, D]

        # (B) Fine-grained Object
        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)   # [N_obj, D]

        # =================================================
        # 2. Video Encoding -> Euclidean Space
        # =================================================
        video_features = self.video_encoder(video) # [B, D, T]

        # Object Branch: Mean Pooling -> MLP
        o_feat = self.c2c_OE1(video_features.mean(dim=-1)) # [B, D]
        
        # Verb Branch: Conv1d -> Mean Pooling
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1) # [B, D]

        # =================================================
        # 3. Space Transformation: Euclidean -> Hyperbolic [Key Modification]
        # =================================================
        # Replace original F.normalize with LorentzMath.exp_map_0
        # It is recommended to normalize first then apply exp_map to ensure inputs are on the unit sphere for numerical stability
        
        # (A) Hyperbolic transformation of video features
        o_feat_norm = F.normalize(o_feat, dim=1)
        v_feat_norm = F.normalize(v_feat, dim=1)
        
        o_feat_hyp = LorentzMath.exp_map_0(o_feat_norm) # [B, D+1]
        v_feat_hyp = LorentzMath.exp_map_0(v_feat_norm) # [B, D+1]

        # (B) Hyperbolic transformation of text features
        verb_text_norm = F.normalize(verb_text_features, dim=-1)
        obj_text_norm = F.normalize(obj_text_features, dim=-1)
        
        verb_text_hyp = LorentzMath.exp_map_0(verb_text_norm) # [N_verb, D+1]
        obj_text_hyp = LorentzMath.exp_map_0(obj_text_norm)   # [N_obj, D+1]

        # =================================================
        # 4. Compute Additional Features for Hierarchical Constraints (Coarse & Comp)
        # =================================================
        coarse_verb_hyp = None
        coarse_obj_hyp = None
        comp_hyp = None

        # Compute only after set_hierarchy_prompts is called (usually in Training phase)
        if self.coarse_verb_tokens is not None:
            # Move tokens to device
            cv_tokens = self.coarse_verb_tokens.to(device)
            co_tokens = self.coarse_obj_tokens.to(device)
            c_tokens = self.comp_tokens.to(device) # Composition

            # Encode (Euclidean)
            cv_emb = self._encode_plain_text(cv_tokens, device)
            co_emb = self._encode_plain_text(co_tokens, device)
            c_emb = self._encode_plain_text(c_tokens, device)
            
            # Project to Hyperbolic Space
            coarse_verb_hyp = LorentzMath.exp_map_0(F.normalize(cv_emb, dim=-1))
            coarse_obj_hyp = LorentzMath.exp_map_0(F.normalize(co_emb, dim=-1))
            comp_hyp = LorentzMath.exp_map_0(F.normalize(c_emb, dim=-1))

        # =================================================
        # 5. Compute Logits (Negative Hyperbolic Distance)
        # =================================================
        # Compute Pairwise hyperbolic distances
        # v_feat_hyp: [B, D+1], verb_text_hyp: [N, D+1]
        
        # Output dists: [B, N]
        dist_v = LorentzMath.hyp_distance(v_feat_hyp.unsqueeze(1), verb_text_hyp.unsqueeze(0))
        dist_o = LorentzMath.hyp_distance(o_feat_hyp.unsqueeze(1), obj_text_hyp.unsqueeze(0))
        
        # Logits = -Distance (Smaller distance indicates higher similarity)
        verb_logits = -dist_v
        obj_logits = -dist_o

        # =================================================
        # 6. Return Results
        # =================================================
        
        # Training mode: Return Dict containing Logits and all Embeddings (for Loss calculation)
        if self.training:
            # Note: Keys must match the unpacked keys in train_models.py
            return {
                "verb_logits": verb_logits,     # [B, N_verb]
                "obj_logits": obj_logits,       # [B, N_obj]
                
                # Features for Cone Loss
                "v_feat_hyp": v_feat_hyp,       # [B, D+1]
                "o_feat_hyp": o_feat_hyp,       # [B, D+1]
                "verb_text_hyp": verb_text_hyp, # [N_verb, D+1]
                "obj_text_hyp": obj_text_hyp,   # [N_obj, D+1]
                
                # Additional hierarchical features
                "coarse_verb_hyp": coarse_verb_hyp,
                "coarse_obj_hyp": coarse_obj_hyp,
                "comp_hyp": comp_hyp
            }
        
        # Inference mode: Return composed scores
        else:
            # If pairs are specified (usually in test/val loop)
            if pairs is not None:
                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                
                # Extract corresponding logits
                # verb_logits: [B, N_verb] -> [B, N_pairs]
                v_score = verb_logits[:, verb_idx]
                o_score = obj_logits[:, obj_idx]
                
                # Composed score: Simple addition (in Log probability space)
                # P(v,o) = P(v) * P(o)  =>  log P(v,o) = log P(v) + log P(o)
                # Logits here are already -Distance (equivalent to log P)
                com_logits = v_score + o_score
                
                return com_logits
            
            # If no pairs are specified (rare case), return original logits
            else:
                return verb_logits, obj_logits


# ==================================================================================
# 3. Helper Loading Functions (Unchanged)
# ==================================================================================

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


def build_model(train_dataset, cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    print("Building custom CLIP (Hyperbolic Version)")
    model = CustomCLIP(cfg, train_dataset, clip_model)

    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name and cfg.learn_input_method != 'zero':
            param.requires_grad_(True)
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
        elif 'c2c' in name:
            param.requires_grad = True
    return model