from torch.nn.modules.loss import CrossEntropyLoss

# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# [NEW] Import Hyperbolic Operations
# Ensure codes/utils/hyperbolic_ops.py exists
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("Warning: Could not import LorentzMath from utils.hyperbolic_ops. Hyperbolic losses will not work.")

def loss_calu(predict, target, config):
    """
    Original loss calculation function. 
    Kept for backward compatibility or if you want to switch back to Euclidean baseline.
    """
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits, logits_att, logits_obj, logits_soft_prompt = predict
    loss_logit_df = loss_fn(logits, batch_target)
    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj) + config.sp_w * loss_logit_sp
    return loss

# =========================================================================
# [NEW] Hyperbolic Losses for C2C Migration
# =========================================================================

class HyperbolicPrototypicalLoss(nn.Module):
    """
    Discriminative Loss in Hyperbolic Space.
    Replaces Standard CrossEntropy/Cosine Loss.
    
    Inputs are expected to be Hyperbolic Embeddings (with time dimension).
    We calculate the negative hyperbolic distance as logits.
    """
    def __init__(self, temperature=0.1, c=1.0):
        super(HyperbolicPrototypicalLoss, self).__init__()
        self.temperature = temperature
        self.c = c
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_emb, prototype_emb, targets):
        """
        Args:
            query_emb: [Batch, Dim+1] (e.g., Video Features in Hyp Space)
            prototype_emb: [Num_Classes, Dim+1] (e.g., Verb/Object Text Features in Hyp Space)
            targets: [Batch] Class Indices
        """
        # Expand dims for broadcasting
        # query: [B, 1, D]
        # proto: [1, N, D]
        q = query_emb.unsqueeze(1)
        p = prototype_emb.unsqueeze(0)
        
        # Calculate Hyperbolic Distance
        # dists shape: [Batch, Num_Classes]
        dists = LorentzMath.hyp_distance(q, p, c=self.c)
        
        # Convert to logits: closer distance = higher probability
        # Logits = -distance / T
        logits = -dists / self.temperature
        
        # Calculate Cross Entropy
        return self.loss_fn(logits, targets)

class EntailmentConeLoss(nn.Module):
    """
    Hierarchical Entailment Cone Loss.
    Enforces that 'child' embeddings lie within the 'cone' of 'parent' embeddings.
    
    Used for:
    1. Verb -> Coarse Verb
    2. Object -> Coarse Object
    3. Composition -> Verb
    4. Composition -> Object
    """
    def __init__(self, aperture=0.01, margin=0.01):
        super(EntailmentConeLoss, self).__init__()
        self.aperture = aperture  # Cone aperture angle (K in paper)
        self.margin = margin

    def forward(self, child_emb, parent_emb):
        """
        Args:
            child_emb: [Batch, Dim+1] 
            parent_emb: [Batch, Dim+1] (Must be aligned: parent_emb[i] is parent of child_emb[i])
        """
        # 1. Depth Check (Norm Penalty)
        # In Lorentz model, x[0] (time component) represents distance from origin.
        # Parent (abstract) should be closer to origin (smaller x[0]) than Child (specific).
        # We want: child_x0 > parent_x0
        
        child_r = child_emb[..., 0]
        parent_r = parent_emb[..., 0]
        
        # Penalty if parent is further or equal to child (plus margin)
        # Loss = max(0, parent_r - child_r + margin)
        depth_loss = F.relu(parent_r - child_r + self.margin)

        # 2. Angle Check (Cone Boundary)
        # Cosine similarity in the space components (approximation of angle in tangent space)
        child_space = child_emb[..., 1:]
        parent_space = parent_emb[..., 1:]
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(child_space, parent_space, dim=-1)
        
        # We want cos_sim > (1 - aperture)
        # If cos_sim is smaller, it means angle is too large -> Penalty
        angle_loss = F.relu((1.0 - self.aperture) - cos_sim)

        return depth_loss.mean() + angle_loss.mean()

# =========================================================================
# [END] New Hyperbolic Losses
# =========================================================================

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label,mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss* batch_size
        else:
            return loss


def hsic_loss(input1, input2, unbiased=False):
    def _kernel(X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        # tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        # tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    Loss from No One Left Behind: Improving the Worst Categories in Long-Tailed Learning
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        '''

        Args:
            p_o_on_v: b,n_v,n_o
            o_label: b,
            n_c: b,n_o

        Returns:

        '''
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  # b,n_o

        num_c = n_c.sum().view(1, -1)  # 1,n_o

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  # b,n_o
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  # b,n_o

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  # 1,n_o
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss