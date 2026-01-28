import torch
import torch.nn.functional as F

class LorentzMath:
    """
    Static class for Hyperbolic operations in the Lorentz model.
    Based on H2EM paper equations for converting C2C embeddings.
    
    This utility handles the geometry:
    1. Manifold: Lorentz Model (-x0^2 + x1^2 + ... + xn^2 = -1/c)
    2. Tangent Space: Euclidean space at origin (where CLIP features live initially)
    """

    @staticmethod
    def arcosh(x, eps=1e-7):
        """
        Numerically stable arcosh.
        arcosh(x) = log(x + sqrt(x^2 - 1))
        Input x must be >= 1. We clamp it to 1 + eps.
        """
        x = torch.clamp(x, min=1.0 + eps)
        return torch.log(x + torch.sqrt(x**2 - 1))

    @staticmethod
    def lorentz_product(x, y, keepdim=False):
        """
        Minkowski Inner Product between two vectors in Lorentz space.
        Formula: <x, y> = -x0*y0 + x1*y1 + ... + xn*yn
        
        Args:
            x: [..., d+1] Tensor
            y: [..., d+1] Tensor
        """
        # Element-wise product
        prod = x * y
        
        # Split into time component (index 0) and space components (index 1:)
        # We assume the last dimension is the feature dimension
        time_prod = -prod[..., 0:1]
        space_prod = torch.sum(prod[..., 1:], dim=-1, keepdim=True)
        
        res = time_prod + space_prod
        
        if not keepdim:
            res = res.squeeze(-1)
        return res

    @staticmethod
    def exp_map_0(x, c=1.0):
        """
        Exponential Map at the origin (Tangent Space -> Hyperbolic Manifold).
        This maps your Euclidean CLIP features to Hyperbolic Space.
        
        Formula (H2EM Eq. 5):
            x_E \in R^d (Euclidean)
            x_H = [cosh(sqrt(c)|x|), sinh(sqrt(c)|x|) * x / (sqrt(c)|x|)]
        
        Args:
            x: Euclidean features [..., d] (e.g., Output from CLIP image/text encoder)
            c: Curvature (scalar, default 1.0)
            
        Returns:
            z: Hyperbolic features [..., d+1] (Time dimension prepended)
        """
        # Ensure c is a tensor on the correct device
        sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
        
        # 1. Calculate Euclidean norm |x|
        # Add eps to avoid division by zero
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-7)
        
        # 2. Calculate time component x0
        # x0 = cosh(sqrt(c) * |x|)
        theta = sqrt_c * x_norm
        x0 = torch.cosh(theta)
        
        # 3. Calculate space components x1...xn
        # coeff = sinh(theta) / (theta) * (1/sqrt(c) * sqrt(c)) -> sinh(theta) / |x_norm| * (1/sqrt(c)) 
        # Easier: space = (sinh(theta) / theta) * x  <-- NO, verify carefully:
        # Correct Formula: sinh(sqrt(c)|x|) * (x / |x|) * (1/sqrt(c))
        #               = (sinh(theta) / theta) * x
        # Note: sin(x)/x is sinc function, PyTorch has torch.sinc but it's for sin, not sinh.
        # We compute manually to be safe.
        
        scale_factor = torch.sinh(theta) / (theta) 
        x_space = scale_factor * x
        
        # 4. Concatenate along the last dimension
        z = torch.cat([x0, x_space], dim=-1)
        
        return z

    @staticmethod
    def hyp_distance(x, y, c=1.0, keepdim=False):
        """
        Geodesic distance in Lorentz model.
        Used for calculating the similarity score (negative distance).
        
        Formula (H2EM Eq. 2):
            d(x, y) = 1/sqrt(c) * arcosh( -c * <x, y>_L )
            
        Args:
            x, y: Hyperbolic features [..., d+1]
        """
        sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
        
        # 1. Calculate Lorentz Product <x, y>_L
        # Result should be <= -1/c (approximately)
        inner = LorentzMath.lorentz_product(x, y, keepdim=True)
        
        # 2. Scale by curvature: -c * inner
        # We explicitly clamp the inner product to avoid NaN in arcosh
        # Because floating point errors might make -c*inner slightly less than 1.0
        val = -1.0 * c * inner
        
        # 3. Calculate Distance
        dist = LorentzMath.arcosh(val) / sqrt_c
        
        if not keepdim:
            dist = dist.squeeze(-1)
            
        return dist

    @staticmethod
    def check_validity(x, c=1.0, tol=1e-4):
        """
        Debug helper: Check if vectors satisfy the Lorentz constraint.
        -x0^2 + x1^2 + ... = -1/c
        """
        inner = LorentzMath.lorentz_product(x, x)
        target = -1.0 / c
        diff = torch.abs(inner - target)
        return torch.all(diff < tol), diff.max().item()