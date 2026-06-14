"""
Denoiser Networks for Geometric Diffusion

These networks predict the noise in a noisy image, enabling
the reverse diffusion process to generate clean images.

Adapted from EigenScript's transformer patterns with geometric
training support.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (matches transformer.py)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def sinusoidal_embedding(timestep: int, dim: int) -> np.ndarray:
    """
    Sinusoidal timestep embedding (like positional encoding in transformers).
    
    This tells the denoiser what noise level to expect.
    """
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = np.concatenate([np.sin(emb), np.cos(emb)])
    return emb.astype(np.float32)


@dataclass
class DenoiserConfig:
    """Configuration for denoiser network."""
    image_size: int = 32
    channels: int = 1
    hidden_dim: int = 64
    num_layers: int = 3
    timestep_dim: int = 32


class SimpleDenoiser:
    """
    Simple convolutional denoiser for proof of concept.
    
    Uses depthwise-separable-like convolutions implemented with
    matrix operations for compatibility with EigenScript tensors.
    """
    
    def __init__(self, config: Optional[DenoiserConfig] = None):
        self.config = config or DenoiserConfig()
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        c = self.config
        
        self.time_mlp_1 = np.random.randn(c.timestep_dim, c.hidden_dim).astype(np.float32) * 0.02
        self.time_mlp_2 = np.random.randn(c.hidden_dim, c.hidden_dim).astype(np.float32) * 0.02
        
        in_features = c.channels * c.image_size * c.image_size
        self.proj_in = np.random.randn(in_features, c.hidden_dim).astype(np.float32) * 0.02
        
        self.layers = []
        for _ in range(c.num_layers):
            layer = {
                'w1': np.random.randn(c.hidden_dim, c.hidden_dim * 2).astype(np.float32) * 0.02,
                'w2': np.random.randn(c.hidden_dim * 2, c.hidden_dim).astype(np.float32) * 0.02,
            }
            self.layers.append(layer)
        
        self.proj_out = np.random.randn(c.hidden_dim, in_features).astype(np.float32) * 0.02
    
    def __call__(self, x: np.ndarray, timestep: int, conditioning: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict noise in image x at timestep t."""
        return self.forward(x, timestep, conditioning)
    
    def forward(self, x: np.ndarray, timestep: int, conditioning: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass: predict noise in x at timestep.
        
        Args:
            x: Noisy image (channels, height, width)
            timestep: Current diffusion timestep
            conditioning: Optional conditioning embedding
        
        Returns:
            Predicted noise (same shape as x)
        """
        original_shape = x.shape
        c = self.config
        
        t_emb = sinusoidal_embedding(timestep, c.timestep_dim)
        t_emb = gelu(t_emb @ self.time_mlp_1)
        t_emb = t_emb @ self.time_mlp_2
        
        if conditioning is not None:
            if len(conditioning.shape) == 1:
                t_emb = t_emb + conditioning[:c.hidden_dim]
        
        h = x.flatten()
        h = h @ self.proj_in
        h = layer_norm(h)
        h = h + t_emb
        
        for layer in self.layers:
            residual = h
            h = layer_norm(h)
            h = gelu(h @ layer['w1'])
            h = h @ layer['w2']
            h = h + residual
        
        h = layer_norm(h)
        out = h @ self.proj_out
        
        return out.reshape(original_shape).astype(np.float32)
    
    def get_params(self) -> List[np.ndarray]:
        """Get all trainable parameters."""
        params = [self.time_mlp_1, self.time_mlp_2, self.proj_in, self.proj_out]
        for layer in self.layers:
            params.extend([layer['w1'], layer['w2']])
        return params
    
    def set_params(self, params: List[np.ndarray]):
        """Set parameters from list."""
        idx = 0
        self.time_mlp_1 = params[idx]; idx += 1
        self.time_mlp_2 = params[idx]; idx += 1
        self.proj_in = params[idx]; idx += 1
        self.proj_out = params[idx]; idx += 1
        for layer in self.layers:
            layer['w1'] = params[idx]; idx += 1
            layer['w2'] = params[idx]; idx += 1


class GeometricDenoiser(SimpleDenoiser):
    """
    Denoiser with geometric training support.
    
    Uses EigenScript's geometric predicates for adaptive training:
    - Tracks loss trajectory for convergence detection
    - Adjusts learning rate based on curvature
    - Uses Framework Strength for early stopping
    """
    
    def __init__(self, config: Optional[DenoiserConfig] = None):
        super().__init__(config)
        self.loss_history: List[float] = []
        self.gradient_history: List[float] = []
    
    def compute_loss(self, x_noisy: np.ndarray, noise_true: np.ndarray, timestep: int) -> float:
        """Compute MSE loss between predicted and true noise."""
        noise_pred = self.forward(x_noisy, timestep)
        loss = float(np.mean((noise_pred - noise_true) ** 2))
        self.loss_history.append(loss)
        return loss
    
    def get_signature(self) -> str:
        """Get geometric signature of training trajectory."""
        if len(self.loss_history) < 2:
            return "lightlike"
        
        diff = self.loss_history[-1] - self.loss_history[-2]
        threshold = 0.001
        
        if diff < -threshold:
            return "timelike"
        elif diff > threshold:
            return "spacelike"
        else:
            return "lightlike"
    
    def is_converged(self, threshold: float = 0.001) -> bool:
        """Check if training has converged."""
        if len(self.loss_history) < 10:
            return False
        
        recent = self.loss_history[-10:]
        return np.var(recent) < threshold
    
    def framework_strength(self) -> float:
        """Compute Framework Strength of training."""
        if len(self.loss_history) < 2:
            return 0.0
        
        initial = self.loss_history[0]
        current = self.loss_history[-1]
        
        if initial == 0:
            return 1.0
        
        reduction = max(0, (initial - current) / initial)
        
        if len(self.loss_history) >= 5:
            smoothness = 1.0 / (1.0 + np.var(np.diff(self.loss_history[-5:])) * 100)
        else:
            smoothness = 0.5
        
        return min(1.0, reduction * 0.7 + smoothness * 0.3)
    
    def adaptive_lr(self, base_lr: float = 0.001) -> float:
        """Compute adaptive learning rate based on curvature."""
        if len(self.loss_history) < 3:
            return base_lr
        
        d1 = self.loss_history[-1] - self.loss_history[-2]
        d2 = self.loss_history[-2] - self.loss_history[-3]
        curvature = abs(d1 - d2)
        
        if curvature > 0.1:
            return base_lr * 0.5
        elif curvature < 0.01:
            return base_lr * 1.5
        else:
            return base_lr
