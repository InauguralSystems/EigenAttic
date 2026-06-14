"""
Geometric Diffusion Core

Adapts encryption patterns for diffusion-based image generation:
- TemporalState → DiffusionState (tracks denoising timesteps)
- XORObserver → NoiseObserver (observes noise levels geometrically)
- Minkowski signatures → convergence detection for image quality

The key insight: In encryption, XOR "hides" data with keystream.
In diffusion, Gaussian noise "hides" images. Both are reversible
when you know the schedule.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class NoiseSchedule:
    """
    Defines how noise is added/removed over timesteps.
    
    Analogous to how keystream is generated in encryption -
    deterministic given the schedule parameters.
    """
    num_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = "linear"
    
    def __post_init__(self):
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)
    
    def _compute_betas(self) -> np.ndarray:
        """Compute beta schedule (noise variance per step)."""
        if self.schedule_type == "linear":
            return np.linspace(self.beta_start, self.beta_end, self.num_steps)
        elif self.schedule_type == "cosine":
            steps = np.arange(self.num_steps + 1)
            f = np.cos((steps / self.num_steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas = f / f[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            return np.clip(betas, 0.0001, 0.999)
        elif self.schedule_type == "geometric":
            return np.geomspace(self.beta_start, self.beta_end, self.num_steps)
        else:
            return np.linspace(self.beta_start, self.beta_end, self.num_steps)


@dataclass
class DiffusionState:
    """
    Tracks the temporal evolution of the diffusion process.
    
    Adapted from TemporalState in encryption - tracks denoising
    progress instead of XOR iterations.
    """
    timestep: int = 0
    noise_history: List[float] = field(default_factory=list)
    image_history: List[np.ndarray] = field(default_factory=list)
    max_history: int = 50
    
    def advance(self, noise_level: float, image: Optional[np.ndarray] = None) -> int:
        """Record observation and advance timestep."""
        self.timestep += 1
        self.noise_history.append(noise_level)
        
        if image is not None and len(self.image_history) < self.max_history:
            self.image_history.append(image.copy())
        
        if len(self.noise_history) > self.max_history:
            self.noise_history = self.noise_history[-self.max_history:]
        
        return self.timestep
    
    def get_signature(self) -> str:
        """
        Compute Minkowski-style signature of diffusion state.
        
        Adapted from encryption signatures:
        - timelike: noise decreasing (denoising working)
        - spacelike: noise increasing (going wrong direction)
        - lightlike: converged (noise stable)
        """
        if len(self.noise_history) < 2:
            return "lightlike"
        
        diff = self.noise_history[-1] - self.noise_history[-2]
        threshold = 0.001
        
        if diff < -threshold:
            return "timelike"
        elif diff > threshold:
            return "spacelike"
        else:
            return "lightlike"
    
    def is_converged(self, threshold: float = 0.01) -> bool:
        """Check if diffusion has converged (like .converged() predicate)."""
        if len(self.noise_history) < 5:
            return False
        
        recent = self.noise_history[-5:]
        variance = np.var(recent)
        return variance < threshold and self.get_signature() == "lightlike"
    
    def framework_strength(self) -> float:
        """
        Compute Framework Strength for diffusion (from geometric training).
        
        Measures how well the denoising is progressing.
        FS > 0.9 indicates good convergence.
        """
        if len(self.noise_history) < 2:
            return 0.0
        
        start_noise = self.noise_history[0]
        current_noise = self.noise_history[-1]
        
        if start_noise == 0:
            return 1.0
        
        reduction = (start_noise - current_noise) / start_noise
        smoothness = 1.0 / (1.0 + np.var(np.diff(self.noise_history)))
        
        return np.clip(reduction * smoothness, 0.0, 1.0)


class GeometricDiffusion:
    """
    Main diffusion engine using EigenScript's geometric semantics.
    
    Combines:
    - Noise schedule (like keystream in encryption)
    - Diffusion state (like temporal state in encryption)
    - Geometric signatures for quality monitoring
    """
    
    def __init__(
        self,
        image_size: int = 32,
        channels: int = 1,
        schedule: Optional[NoiseSchedule] = None
    ):
        self.image_size = image_size
        self.channels = channels
        self.schedule = schedule or NoiseSchedule()
        self.state = DiffusionState()
        self.shape = (channels, image_size, image_size)
    
    def add_noise(
        self,
        x: np.ndarray,
        timestep: int,
        noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: add noise at timestep t.
        
        Analogous to XOR with keystream in encryption.
        
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = np.random.randn(*x.shape).astype(np.float32)
        
        sqrt_alpha = self.schedule.sqrt_alpha_cumprod[timestep]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alpha_cumprod[timestep]
        
        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        
        return noisy.astype(np.float32), noise.astype(np.float32)
    
    def remove_noise(
        self,
        x_t: np.ndarray,
        predicted_noise: np.ndarray,
        timestep: int
    ) -> np.ndarray:
        """
        Reverse diffusion: remove noise using predicted noise.
        
        Analogous to decrypt XOR in encryption.
        """
        alpha = self.schedule.alphas[timestep]
        alpha_cumprod = self.schedule.alpha_cumprod[timestep]
        beta = self.schedule.betas[timestep]
        
        sqrt_alpha = np.sqrt(alpha)
        sqrt_one_minus_alpha_cumprod = self.schedule.sqrt_one_minus_alpha_cumprod[timestep]
        
        x_prev = (x_t - (beta / sqrt_one_minus_alpha_cumprod) * predicted_noise) / sqrt_alpha
        
        if timestep > 0:
            noise = np.random.randn(*x_t.shape).astype(np.float32) * np.sqrt(beta) * 0.5
            x_prev = x_prev + noise
        
        return x_prev.astype(np.float32)
    
    def sample(
        self,
        denoiser: Callable[[np.ndarray, int], np.ndarray],
        conditioning: Optional[np.ndarray] = None,
        initial_noise: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, str], None]] = None
    ) -> Tuple[np.ndarray, DiffusionState]:
        """
        Generate an image using the reverse diffusion process.
        
        Uses geometric predicates for convergence monitoring.
        
        Args:
            denoiser: Function that predicts noise given (x_t, timestep)
            conditioning: Optional conditioning signal (e.g., text embedding)
            initial_noise: Starting noise (or generate random)
            callback: Optional callback(timestep, image, signature)
        
        Returns:
            (generated_image, diffusion_state)
        """
        self.state = DiffusionState()
        
        if initial_noise is None:
            x = np.random.randn(*self.shape).astype(np.float32)
        else:
            x = initial_noise.astype(np.float32)
        
        for t in reversed(range(self.schedule.num_steps)):
            predicted_noise = denoiser(x, t, conditioning) if conditioning is not None else denoiser(x, t)
            
            x = self.remove_noise(x, predicted_noise, t)
            
            noise_level = float(np.mean(np.abs(predicted_noise)))
            self.state.advance(noise_level, x if t % 10 == 0 else None)
            
            signature = self.state.get_signature()
            
            if callback:
                callback(t, x, signature)
            
            if signature == "spacelike" and t < self.schedule.num_steps // 2:
                print(f"Warning: Diverging at step {t}, signature={signature}")
            
            if self.state.is_converged() and t < self.schedule.num_steps // 4:
                print(f"Early convergence detected at step {t}")
                break
        
        x = np.clip(x, -1.0, 1.0)
        
        return x, self.state
    
    def get_noise_level(self, x: np.ndarray) -> float:
        """Estimate noise level in image (like measuring encryption strength)."""
        dx = np.abs(np.diff(x, axis=-1))
        dy = np.abs(np.diff(x, axis=-2))
        return float(np.mean(dx) + np.mean(dy))
