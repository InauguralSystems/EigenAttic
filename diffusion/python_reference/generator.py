"""
Image Generator Interface

High-level interface for generating images using geometric diffusion.
Provides simple API for image generation and training.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import json
import os

from eigenscript.diffusion.core import GeometricDiffusion, NoiseSchedule, DiffusionState
from eigenscript.diffusion.denoiser import SimpleDenoiser, GeometricDenoiser, DenoiserConfig


@dataclass
class GeneratorConfig:
    """Configuration for image generator."""
    image_size: int = 32
    channels: int = 1
    hidden_dim: int = 64
    num_layers: int = 3
    num_diffusion_steps: int = 50
    schedule_type: str = "linear"


class EigenImageGenerator:
    """
    High-level image generator using EigenScript's geometric diffusion.
    
    This is the main interface for generating images, combining:
    - Geometric diffusion process (adapted from encryption)
    - Denoiser network
    - Training loop with geometric convergence
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        
        self.schedule = NoiseSchedule(
            num_steps=self.config.num_diffusion_steps,
            schedule_type=self.config.schedule_type
        )
        
        self.diffusion = GeometricDiffusion(
            image_size=self.config.image_size,
            channels=self.config.channels,
            schedule=self.schedule
        )
        
        denoiser_config = DenoiserConfig(
            image_size=self.config.image_size,
            channels=self.config.channels,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        )
        self.denoiser = GeometricDenoiser(denoiser_config)
        
        self.training_step = 0
    
    def generate(
        self,
        seed: Optional[int] = None,
        conditioning: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, str], None]] = None
    ) -> Tuple[np.ndarray, DiffusionState]:
        """
        Generate an image.
        
        Args:
            seed: Random seed for reproducibility
            conditioning: Optional conditioning embedding (e.g., from text)
            callback: Optional callback(timestep, image, signature)
        
        Returns:
            (generated_image, diffusion_state)
        """
        if seed is not None:
            np.random.seed(seed)
        
        image, state = self.diffusion.sample(
            denoiser=self.denoiser,
            conditioning=conditioning,
            callback=callback
        )
        
        return image, state
    
    def generate_batch(
        self,
        batch_size: int = 4,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate multiple images."""
        if seed is not None:
            np.random.seed(seed)
        
        images = []
        for i in range(batch_size):
            img, _ = self.generate()
            images.append(img)
        
        return images
    
    def train_step(
        self,
        images: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> float:
        """
        Single training step using SPSA gradient estimation.
        
        Uses Simultaneous Perturbation Stochastic Approximation (SPSA)
        which is more efficient than per-parameter finite differences.
        
        Args:
            images: Batch of training images (batch, channels, height, width)
                    or single image (channels, height, width)
            learning_rate: Learning rate (or use adaptive)
        
        Returns:
            Loss value
        """
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        batch_size = images.shape[0]
        total_loss = 0.0
        
        lr = learning_rate or self.denoiser.adaptive_lr()
        eps = 0.01
        
        original_params = [p.copy() for p in self.denoiser.get_params()]
        accumulated_grads = [np.zeros_like(p) for p in original_params]
        
        for i in range(batch_size):
            x = images[i]
            t = np.random.randint(0, self.schedule.num_steps)
            noise = np.random.randn(*x.shape).astype(np.float32)
            x_noisy, _ = self.diffusion.add_noise(x, t, noise)
            
            self.denoiser.set_params([p.copy() for p in original_params])
            base_loss = self._compute_mse(x_noisy, noise, t)
            total_loss += base_loss
            
            perturbations = [np.random.choice([-1, 1], size=p.shape).astype(np.float32) for p in original_params]
            
            plus_params = [p + eps * delta for p, delta in zip(original_params, perturbations)]
            self.denoiser.set_params(plus_params)
            loss_plus = self._compute_mse(x_noisy, noise, t)
            
            minus_params = [p - eps * delta for p, delta in zip(original_params, perturbations)]
            self.denoiser.set_params(minus_params)
            loss_minus = self._compute_mse(x_noisy, noise, t)
            
            grad_scale = (loss_plus - loss_minus) / (2 * eps)
            for j, delta in enumerate(perturbations):
                accumulated_grads[j] += grad_scale * delta / batch_size
        
        updated_params = [p - lr * g for p, g in zip(original_params, accumulated_grads)]
        self.denoiser.set_params(updated_params)
        
        self.training_step += 1
        return total_loss / batch_size
    
    def _compute_mse(self, x_noisy: np.ndarray, noise_true: np.ndarray, timestep: int) -> float:
        """Compute MSE loss between predicted and true noise."""
        noise_pred = self.denoiser.forward(x_noisy, timestep)
        return float(np.mean((noise_pred - noise_true) ** 2))
    
    def train(
        self,
        images: np.ndarray,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        callback: Optional[Callable[[int, float, str], None]] = None
    ) -> List[float]:
        """
        Train the generator on a dataset.
        
        Uses geometric training with:
        - Adaptive learning rate based on curvature
        - Framework Strength early stopping
        - Signature-based strategy switching
        
        Args:
            images: Training images
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Base learning rate
            callback: Optional callback(epoch, loss, signature)
        
        Returns:
            Loss history
        """
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        n_samples = images.shape[0]
        losses = []
        
        print(f"Training on {n_samples} images for {epochs} epochs...")
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = images[batch_indices]
                
                lr = self.denoiser.adaptive_lr(learning_rate)
                loss = self.train_step(batch, lr)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            
            signature = self.denoiser.get_signature()
            fs = self.denoiser.framework_strength()
            
            if callback:
                callback(epoch, avg_loss, signature)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.4f}, sig={signature}, FS={fs:.3f}")
            
            if fs > 0.95 and epoch > 10:
                print(f"Early stopping at epoch {epoch} (FS={fs:.3f})")
                break
            
            if signature == "spacelike" and epoch > 20:
                print(f"Warning: Diverging at epoch {epoch}, reducing learning rate")
                learning_rate *= 0.5
        
        return losses
    
    def save(self, filepath: str):
        """Save generator state to file."""
        params = self.denoiser.get_params()
        
        state = {
            "config": {
                "image_size": self.config.image_size,
                "channels": self.config.channels,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_diffusion_steps": self.config.num_diffusion_steps,
                "schedule_type": self.config.schedule_type,
            },
            "training_step": self.training_step,
            "loss_history": self.denoiser.loss_history[-100:],
        }
        
        np.savez(
            filepath,
            params=np.array(params, dtype=object),
            state=json.dumps(state)
        )
        print(f"Saved generator to {filepath}")
    
    def load(self, filepath: str):
        """Load generator state from file."""
        data = np.load(filepath, allow_pickle=True)
        
        params = list(data["params"])
        self.denoiser.set_params(params)
        
        state = json.loads(str(data["state"]))
        self.training_step = state["training_step"]
        self.denoiser.loss_history = state.get("loss_history", [])
        
        print(f"Loaded generator from {filepath}")


def create_simple_patterns(n_samples: int = 100, size: int = 32) -> np.ndarray:
    """Create simple pattern dataset for testing."""
    images = []
    
    for i in range(n_samples):
        img = np.zeros((1, size, size), dtype=np.float32)
        
        pattern_type = i % 5
        
        if pattern_type == 0:
            cx, cy = size // 2, size // 2
            r = size // 4
            for y in range(size):
                for x in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                        img[0, y, x] = 1.0
        
        elif pattern_type == 1:
            for j in range(0, size, 4):
                img[0, j:j+2, :] = 1.0
        
        elif pattern_type == 2:
            for j in range(0, size, 4):
                img[0, :, j:j+2] = 1.0
        
        elif pattern_type == 3:
            for y in range(size):
                for x in range(size):
                    if (x + y) % 8 < 4:
                        img[0, y, x] = 1.0
        
        else:
            x1, y1 = size // 4, size // 4
            x2, y2 = 3 * size // 4, 3 * size // 4
            img[0, y1:y2, x1:x2] = 1.0
        
        img = img * 2 - 1
        images.append(img)
    
    return np.array(images, dtype=np.float32)


def image_to_ascii(img: np.ndarray, width: int = 40) -> str:
    """Convert image to ASCII art for terminal display."""
    if len(img.shape) == 3:
        img = img[0]
    
    h, w = img.shape
    aspect = h / w
    
    height = int(width * aspect * 0.5)
    
    chars = " .:-=+*#%@"
    
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    result = []
    for y in range(height):
        row = ""
        for x in range(width):
            src_y = int(y * h / height)
            src_x = int(x * w / width)
            val = img_norm[src_y, src_x]
            char_idx = int(val * (len(chars) - 1))
            row += chars[char_idx]
        result.append(row)
    
    return "\n".join(result)
