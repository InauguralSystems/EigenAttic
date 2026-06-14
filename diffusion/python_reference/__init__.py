"""
EigenScript Geometric Diffusion Module

Adapts the geometric encryption patterns (XOR observer, temporal state,
Minkowski signatures) for diffusion-based image generation.

Key insight: Diffusion models add/remove noise like encryption adds/removes
keystream - both are reversible transformations tracked through time.
"""

from eigenscript.diffusion.core import (
    GeometricDiffusion,
    DiffusionState,
    NoiseSchedule,
)
from eigenscript.diffusion.denoiser import (
    SimpleDenoiser,
    GeometricDenoiser,
)
from eigenscript.diffusion.generator import (
    EigenImageGenerator,
    GeneratorConfig,
    create_simple_patterns,
    image_to_ascii,
)

__all__ = [
    "GeometricDiffusion",
    "DiffusionState",
    "NoiseSchedule",
    "SimpleDenoiser",
    "GeometricDenoiser",
    "EigenImageGenerator",
    "GeneratorConfig",
    "create_simple_patterns",
    "image_to_ascii",
]
