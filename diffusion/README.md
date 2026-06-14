# diffusion/ — geometric diffusion image generation PoC

Sibling experiment to `crypto/`. From the source's own docstring:

> Adapts encryption patterns for diffusion-based image generation:
> - `TemporalState` → `DiffusionState` (tracks denoising timesteps)
> - `XORObserver` → `NoiseObserver` (observes noise levels geometrically)
> - Minkowski signatures → convergence detection for image quality
>
> The key insight: In encryption, XOR "hides" data with keystream.
> In diffusion, Gaussian noise "hides" images. Both are reversible
> when you know the schedule.

So `diffusion/` and `crypto/` share primitives. Same temporal-
observer machinery, different application. Currently Python-only,
written against EigenChat's vendored Python-prototype EigenScript.

## Files

| File | Lines | Contents |
|---|---|---|
| `python_reference/core.py` | 261 | `NoiseSchedule`, `DiffusionState`, `NoiseObserver` |
| `python_reference/denoiser.py` | 226 | Denoising step + scheduler logic |
| `python_reference/generator.py` | 348 | High-level image generation loop |
| `python_reference/__init__.py` | 37 | Module exports |

## Why bring this along

If `crypto/`'s primitives turn out to be interesting (or if they
reduce to something boring), the same answer probably applies to
`diffusion/` because both build on the same `XORObserver` /
`TemporalState` shape. Investigating one cheaply covers most of
the other.

## Possible next steps

1. **Read both `crypto/python_reference/core.py` and
   `diffusion/python_reference/core.py` together** — the
   primitives are the same, the difference is what they're
   wrapped around.
2. **Decide as a pair**: do the geometric/observer primitives
   add anything to the underlying construction (XOR keystream
   / Gaussian noise schedule)? If they reduce to standard
   patterns, archive both. If not, port both together to
   current EigenScript.

## Not for production

Same rules as `crypto/`: this is a math toy. Any real use case
graduates to a fresh repo with proper scope.
