# crypto/ — geometric encryption PoC

Experimental "geometric encryption" lifted from EigenChat. Tagged
EXPERIMENTAL in the source. **Currently Python-only**, written
against EigenChat's vendored Python prototype of EigenScript.
Nothing runs against current EigenScript.

## Files

| File | Lines | Contents |
|---|---|---|
| `python_reference/core.py` | 434 | LRVM-based encryption/decryption core, key generation |
| `python_reference/xor_observer.py` | 138 | XOR-observer construction |
| `python_reference/__init__.py` | 29 | Module exports |
| `GEOMETRIC_ENCRYPTION_SPEC.md` | 435 | The original v2.1 spec from EigenChat `docs/crypto/` (salvaged 2026-07-03, missed in the first pass) |
| `encryption_example.eigs` | ~60 | Old-dialect demo calling the `eigen_encrypt`/`eigen_decrypt`/`eigen_keygen` builtins the Python prototype exposed. Won't run on current EigenScript. |

## Provenance note (2026-07-03)

Two recall-checked facts, verified against the EigenChat source:

- **"Diffusion" in this scheme is the classical Shannon sense** —
  "CBC mode for block diffusion." There was never a diffusion-
  *process* cipher (plaintext hidden in a noising trajectory).
  The spec's algorithm sections are plain XOR-keystream + CBC;
  the LRVM/trajectory language is framing, not mechanism.
- **The arrow ran crypto → diffusion, not the reverse.** The
  `diffusion/` pile's own docstring: "Adapts encryption patterns
  for diffusion-based image generation" — keystream-hides-data
  generalized to noise-hides-images, both reversible given the
  schedule. The shared insight is real; the DDPM pile is where
  "diffusion-based" literally applies.

## The idea (as best I can tell)

Uses the LRVM (Lightlike Relational Vector Model — the geometric
space EigenScript's `of` operator lives in) to hide plaintext in
a vector trajectory. The `OF` operator's lightlike property
(`‖OF‖² = 0`) is what supposedly makes the construction nontrivial.

The XOR-observer variant is a more conventional XOR-with-PRNG
shape using "observation" as the keystream source.

## The honest question

Whether this is "real geometric encryption" or "XOR with extra
steps wearing a math costume" is the first thing to figure out.
The answer matters before any porting work.

## Possible next steps

1. **Read `core.py` end-to-end** and reduce the construction to
   its primitive operations.
2. **Try to express the same thing as a known cipher** (one-time
   pad? stream cipher? something else?). If it reduces, document
   that and archive.
3. If it doesn't reduce, **port the math to current EigenScript**
   using existing geometric builtins and write a real spec.

## Not for production

If this ever produces something interesting, it does **not** ship
from EigenAttic. Graduate to a real repo with a real threat model,
a real spec, and a real review. Until then, treat it as a math toy.
