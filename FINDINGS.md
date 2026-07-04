# Findings: crypto + diffusion primitives, investigated as a pair

**Top-line:** Both piles are a **standard construction wrapped in a
cosmetic "Minkowski signature" overlay that doesn't affect the
output**. The geometric layer is monitoring, not math.

> ### ⚠️ SECURITY: the v2 crypto is BROKEN (two-time pad). Do not port it.
>
> Beyond the "decorative geometry" finding below, the v2 cipher has a
> **fatal confidentiality flaw**: the keystream carries no IV and no
> nonce, so every message encrypted under a given key reuses the
> **identical** keystream. Encrypting two messages under one key leaks
> `P_a XOR P_b` to anyone holding the (cleartext) ciphertexts and IVs —
> a classic two-time pad. With a crib, one plaintext fully recovers the
> other. Reproduce it: `python3 crypto/two_time_pad_break.py`.
>
> **Root cause** (`crypto/python_reference/core.py`): `cbc_encrypt_blocks`
> calls `KeystreamGenerator(enc_key, 0)`. The counter always starts at 0
> and `enc_key` is fixed per key, so `KS_i = sha256(enc_key ‖ i)` is
> message-independent. For any block, `C_i XOR C_{i-1} = P_i XOR KS_i`;
> XOR two messages' ciphertexts and the keystream *and* the CBC chain
> cancel. The IV only perturbs the chain — it never reaches the PRF.
>
> **The gut-punch:** the spec (`crypto/GEOMETRIC_ENCRYPTION_SPEC.md`
> §3.2) *specified* the fix — "the iteration counter becomes a nonce" —
> and the code dropped it. `temporal_offset` is defined, written into
> the header, and then **ignored** in favor of a hard-coded `0`.
>
> **Why it passed testing:** round-trip, "same plaintext → different
> ciphertext" (§7.2), and the avalanche test (§9.2) all pass — a fresh
> IV makes the ciphertexts *look* different via CBC cascade. None of
> those tests XORs two different messages together, which is the only
> thing that exposes the leak. Every green test hides the break.
>
> **What IS correct:** integrity is textbook — HMAC-SHA256,
> Encrypt-then-MAC, constant-time compare, MAC over header+IV+ciphertext,
> verified before unpad (no v2 padding oracle). PBKDF2 stretching and
> enc/mac key separation are sound in shape. The flaw is narrow and
> confined to keystream construction — but it's fatal.
>
> **If EigenOS ever wants encryption-at-rest:** don't lift this. The
> honest primitive is `sha256(enc_key ‖ IV ‖ counter)` CTR + the
> (already-correct) HMAC EtM — ~15 lines over the SHA-256/HMAC builtins
> the kernel already ships, without this booby trap. See
> [[project_eigen_os]] for the buffer-input gap in the hash builtins
> that would need closing first.

The two piles share the exact same pattern:

| Pile | Real construction underneath | Decorative geometric overlay |
|---|---|---|
| `crypto/` (v2) | SHA-256 keystream + CBC chain + Encrypt-then-MAC | `XORObserver` / `TemporalState` — **not used by v2 path at all** |
| `crypto/` (v1) | XOR stream cipher with SHA-256 keystream | `XORObserver` returns a signature label; **the label is computed but ignored** |
| `diffusion/` | Standard DDPM (Ho et al. 2020) | `DiffusionState.get_signature()` + `framework_strength()` — print warnings, don't change sampling |

---

## Crypto: what's actually under the hood

### v2 (the path `encrypt()`/`decrypt()` actually take)

Walking `cbc_encrypt_blocks`:

```python
# per block:
chained         = plaintext_block XOR prev_ciphertext_block   # CBC
block_keystream = sha256(seed || counter)                     # CTR-style PRF
encrypted_block = chained XOR block_keystream
```

Plus:
- Key derivation: PBKDF2-HMAC-SHA256, 100k iters
- Key separation: `enc_key = sha256(seed || "encryption")`, `mac_key = sha256(seed || "authentication")` — **note: this is NOT real HKDF**, just hash-then-label, but it's a reasonable approximation
- Authentication: HMAC-SHA256 over `header || IV || ciphertext` (Encrypt-then-MAC, correct ordering)
- Padding: PKCS7

**This is a homemade authenticated encryption scheme using SHA-256
as a PRF**. The construction is unusual: it uses both a CTR-style
keystream AND a CBC chain at the same time. Either alone would
suffice; doing both makes the security argument muddier rather
than stronger. (If the keystream is the cipher, CBC adds nothing
for confidentiality. If CBC is the cipher, you need a real block
cipher — SHA-256-as-block-cipher isn't one.)

**The geometric stuff:**
- `EigenKey.observer_config` (16 bytes from PBKDF2) — derived but **never read** in v2
- `EigenKey.temporal_offset` — written to header in plaintext, **never used** in v2 (`KeystreamGenerator(enc_key, 0)` ignores it)
- `XORObserver`, `TemporalState`, `apply_xor_observation` — **only imported by `_decrypt_v1`**

### v1 (legacy)

```python
keystream = sha256_counter_mode(seed, initial_iteration, length)
plaintext_padded, signatures = apply_xor_observation(cipher_data, keystream, observer)
```

Which expands to:
```python
plaintext_padded = cipher_data XOR keystream   # a stream cipher
signatures       = [sign(history[i] - history[i-1]) for i in ...]
                   # "timelike" / "spacelike" / "lightlike" labels
                   # COMPUTED BUT NEVER CONSUMED
```

The signature labels are returned from `apply_xor_observation` and **immediately discarded** in `_decrypt_v1`. They don't gate decryption, don't authenticate, don't influence the keystream. They are **decorative state**.

### What the geometric layer would need to do to be load-bearing

For "geometric encryption" to be a real cipher concept rather than a label on a regular stream cipher, the observer state would need to:

1. Influence the keystream (e.g., the signature reseeds or perturbs the next keystream block), AND
2. Be reproducible by both ends, AND
3. Add a security property the underlying stream cipher doesn't already have.

The code does **none of these**. The observer is read-only against the keystream.

---

## Diffusion: what's actually under the hood

### The math

`add_noise`:
```python
x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
```

That's **Eq. (4) of Ho et al. 2020** (Denoising Diffusion Probabilistic Models), verbatim.

`remove_noise`:
```python
x_prev = (x_t - (beta[t] / sqrt(1 - alpha_cumprod[t])) * predicted_noise) / sqrt(alpha[t])
         + sqrt(beta[t]) * 0.5 * random_noise   # if t > 0
```

That's the **standard DDPM reverse sampler** (with a `0.5` variance scale that's slightly non-standard but harmless).

Noise schedule supports linear / cosine / geometric betas — also standard DDPM territory (cosine is from Nichol & Dhariwal 2021).

Denoiser: a tiny MLP with sinusoidal timestep embedding (same shape every DDPM tutorial uses).

### The "geometric" additions

```python
def get_signature(self) -> str:
    diff = self.noise_history[-1] - self.noise_history[-2]
    if diff < -threshold:  return "timelike"
    elif diff > threshold: return "spacelike"
    else:                  return "lightlike"
```

That is **literally `sign(discrete derivative)` with three relativity-flavored labels**.

```python
def framework_strength(self) -> float:
    reduction  = (start_noise - current_noise) / start_noise
    smoothness = 1.0 / (1.0 + np.var(np.diff(self.noise_history)))
    return clip(reduction * smoothness, 0.0, 1.0)
```

That is a **normalized "how much progress × how stable" loss-curve metric**. Useful as a training dashboard reading. Not part of the diffusion math.

In `sample()`, these are used to:
- Print a warning when `signature == "spacelike"` mid-process
- Break out of the loop early when `is_converged()` fires

**Neither hook changes the predicted noise, the schedule, or the sampling step.** They're monitoring overlays.

---

## The shared pattern

Both piles instantiate the same template:

> **Real construction** (a known cipher / a known generative model)
> wrapped in a **parallel observer layer** that:
> 1. Records a history list of scalars
> 2. Computes a sign-of-derivative label and calls it "Minkowski signature"
> 3. Reports a normalized progress metric ("Framework Strength")
> 4. Sometimes prints a warning or short-circuits a loop
>
> The observer layer is **never read by the construction it wraps**.

That's it. That's the whole geometric thing in both piles. It's
instrumentation phrased in relativity vocabulary.

The fingerprint is clearer in crypto: the v1 → v2 evolution
**dropped the geometric layer entirely** when authentication and
CBC were added. If the geometry had been load-bearing, v2 would
preserve it. Instead v2 imports the observer module **only** for
backward-compat decrypt.

---

## Verdict

- **Crypto reduces to**: homemade Encrypt-then-MAC using SHA-256
  as a PRF, with a non-functional decorative layer. The
  underlying construction is unusual (CTR+CBC) but defensible if
  reviewed by an actual cryptographer; nothing about it requires
  "geometry."
- **Diffusion reduces to**: textbook DDPM with a loss-curve
  dashboard. The underlying math is fine because it's just DDPM;
  nothing about it requires "geometry."

**Neither is "novel cryptography" or "novel diffusion."** Both
are competent reimplementations of standard constructions with a
poetic monitoring overlay.

---

## What to do about it

**Recommendation: archive both, don't port.**

Reasons:
1. The geometric layer doesn't survive contact with making either
   construction "real" — crypto already dropped it in v2.
2. Porting either to current EigenScript means porting either
   (a) a homemade AEAD that nobody should use over `lib/auth.eigs`
   if/when that gains real primitives, or (b) DDPM, which is well
   covered by NumPy/PyTorch tutorials and doesn't need a from-
   scratch EigenScript port to teach anything.
3. The "geometric signatures" pattern (sign-of-derivative labels
   + framework-strength progress metric) **does have value** —
   but as an instrumentation idiom, not as math. iLambdaAi
   already has its own convergence/Framework-Strength notions;
   if either is missing a loss-curve dashboard primitive, lift
   the *idiom* (a 20-line `lib/geometric_monitor.eigs` or
   similar), not these 1500 lines.

**If we keep anything:** the `framework_strength()` formula
(`reduction × smoothness`, with smoothness = `1 / (1 + var(diffs))`)
is a nice compact loss-curve quality metric and would slot into
iLambdaAi's training telemetry cleanly. That's a 5-line lift,
worth pulling out as a standalone snippet.

**If we keep nothing:** the contents stay in EigenAttic as a
historical record of "what the LRVM/geometric framing looked
like when applied to real constructions." That's still useful
context.
