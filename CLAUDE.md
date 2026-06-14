## What this is

EigenAttic is a **sandbox repo** holding two piles of salvaged work
from older EigenScript projects:

1. **`ml/`** — pure-EigenScript ML library code (~9.6K lines of
   `.eigs`) written against the v0.8.x dialect. RoPE, RMSNorm,
   SwiGLU, KVCache, attention, Adam/AdamW, CNN/RNN, LLM blocks.
   The work here is porting forward to v0.14.2+.
2. **`crypto/`** — experimental "geometric encryption" PoC. Lives
   under `crypto/python_reference/` as the original Python code
   from EigenChat's vendored Python-prototype EigenScript. The
   work is porting to current EigenScript and deciding if the
   idea has legs.

Unlike EigenGauntlet / EigenRegex / EigenMiniSat / DMG, **EigenAttic
is not a forcing function.** It has no GAPS.md, no test discipline,
no ship promise. Pick up what's interesting, leave the rest.

## Where it came from

Both piles were extracted from `EigenChat` (InauguralPhysicist) — an
older Replit-era Flask chatbot whose vendored EigenScript was the
**Python prototype** of the language (LLVM/CUDA codegen, Python
evaluator). That whole tree is a dead branch relative to current
EigenScript (C interpreter + native JIT). The subset judged worth
keeping is what's in EigenAttic.

## Toolchain

Same v0.13.0 min / v0.14.2 tested pin as the rest of the portfolio,
but **nothing currently runs** — `.eigs` files are v0.8.x syntax,
crypto is Python-only.

```bash
EIGS=${EIGENSCRIPT_BIN:-/home/jon/EigenScript/src/eigenscript}
```

## ML pile — port-forward notes

The v0.8.x → v0.14.2 syntax gap is the main blocker. Specifics to
watch for when porting:

- `extends Model` / `register_module` / `Tensor` class patterns
  don't match how iLambdaAi or current `lib/` modules organize
  state. Most likely path is dict-as-state, not class-extension.
- `arg`, `arg[0]`, multi-arg `arg` patterns are the old `define f
  as:` convention — v0.13.0+ has real named params and default
  args, so port to those.
- The transformer in iLambdaAi already implements
  attention/embed/softmax against current builtins and is JIT-
  optimized (`ids_to_text +33% under v0.12.0 JIT`). Don't blindly
  re-port what's already faster on the live side — only port
  pieces that are genuinely missing (Adam/AdamW, RoPE math,
  KVCache shape).

### Realistic graduation candidates

If anything in `ml/` gets ported and proves itself, the likely
upstream target is **`EigenScript/lib/`** as a new module (e.g.
`lib/optim.eigs`, `lib/positional.eigs`). Don't graduate by
copying — re-derive into the current stdlib's idioms.

## Crypto pile — what it actually is

`crypto/python_reference/core.py` implements a "geometric
encryption" scheme based on the LRVM (Lightlike Relational Vector
Model) — uses the `OF` operator's lightlike norm-zero property to
hide plaintext in a vector trajectory. `xor_observer.py` is a
companion XOR-observer construction.

This was tagged EXPERIMENTAL in EigenChat's `replit.md`. It's a
toy. The work is:

1. Decide whether the LRVM-as-cipher idea is interesting enough to
   port at all (it may just be "XOR with extra steps").
2. If yes, port the core math to current EigenScript using the
   existing geometric builtins.
3. If no, archive it here and move on.

## Hard-won rules

- **No GAPS.md.** This is a sandbox; friction here doesn't get an
  upstream fix unless it surfaces independently elsewhere.
- **No "preserve compatibility with the v0.8.x version" tax.**
  Port forward freely; the old code is a reference, not a contract.
- **Don't pretend it ships.** If you start using something from
  EigenAttic for real, graduate it to a real repo. Don't grow
  this one into a product.

## Current state

Everything is as-extracted. Nothing has been ported, tested, or
modernized. Inventory in the top-level README.
