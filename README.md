# EigenAttic

Salvaged work from older EigenScript projects, parked in one place so
none of it gets lost and any of it can be picked up when there's an
itch to scratch.

Two directories, two different missions:

- **`ml/`** — pure-EigenScript ML library code (attention, RoPE,
  RMSNorm, SwiGLU, KVCache, Adam/AdamW, CNN, RNN, LLM blocks).
  Written against EigenScript v0.8.x syntax. The work is **porting
  it forward to v0.14.2+** and deciding what (if anything) graduates
  into `EigenScript/lib/`.
- **`crypto/`** — experimental "geometric encryption" PoC
  (XOR-observer pattern + a 434-line core). Currently Python,
  written against the old EigenScript Python prototype. The work
  is **porting it to current EigenScript and deciding whether the
  idea has legs**.

Neither pile ships as part of the live portfolio. EigenAttic is
explicitly a *sandbox* — pick up what's interesting, leave the
rest alone, no production promise.

## Where it came from

Both piles were extracted from `EigenChat`, an older Replit-era
Flask chatbot product on InauguralPhysicist. The vendored
EigenScript inside EigenChat is the **old Python prototype** of
the language (lexer + LLVM/CUDA codegen + Python evaluator) — a
dead branch relative to the current C interpreter + JIT. Almost
all of that vendored tree is dead-end code; what's in EigenAttic
is the subset judged worth keeping.

## Layout

```
ml/
  core/        attention, llm, positional, cnn, rnn, layers, model, generation, advanced
  training/    optimizers, loss, trainer
  utils/       checkpoint, diagnostics
  data/        loader
  examples/    runnable demos (v0.8.x syntax — won't run as-is)
  SYNTAX_REFERENCE_v0.8.x.md   from-source reference for the dialect everything was written in

crypto/
  python_reference/   the original Python implementation as it came out of EigenChat
```

## Status

- **ML port**: nothing built or working. Files are as-extracted —
  most use `extends Model` / `register_module` / `Tensor` patterns
  that don't match v0.14.2 conventions. Treat as a starting point,
  not a working library.
- **Crypto port**: Python-only, won't run against current
  EigenScript at all. Port-or-discard decision pending.

## License

MIT.
