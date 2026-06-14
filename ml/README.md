# ml/ — pure-EigenScript ML library (v0.8.x, needs port)

~9.6K lines of `.eigs` covering attention, transformer blocks,
optimizers, and convolutional/recurrent layers. All written
against EigenScript v0.8.x — **none of it runs against v0.14.2
as-is**.

## What's here

| Module | Lines | Contents |
|---|---|---|
| `core/llm.eigs` | 1162 | RMSNorm, SwiGLU, KVCache, GQA, CausalLM |
| `core/attention.eigs` | 916 | ScaledDotProductAttention, MultiHeadAttention, PositionalEncoding, TransformerEncoder/Decoder |
| `core/generation.eigs` | 1259 | Sampling, beam search, decoding helpers |
| `core/advanced.eigs` | 1361 | Higher-level composed blocks |
| `core/cnn.eigs` | 748 | Conv layers, pooling, LayerNorm |
| `core/rnn.eigs` | 832 | LSTM/GRU cells |
| `core/positional.eigs` | 579 | RoPE (rotary positional embeddings) |
| `core/layers.eigs` | 400 | Linear, Dropout, Sequential |
| `core/model.eigs` | 268 | Base Model class |
| `training/optimizers.eigs` | 762 | SGD, Adam, AdamW, RAdam + LR schedulers |
| `training/trainer.eigs` | 553 | Training loop |
| `training/loss.eigs` | 474 | Loss functions |
| `utils/checkpoint.eigs` | — | Save/load |
| `utils/diagnostics.eigs` | — | Training diagnostics |
| `data/loader.eigs` | — | Data loading utilities |
| `examples/` | — | One-file demos per module (also v0.8.x) |

## Port-forward checklist

The shape of the v0.8.x → v0.14.2 work, roughly:

1. **`define f as: arg / arg[0]` → named params.** v0.13.0+
   supports real parameter lists and defaults.
2. **`extends Model` / `register_module` class patterns** don't
   exist in current EigenScript. Pick: dict-as-state (matches
   iLambdaAi / current `lib/`) or wait for a real class system.
3. **`Tensor` references** — verify against current tensor
   builtins; the surface has moved.
4. **`from X import Y`** — current EigenScript has a real package
   system (`pkg.eigs` in stdlib); update import shape accordingly.

## Don't port what's already live

iLambdaAi's transformer is already JIT-optimized against v0.12.0+
builtins. Don't re-port attention/softmax/embed if iLambdaAi
already has them faster. The honest port candidates are pieces
iLambdaAi doesn't have:

- **Optimizers** (AdamW, RAdam) — iLambdaAi uses `native_train_step`
  builtins, so a pure-`.eigs` optimizer library could become
  `lib/optim.eigs` if it's worth it.
- **RoPE** as a reusable module — iLambdaAi inlines this.
- **KVCache** as a reusable shape — same.

## Reference

`SYNTAX_REFERENCE_v0.8.x.md` is the original syntax guide from
EigenChat's vendored prototype. Useful for reading the source as
written, not as a target for new work.
