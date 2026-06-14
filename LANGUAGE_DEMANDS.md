# Language Demands: what real ML code asked of EigenScript v0.8.x

**Read this as evidence, not as a port plan.** The `ml/` pile is
~9.6K lines of non-trivial library code (attention, RoPE, CNN/RNN,
Adam/AdamW, KVCache, LLM blocks) that was actually being written by
someone trying to do ML in EigenScript. Wherever the code looks
ugly, that ugliness is a **demand on the language**: a recurring
workaround for something the language didn't give you.

This file catalogs those demands by category, cites the workaround
in the ml/ pile, and marks each item against current EigenScript
v0.14.2 (`v0.14.2 has it / partial / absent`). At the end, a
ranked recommendation list.

---

## 1. Class system — the biggest single demand

**Workaround pattern**: every "class" is a `define X as:` that opens a
scope, with `# extends Foo` as a comment, and manual list bookkeeping
to fake inheritance and polymorphism.

Citations:
- `core/model.eigs:11` — `define Model as:` with `_parameters is []`,
  `_modules is []`, `_training is true`. Methods: `register_parameter`,
  `register_module`, `parameters` (recursive walk), `train`, `eval`,
  `is_converged`.
- `core/layers.eigs:13` — `define Linear as:` with `# extends Model`
  comment; manually calls `register_parameter of [weight]`,
  `register_parameter of [bias]`.
- `core/attention.eigs` — 44 `register_parameter`/`register_module`
  occurrences; multi-head attention assembles q/k/v/o projections
  via manual registration walk.
- `training/optimizers.eigs:68/140/316` — SGD/Adam/RAdam each
  reimplement state-list bookkeeping by hand because there's no base
  optimizer class to inherit from.

**Total**: 259 occurrences of `register_parameter`/`register_module`/
`# extends` across 17 files. That's the strongest single demand
signal in the pile.

What was actually being asked for:
- A real `class` / `extends` keyword with method-override resolution.
- A `self` binding inside methods (current pattern uses lexical-scope
  capture, which only works for shallow class trees).
- `Parameter` and `Module` as built-in types, not user-rolled lists.
- A `ModuleList` / `ParameterList` for ordered child storage with
  iteration, indexing, and recursive `.parameters()` traversal.

**Status in v0.14.2**: **absent**. Current `lib/` has no `nn.eigs`,
no `Module`, no `Parameter`. The dict-as-state idiom that iLambdaAi
uses is the current workaround and works well enough for one model
but doesn't compose into a library.

---

## 2. Arithmetic verbosity — operator overloading on tensors

**Workaround pattern**: every arithmetic op on tensors is a function
call: `multiply of [a, b]`, `add of [a, b, c]`, `subtract of [a, b]`,
`divide of [a, b]`. No infix operators on tensor values.

Citations: **256 occurrences** of `multiply of [`/`add of [`/
`subtract of [`/`divide of [` across 22 files. Heaviest hits:

- `training/optimizers.eigs:55` — Adam update step:
  ```
  m_new is add of [multiply of [beta1, m], multiply of [one_minus_beta1, grad]]
  v_new is add of [multiply of [beta2, v], multiply of [one_minus_beta2, multiply of [grad, grad]]]
  ```
  Compare to numpy: `m = beta1*m + (1-beta1)*grad`. The pile makes
  numerical code 3-4× longer than the equivalent mathematical
  expression.
- `core/attention.eigs:44` — every scaled-dot-product line is wrapped
  in `multiply of [...]` calls.
- `training/loss.eigs:24` — cross-entropy, MSE, MAE all hand-rolled
  through `of`-form arithmetic.

What was actually being asked for: **infix `+`/`-`/`*`/`/` operators
that dispatch on tensor type** (i.e., operator overloading on the
tensor type, broadcasting included).

**Status in v0.14.2**: **partial**. Scalar arithmetic uses infix.
`lib/tensor.eigs` (127 lines) provides function-form ops, not
infix overloading. iLambdaAi works around this with hot-loop helpers
that hoist globals; it's livable but verbose at scale.

---

## 3. Missing stdlib modules

The pile imports/defines whole subsystems that current `lib/` doesn't
have. Each is a candidate for a real stdlib module:

| Demand | Evidence | v0.14.2 status |
|---|---|---|
| `lib/nn.eigs` (Module base, Linear, activations, Dropout, Sequential, Flatten) | `core/model.eigs` + `core/layers.eigs` (401 lines) | **absent** |
| `lib/optim.eigs` (SGD/Adam/AdamW/RAdam + StepLR/ExpLR) | `training/optimizers.eigs` (5 optimizers, 2 schedulers) | **absent** (existing `lib/optimize.eigs` is gradient-descent helpers, not training optimizers) |
| `lib/loss.eigs` (CrossEntropy, MSE, MAE, BCE) | `training/loss.eigs` | **absent** |
| `lib/dataloader.eigs` (batching, shuffling, iterator protocol) | `data/loader.eigs` | **absent** |
| `lib/checkpoint.eigs` (model save/load, optimizer state I/O) | `utils/checkpoint.eigs` | **absent**; would need `write_pickle`/`read_pickle` or JSON serializer for nested params |
| `lib/embeddings.eigs` (token + positional) | `core/positional.eigs` (RoPE, sinusoidal, learned) | **absent** |
| `lib/init.eigs` (Xavier, Kaiming, normal init) | scattered across `core/*.eigs` | **absent** |

The four highest-signal ones are `nn`, `optim`, `loss`, `dataloader` —
together they're "what someone needs to start training a model in
EigenScript without rewriting PyTorch from scratch."

---

## 4. Tensor introspection and numerical primitives

Demanded by the pile, listing only the ones that surface as friction:

- **Shape/dtype/device queries** — needed everywhere; the pile fakes
  these by tracking shape in parallel dict state alongside the tensor.
- **Random distributions** — `randn`, `uniform`, `bernoulli` used in
  dropout, init, sampling.
- **Reductions** — `sum`, `mean`, `var`, `max`/`argmax` over axis.
- **Gather/scatter** — embedding lookup is `gather of [embed, ids]`.
- **`cumprod`/`cumulative_sum`** — diffusion schedule + softmax cumulative.
- **`topk`/`argsort`** — generation sampler (top-k decoding).
- **`where`-based conditional masking** — `where of [mask, scores, -1e9]`
  for attention masking. (`core/attention.eigs`)
- **In-place slice assignment** — `k_cache[:b, :, t1:t2, :] is k` for
  KVCache writes. The pile uses this freely; it's the only way KV
  caching is tractable.

**Status in v0.14.2**: **partial**. `lib/tensor.eigs` covers some;
gather/scatter/cumprod/topk/in-place slice-assignment are the gaps.

---

## 5. Control-flow and binding ergonomics

Smaller but recurring demands:

- **Varargs splat**: `*tensor_list is arg` in optimizer step methods
  and Sequential forward. (`training/optimizers.eigs`, `core/layers.eigs`)
- **Tuple unpacking on assignment**: `attn_output, attn_weights is forward of [...]`
  (`core/attention.eigs`). Currently single-return; multi-return
  forces dict-wrap.
- **Iterator protocol**: `__iter__`/`__next__` stubs in `data/loader.eigs`
  point at a missing user-definable iterator interface.
- **`enumerate`/`zip`/`reverse of range`** used in training loops.
- **`in` operator on dicts** for "has parameter" checks.
- **First-class function values + lambda**: `sort of [list, key: (fn x: ...)]`
  pattern surfaces in `core/generation.eigs`. Suggests v0.8.x had some
  notion of inline function literals — needs checking against current
  EigenScript's callable model.

**Status in v0.14.2**: **mixed**. Varargs partial, tuple unpack
absent, lambda-style callables unclear, `enumerate`/`zip` absent,
iterator protocol absent.

---

## 6. Geometric predicates — the load-bearing demand that *was* honored

26 of the 30-ish ml/ files use `converged` / `stable` / `improving` /
`oscillating` / `equilibrium` / `diverging` as direct predicates. This
isn't decoration like the crypto/diffusion overlays — it's how
training loops decide to stop, swap optimizers, or back off learning
rates. (`core/model.eigs:` `is_converged`, `training/trainer.eigs`,
many examples in `examples/`.)

**Status in v0.14.2**: **present and live**. iLambdaAi's
`lib/geometric_training.eigs` integrates these against runtime
`observe`/`report` builtins with a six-state classifier. The
v0.8.x pile was using a simpler version of the same idiom; the
language's evolution kept this layer and dropped what was
decorative. This is the clearest example of "the v0.8.x demands
were heard correctly."

---

## 7. Ranked recommendation list

Ordered by **signal strength** in the pile, not by ease of port:

1. **`lib/nn.eigs` — Module/Parameter/Linear/Sequential base classes.**
   259 workaround occurrences. The single largest pain point. Would
   close out item #1 (class system) at the level mattering to ML
   library code, even without a general-purpose `class` keyword in
   the language.
2. **Infix arithmetic overloading on tensors.** 256 workaround
   occurrences. Doesn't need to be general operator overloading —
   restricting it to the tensor builtin type is enough.
3. **`lib/optim.eigs` — SGD/Adam/AdamW + StepLR/ExpLR.** Distinct
   from current `lib/optimize.eigs` (math/gradient descent). The
   v0.8.x pile has 5 optimizers + 2 schedulers fully implemented;
   re-deriving in current idioms is a 1-2 day job.
4. **`lib/loss.eigs` — CrossEntropy/MSE/MAE/BCE.** Small module,
   high reuse; would unblock `lib/nn.eigs` consumers.
5. **In-place tensor slice-assignment.** Specifically demanded by
   KVCache; no clean workaround in the current language.
6. **`lib/dataloader.eigs` + iterator protocol.** Less critical
   than the above because most current consumers (iLambdaAi)
   train on a single fixed corpus; would matter as soon as a
   second model lands.
7. **`lib/checkpoint.eigs`** — needs `write_pickle`/`read_pickle`
   or a JSON-of-tensors codec landing first. Wait on demand.
8. **Tuple unpacking + lambda values.** Quality-of-life; low
   urgency without #1 and #2 first.

Items 1+2 together close >500 of the workaround occurrences in
the pile. That's where the leverage is.

---

## 8. What this pile is *not* evidence for

- **Not evidence that operator overloading should be general.** The
  pile only ever needs it on tensors. A tensor-typed overload is
  enough; full Python-style dunder methods would be over-shooting
  what real code asked for.
- **Not evidence that LRVM / geometric-encryption primitives are
  needed.** Those are in `crypto/` and `diffusion/`, and FINDINGS.md
  already shows they were decorative. The ml/ pile uses the
  *predicate* side of the geometric framework (converged/stable/etc.)
  and never touches the `XORObserver`/`TemporalState` side.
- **Not evidence that v0.8.x should be revived.** The dialect is dead.
  This is a survey of the *demands* the code put on the language; the
  port forward, if it happens, lands in current `lib/`.

---

## How to use this file

If you're considering adding to `EigenScript/lib/`, check this file
first for prior demand. An item with high citation count here is
backed by real ML code that needed it. An item missing here is
either covered elsewhere or hasn't been demanded by working code
yet — both are reasons to defer.

If you're considering porting something *from* `ml/`, the ranked
list above is the order of value-per-day-of-effort. Items 1 and 2
should land together; everything below them is incremental.
