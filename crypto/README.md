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
