# 32 · Tokenization: Build the GPT Tokenizer (Karpathy 8)

**Phase 4 — NLP and Transformers** · Estimated time: 1 week · Prerequisites: [31 · Attention and the Transformer: Let's Build GPT](31-attention-build-gpt.md), [28 · Language Models 101: makemore](28-language-models-makemore.md)

> Every LLM has a first layer that is rarely discussed: the tokenizer. Before GPT sees a prompt, the text is chopped into "tokens", chunks that are bigger than characters but smaller than words. A surprising number of famous LLM failures (miscounting the r's in "strawberry", weak arithmetic, oddly high costs in Hindi) are really tokenizer failures. This lesson builds the exact algorithm GPT uses, byte-pair encoding, completely from scratch, and then plugs it into the GPT built in lesson 31. Tokenization is an unglamorous, hidden part of understanding LLMs, but building a tokenizer turns one question into a lasting tool for debugging real LLM behavior: "how did that get tokenized?"

## What this lesson builds

- **Project 1 — BPE from scratch**: a working `BPETokenizer` class with `train(text, vocab_size)`, `encode(text)` and `decode(ids)` that round-trips any text perfectly: English, emoji, Hindi, anything.
- **Project 2 — Tokenizer forensics**: a comparison report (`findings.md` and a script) that dissects how the real GPT-2 and GPT-4 tokenizers split tricky strings, with measured tokens-per-character for English vs Hindi.
- **Project 3 — Marry it to the GPT**: the lesson-31 GPT retrained on BPE tokens instead of characters, with a side-by-side comparison of sample quality and effective context length.

## Concepts covered

- Why not characters? Why not words? The BPE compromise between the two.
- The BPE algorithm: repeatedly merge the most frequent adjacent pair into a new token.
- The encode/decode round-trip, and why `decode(encode(x)) == x` is the sacred invariant.
- Byte-level BPE: start from raw bytes (0–255) so that *any* Unicode text is handled with no unknown-token hacks.
- Vocab size trade-offs: a bigger vocab means shorter sequences, but a bigger embedding table and rarer tokens.
- Special tokens: reserved ids, such as an end-of-text marker, that never come from the text itself.
- Inspecting production tokenizers with `tiktoken`.
- Why LLMs are bad at spelling and counting letters: they never see letters.

## Before starting

This lesson needs the working GPT from lesson 31 (Project 3 reuses it) and the tiny Shakespeare `input.txt` from that lesson. The commands below install `tiktoken` (OpenAI's real tokenizer library, for Project 2) and reuse tiny Shakespeare by copying it from the lesson-31 folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install tiktoken
mkdir -p work/32-tokenization-bpe
cp work/31-attention-build-gpt/input.txt work/32-tokenization-bpe/
cd work/32-tokenization-bpe
```

Projects 1 and 2 need no PyTorch and run on any computer. Project 3 retrains the lesson-31 GPT, and current PyTorch has no Intel Mac version, so on an Intel Mac do Project 3 in Google Colab, as in lesson 31.

Watch Karpathy's "Let's build the GPT Tokenizer" from the [Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) alongside Project 1: pause the video, build each piece by hand, then compare. His reference implementation lives at [minbpe](https://github.com/karpathy/minbpe); avoid reading its source until the Project 1 tokenizer works.

## Project 1 — BPE from scratch

**Goal**: Implement byte-pair encoding (train, encode, decode) in plain Python, and prove that the round-trip is perfect on English, emoji and Devanagari text.

**Milestones**

- [ ] **Feel the problem first.** In a Python REPL, take the string `"नमस्ते \U0001F44B"` (that escape is the waving-hand emoji, which Python renders as one character) and run `list(s)` and then `list(s.encode("utf-8"))`. The first gives Unicode characters; the second gives raw bytes (integers 0–255). UTF-8 is the encoding that turns any character on Earth into 1–4 bytes. Checkpoint: the emoji alone becomes 4 bytes, and the Hindi word becomes many more bytes than characters.
- [ ] Write `get_pair_counts(ids)`: given a list of integers, return a dict counting every adjacent pair, e.g. `[1, 2, 2, 1, 2]` → `{(1,2): 2, (2,2): 1, (2,1): 1}`. Checkpoint: that exact example passes.
- [ ] Write `merge(ids, pair, new_id)`: return a new list where every occurrence of `pair` is replaced by the single integer `new_id`. Take care with the loop index: after a merge, skip ahead by 2, not 1. Checkpoint: `merge([1,2,2,1,2], (1,2), 99)` → `[99, 2, 99]`.
- [ ] Write `train(text, vocab_size)`: convert text to a list of bytes, then loop `vocab_size - 256` times: count pairs, find the most frequent one, merge it into the next free id (256, 257, ...), and record the merge in a dict `merges[(a, b)] = new_id`. Print each merge along the way. Checkpoint: on tiny Shakespeare, the first few merges are unsurprising English glue, such as `e` + (space), `t` + `h`, or `t` + (space), because those pairs dominate English text.
- [ ] Build the vocab for decoding: `vocab = {i: bytes([i]) for i in range(256)}`, then for each merge in order, `vocab[new_id] = vocab[a] + vocab[b]`. Each token id now maps to a byte string.
- [ ] Write `decode(ids)`: concatenate the byte strings and call `.decode("utf-8", errors="replace")`. The `errors="replace"` matters: a lone token can end mid-character in a multi-byte sequence, and decode must never crash.
- [ ] Write `encode(text)`: convert to bytes, then repeatedly find the pair in the sequence that was merged **earliest during training** (lowest new_id) and apply it, until no known pairs remain. Order matters: merges must replay in training order.
- [ ] **The sacred invariant.** Test `decode(encode(x)) == x` on: a paragraph of Shakespeare, `"नमस्ते दुनिया \U0001F44B\U0001F680"` (Hindi plus two emoji, via escapes), a string of Python code, and the empty string. Checkpoint: all four pass, exactly.
- [ ] Wrap it in a `BPETokenizer` class with `save(path)`/`load(path)` (write the merges to a file; JSON keys must be strings, so store pairs like `"101,32"`). Checkpoint: after saving and loading into a fresh object, the round-trip test still passes.
- [ ] Measure compression: train with `vocab_size=512` on tiny Shakespeare and print `len(text.encode("utf-8")) / len(encode(text))` (bytes per token). Checkpoint: roughly 2 bytes per token (anywhere from 1.7 to 2.5 is normal); with `vocab_size=256` it is exactly 1.0.
- [ ] Add one special token: reserve id `vocab_size` for `<|endoftext|>`, a marker meaning "a document ends here". It is *never produced by merges*; it is inserted deliberately between documents. Write `encode_with_special(text)` that splits on the literal string and inserts the id. Checkpoint: encoding `"hi<|endoftext|>yo"` yields the special id exactly once, in the middle.

<details><summary>Hints</summary>

- `max(counts, key=counts.get)` finds the most frequent pair in one line.
- In `encode`, loop only while the sequence has at least 2 ids (`while len(ids) >= 2:`); an empty or one-token sequence has no pairs, and `min()` of an empty dict raises a `ValueError`. At each step compute the pair counts of the current sequence, then pick `min(pairs, key=lambda p: merges.get(p, float("inf")))`, the pair with the lowest merge id. If its id is `inf`, nothing more can merge: stop.
- If the round-trip breaks only on emoji/Hindi, the bug is almost always decoding token-by-token instead of concatenating **all** bytes first and decoding once at the end.
- `bytes([i])` makes a 1-byte bytes object; `b"ab" + b"cd"` concatenates. Keep everything as `bytes` until the final decode.

</details>

**Definition of done**: `train`, `encode`, `decode`, `save` and `load` all work; the round-trip test passes on English, code, emoji and Devanagari; and the reason byte-level BPE never needs an "unknown token" can be explained out loud.

## Project 2 — Tokenizer forensics

**Goal**: Use `tiktoken` to autopsy the real GPT-2 and GPT-4 tokenizers on tricky inputs, and write up five concrete findings about how tokenization shapes LLM behavior and cost.

**Milestones**

- [ ] Load both tokenizers and write a helper `show(enc, text)` that prints each token id next to its decoded text chunk:

  ```python
  import tiktoken
  gpt2 = tiktoken.get_encoding("gpt2")        # GPT-2's tokenizer, ~50k vocab
  gpt4 = tiktoken.get_encoding("cl100k_base") # GPT-4's tokenizer, ~100k vocab
  ids = gpt2.encode("hello world")
  print([(i, gpt2.decode([i])) for i in ids])
  ```

- [ ] Tokenize `"strawberry"` and `" strawberry"` (note the leading space) with both. Checkpoint: at least one version splits into multiple tokens, and none of the chunks is a single letter. Write down why "how many r's in strawberry" is therefore hard for an LLM: it sees token ids like `[496, 675, 15717]`, never the letters r-a-w.
- [ ] Tokenize numbers: `"127"`, `"677"`, `"12345678"`, `"3.14159"`. Note how inconsistently digits are grouped (sometimes 3 digits per token, sometimes 1 or 2). This is a big reason why LLM arithmetic is shaky: the model must memorize math over arbitrary digit chunks.
- [ ] Tokenize a URL, a snippet of Python (try one with 8 leading spaces of indentation), and `"  lots   of   spaces"`. Checkpoint: GPT-4's tokenizer handles runs of whitespace in visibly fewer tokens than GPT-2's. This was a deliberate fix that made GPT-4 much better at Python.
- [ ] Measure cost by language: take an English paragraph and a Hindi paragraph of similar meaning (write or translate a few sentences), and compute `len(enc.encode(text)) / len(text)` (tokens per character) for both languages on both tokenizers. Checkpoint: Hindi costs several times more tokens per character than English on GPT-2, and is still clearly worse on GPT-4. Since API pricing and context windows are per token, the same meaning literally costs more and fits less in non-English languages.
- [ ] Cross-check a few strings against the [OpenAI tokenizer playground](https://platform.openai.com/tokenizer) in a browser to confirm the numbers.
- [ ] Run the Project-1 tokenizer on the same tricky strings and note where a 512-token vocab trained on Shakespeare falls apart (hint: everywhere that is not Elizabethan English).
- [ ] Write `findings.md` with the 5 best findings, each one sentence of observation plus one sentence of consequence ("GPT-2 tokenizes each space in indented code separately → Python was expensive and error-prone for it").

<details><summary>Hints</summary>

- Compare with and without a leading space on every probe word. BPE merges include the space, so `"hello"` and `" hello"` are usually completely different tokens.
- For the tokens-per-character measurement, use `len(text)` (characters), not `len(text.encode())` (bytes), so the comparison is about what a human reads.
- If Devanagari prints oddly in the terminal, do not fight it: work with the counts, because the numbers tell the story.

</details>

**Definition of done**: `findings.md` exists with 5 evidence-backed findings, including the strawberry/letter-counting explanation and the English-vs-Hindi cost measurement.

## Project 3 — Marry it to the GPT

**Goal**: Swap the character-level tokenizer in the lesson-31 GPT for the BPE tokenizer from Project 1, and see first-hand what a real tokenizer buys: longer effective context and more coherent samples.

**Milestones**

- [ ] Copy the lesson-31 GPT training script into this work folder. Train the `BPETokenizer` on tiny Shakespeare with `vocab_size` around 1024, and save it.
- [ ] Encode the full dataset once with this tokenizer and save the ids (a plain list, or a NumPy array via `np.array(ids, dtype=np.uint16)`; `uint16` is fine since vocab < 65536). Checkpoint: the dataset shrinks from ~1.1M character tokens to roughly 400–650k BPE tokens.
- [ ] Point the GPT at the new data: set `vocab_size=1024` (plus any special tokens in use) in the model config so that the embedding table and final layer resize, and feed it the BPE ids. Everything else (attention, blocks, training loop) is untouched. This is a key property of the design: the model never knows what a token *means*.
- [ ] Retrain with the same `block_size` (context length in tokens) as lesson 31. Checkpoint: the loss is *higher* than the character model's (maybe between 4 and 5, compared with about 1.5 to 1.8 for the lesson-31 character model), and that is expected. With a 1024-token vocab, random guessing starts at `ln(1024) ≈ 6.9` vs `ln(65) ≈ 4.2` for characters, so the numbers live on different scales. Per-token losses across different vocabularies are not comparable.
- [ ] Sample from the trained model, decoding the generated ids with the **BPE** tokenizer. Checkpoint: no mid-character mojibake (garbled symbols), and the samples read at least as Shakespeare-ish as the character model's, often with better long-range structure. This is because each of the `block_size` context slots now holds ~2.5 characters' worth of text, so the model effectively sees about two and a half times as far back as the character model did at the same `block_size`.
- [ ] Do a fair comparison: generate ~500 characters (not tokens) from both models and put them side by side in a short `comparison.md`. Note sample quality, and compute each model's effective context in *characters* (`block_size × bytes-per-token`).
- [ ] Optional: try `vocab_size=2048` and watch the trade-off move. Compression improves, but each token is seen less often in training, so a small dataset like Shakespeare starts to strain.

<details><summary>Hints</summary>

- Encoding 1MB of text with the naive `encode` from Project 1 can take a few minutes. That is fine: do it once and save the ids. (Speeding it up is a stretch goal.)
- If sampling crashes on decode with a `KeyError`, the model produced an id that `vocab` does not contain, usually the special token's id (`vocab_size`). Add an entry for each special token to `vocab`, or skip special ids when decoding. If the text shows replacement characters (`�`), the ids are being decoded one at a time; batch all generated ids into a single `decode` call.
- If the loss barely drops, check that the model's `vocab_size` really matches the tokenizer's. A mismatch of a few ids from special tokens causes silent index errors or wasted rows.

</details>

**Definition of done**: the GPT trains and samples end-to-end on tokens from the handmade BPE tokenizer, and `comparison.md` records sample quality plus effective context length in characters for both models.

## Stretch goals

- **Regex pre-splitting**: real GPT tokenizers first split text with a regex pattern (so merges never cross a word/punctuation boundary, e.g. `"dog."` never becomes one token). Implement the split-then-merge scheme like minbpe's `RegexTokenizer` and compare the merges with and without it.
- **Speed**: profile `encode` and make it at least 10x faster on the full Shakespeare file (cache pair lookups, avoid rescanning the whole sequence after each merge).
- **Vocab sweep**: train tokenizers at vocab 256, 512, 1024, 2048, 4096 and plot bytes-per-token vs vocab size. The plot flattens (diminishing returns), which is why real vocabs stop around 50k–200k.
- **Read minbpe for real**: once the Project 1 tokenizer works, read the [minbpe](https://github.com/karpathy/minbpe) source top to bottom and list three things Karpathy did more elegantly than the handmade version.

## Getting unstuck

- **Round-trip fails on non-English text**: print the byte lists. The bug is nearly always in `decode`: it must join all token bytes first and UTF-8-decode once, with `errors="replace"`.
- **`encode` gives different ids than training produced**: the merge replay order is wrong. Merges must apply in the order they were learned (lowest new id first), not by frequency in the new text.
- **Merge loop index bugs**: hand-trace `merge([1,2,2,1,2], (1,2), 99)` on paper, one index at a time. Nearly everyone makes off-by-one errors here.
- **Project 3 loss looks "bad"**: re-read the milestone. Cross-vocab loss comparisons are meaningless. Judge by samples and by per-character measures.
- Standing advice: read error messages bottom-up (the last line names the real problem); print shapes, lengths and the first 10 elements constantly; ask an AI assistant for a **hint** rather than the solution; and type every line of code by hand, with no pasting.

## Resources

- [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — the lecture this lesson follows; build along with it (it is in the Zero to Hero playlist).
- [minbpe](https://github.com/karpathy/minbpe) — Karpathy's clean reference implementation; read it *after* the Project 1 version works.
- [OpenAI tokenizer playground](https://platform.openai.com/tokenizer) — paste any text and watch a production tokenizer split it; useful for Project 2 cross-checks.

## Skills unlocked

- [ ] I can explain the character/word/subword trade-off and why modern LLMs use byte-level BPE.
- [ ] I can implement BPE training, encoding and decoding from scratch and prove the round-trip on any Unicode text.
- [ ] I can explain why merge *order* matters when encoding new text.
- [ ] I can use tiktoken to inspect how a production tokenizer splits any string.
- [ ] I can explain why LLMs struggle with spelling, letter-counting and arithmetic, from the tokenizer outward.
- [ ] I can explain why the same meaning costs more tokens in Hindi than in English, and what that implies for cost and context.
- [ ] I can swap the tokenizer under a language model and reason about vocab size vs sequence length.
- [ ] I know what a special token is and why it must never be forgeable from ordinary text.

## Next up

Both the GPT and the tokenizer are now handmade, so the next lesson scales the whole pipeline properly with nanoGPT: [33 · Train Your Own GPT (nanoGPT, Karpathy 9)](33-train-your-own-gpt.md).
