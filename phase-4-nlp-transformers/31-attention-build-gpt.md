# 31 · Attention and the Transformer: Let's Build GPT (Karpathy 7)

**Phase 4 — NLP and Transformers** · Estimated time: 2 weeks · Prerequisites: [25 · Training Deep Nets](../phase-3-deep-learning/25-training-deep-nets.md), [28 · makemore](28-language-models-makemore.md), [29 · RNNs and LSTMs](29-rnn-lstm-char-generation.md), [30 · Embeddings](30-embeddings-word2vec.md)

> This is the summit of the from-scratch track. The transformer is the architecture behind GPT, ChatGPT, Claude, and essentially all modern AI — and by the end of this lesson you will have built one yourself, typed line by line, and trained it to write Shakespeare-flavored text. Here is the best part: you already know every ingredient. Matrix multiplication (lesson 08), gradients and backprop (09, 22), embeddings (28, 30), residual connections and layernorm (25) — this lesson just snaps them together with one new idea called *attention*. Two focused weeks. Take them.

## What you will build

- **Project 1 — Attention on paper:** a NumPy notebook where you compute queries, keys, values, attention weights and outputs by hand for a 4-token toy example, printing every intermediate matrix.
- **Project 2 — Your own GPT:** a working GPT (`gpt.py`), built step by step alongside Karpathy's "Let's build GPT" video, trained on tiny Shakespeare — with generated samples and a loss log that beats your lesson-29 LSTM.
- **Project 3 — Ablation lab:** a set of deliberately broken GPT variants (no residuals, no positional embeddings, 1 head vs 6) plus a written finding for each, proving you understand *why* each piece exists.

## Concepts you will learn by doing

- The RNN bottleneck: why squeezing a whole sequence through one fixed-size memory vector limits what the network can remember
- Queries, keys, values — attention as a *soft dictionary lookup* where every token asks a question and every token offers an answer
- Scaled dot-product attention — the one equation that changed everything
- Causal masking — mechanically preventing tokens from peeking at the future
- Multi-head attention — running several small attention lookups in parallel
- Positional embeddings — telling the model *where* each token sits, since attention alone has no sense of order
- The transformer block: attention + MLP + residuals + layernorm — every piece already familiar from lessons 22–25
- Why transformers train in parallel across all positions while RNNs must crawl one step at a time

## Before you start

Check prerequisites honestly: you should have finished lesson 25 (you know what residuals and layernorm are *for*), lesson 28 (you have trained a character-level language model and know what "loss ≈ 2.5" feels like), and lesson 29 (you have an LSTM whose loss you wrote down — you will need that number).

```bash
cd ~/ml/ml-learn
source .venv/bin/activate          # the venv from lesson 01
python3 -c "import torch; print(torch.__version__)"   # installed in lesson 24
mkdir -p work/31-attention-build-gpt
cd work/31-attention-build-gpt
```

Get the data — reuse the tiny Shakespeare file from lesson 29:

```bash
cp ../29-rnn-lstm/input.txt .    # ~1.1 MB of Shakespeare
wc -c input.txt                  # should print about 1115394
```

If you no longer have it, download it again the same way lesson 29 did:

```bash
wget -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**A note on hardware:** the final scaled-up model in the video trains on a GPU. On a CPU-only WSL2 machine, everything up to the scale-up step runs fine in minutes; for the big config either shrink it (instructions in Project 2) or use a free cloud GPU (search for "Google Colab" — upload your `gpt.py` and `input.txt` there).

## Project 1 — Attention on paper first

**Goal:** demystify attention completely by computing it once with matrices small enough to read, before any PyTorch. When you later type `wei = q @ k.transpose(-2, -1)`, you will know exactly what every number is.

**Milestones**

- [ ] Create a notebook `attention_by_hand.ipynb`. Make a toy "sentence" of 4 tokens and give each an 8-dimensional embedding: `x = np.random.randn(4, 8)` with a fixed seed (`np.random.seed(42)`). This is your `(T=4, C=8)` input — 4 positions, 8 channels. Print it.
- [ ] The big idea, in code: each token emits a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what will I hand over if someone attends to me?"). Create three random weight matrices `Wq, Wk, Wv` of shape `(8, 4)` and compute `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`. Checkpoint: all three have shape `(4, 4)` — 4 tokens, head size 4.
- [ ] Compute the raw **attention scores**: `scores = Q @ K.T`, shape `(4, 4)`. Entry `[i, j]` is the dot product of token i's query with token j's key — a similarity score: "how relevant is token j to what token i is looking for?" Print it and pick one entry; verify it by hand with `np.dot(Q[i], K[j])`.
- [ ] Scale: divide scores by `np.sqrt(4)` (square root of head size). To see why, multiply your scores by 8 and softmax each row — the rows collapse to nearly one-hot, meaning each token attends to only one other token and gradients die (lesson 25's saturation problem, back again). Checkpoint: the exaggerated version looks one-hot; the scaled version stays spread out.
- [ ] Apply the **causal mask**: a token may only attend to itself and earlier tokens, because at generation time the future does not exist yet. Use `np.tril` to build a lower-triangular mask and set the upper triangle of `scores` to `-np.inf` *before* softmax. Checkpoint: after softmax (row-wise), every row sums to 1.0, the upper triangle is exactly 0, and row 0 is `[1, 0, 0, 0]` — the first token can only attend to itself.
- [ ] Compute the output: `out = weights @ V`, shape `(4, 4)`. Each output row is a weighted average of value vectors — a *soft dictionary lookup*: instead of returning one entry, it returns a blend, weighted by relevance. Checkpoint: `out[0]` equals `V[0]` exactly (why?).
- [ ] Prove causality: change token 3's embedding (`x[3] += 100`), rerun the whole computation, and diff the outputs. Checkpoint: `out[0]`, `out[1]`, `out[2]` are unchanged to the last decimal; only `out[3]` moved.
- [ ] Now read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — every diagram should map onto a matrix you just printed.

<details><summary>Hints</summary>

- Keep everything tiny and printed. The whole point is that a `(4, 4)` matrix fits in your head; resist the urge to make it "realistic".
- For masking: `scores[np.triu(np.ones((4, 4)), k=1) == 1] = -np.inf` is one way; `np.where(np.tril(np.ones((4,4))) == 0, -np.inf, scores)` is another. `-np.inf` becomes exactly 0 after softmax — that is the trick.
- Softmax per row: subtract the row max first for numerical stability (`e = np.exp(s - s.max(axis=1, keepdims=True))`), then divide by the row sum.
- If `out[0] != V[0]`, your mask is applied after softmax instead of before. Order matters.

</details>

**Definition of done:** one notebook where every intermediate (Q, K, V, scores, masked scores, weights, output) is printed and annotated with a one-line comment in your own words, and both checkpoints (rows sum to 1, future edits don't leak backward) pass.

## Project 2 — Build GPT with Karpathy

**Goal:** type along with "Let's build GPT: from scratch, in code, spelled out" (video 7 of [the Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)) and end up with your own working GPT trained on tiny Shakespeare. Type every line yourself — no copy-paste. Pause the video constantly.

**Milestones**

- [ ] **Data pipeline** (this is lesson 28's machinery — should feel familiar): load `input.txt`, build the character vocabulary, write `encode`/`decode`, split 90/10 into train/val, and write `get_batch(split)` returning `x, y` of shape `(batch_size, block_size)` where `y` is `x` shifted by one. Checkpoint: vocab size is 65, and `decode(encode("hello"))` returns `"hello"`.
- [ ] **Bigram baseline:** build the same bigram model as lesson 28, but as a PyTorch `nn.Module` with a training loop and a `generate` method. Train a few thousand steps. Checkpoint: val loss around **2.5** and the samples are recognizable letter-soup, not English. Write this loss down — it is the number to beat.
- [ ] **The mathematical trick:** before real attention, replicate Karpathy's averaging demo — use a lower-triangular matrix and softmax to make each position an average of all previous positions, in one matrix multiply. This is Project 1's masking trick, now batched to shape `(B, T, T)`. Checkpoint: version 2 (matmul) and version 3 (softmax) produce the same tensor as the explicit loop.
- [ ] **Single attention head:** implement a `Head` module — `nn.Linear` layers for key/query/value, the scaled dot product, `masked_fill` with the registered `tril` buffer, softmax, weighted sum. Add token embeddings *plus* **positional embeddings** (a learned `(block_size, n_embd)` table — without it, attention cannot tell "the dog bit the man" from "the man bit the dog", because the math in Project 1 never used position). Train. Checkpoint: val loss drops to roughly **2.4** — better than bigram, because each character can now consult its past.
- [ ] **Multi-head attention:** run several smaller heads in parallel and concatenate — like several independent lookups, each free to specialize (one head might track vowels, another quotes). Checkpoint: val loss around **2.28**.
- [ ] **Feedforward layer:** add the per-token MLP (the "think about what you gathered" step after attention's "gather"). Checkpoint: loss improves again, roughly **2.24**.
- [ ] **Blocks, residuals, layernorm:** stack `Block`s of attention + MLP. First watch it *fail* — deep stacks without help train badly (lesson 25 predicted this). Then add residual connections and layernorm (pre-norm, as in the video), plus projection layers. Checkpoint: val loss clearly below **2.0**.
- [ ] **Scale up:** add dropout, then grow to the video's config (`n_embd=384, n_head=6, n_layer=6, block_size=256, batch_size=64`). On a GPU this trains in ~15 min to val loss around **1.48**. CPU-only? Use `n_embd=128, n_head=4, n_layer=4, block_size=64` and expect ~1.8 in under an hour — still dramatically better than bigram.
- [ ] **The showdown:** generate 1000 characters. Then look up your lesson-29 LSTM's val loss and parameter count (`sum(p.numel() for p in model.parameters())`). Checkpoint: at comparable size, your GPT's val loss is lower and the samples look more like a play script (character names, line breaks, dialogue structure). Record both numbers in a comment at the top of `gpt.py`.
- [ ] **Why the parallelism matters:** write a 5-line comment in your own words: the LSTM had to process T characters in T sequential steps (each hidden state waits for the previous one); the transformer computes all T positions in one batched matmul, which is exactly what GPUs are good at. That is why transformers scaled and RNNs did not.

<details><summary>Hints</summary>

- Adopt Karpathy's `B, T, C` naming (batch, time, channels) everywhere and comment the shape after every line while learning: `# (B, T, head_size)`. Nine out of ten bugs in this lesson are shape bugs.
- `self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))` stores the mask on the module without making it a trainable parameter. Slice it to `[:T, :T]` inside `forward` — at generation time T can be shorter than `block_size`.
- In `generate`, crop the context to the last `block_size` tokens before each forward pass, or the positional embedding table will be indexed out of range.
- If loss hits `nan` after adding attention: check that you `masked_fill` with `float('-inf')` *before* softmax, and that the scale factor is `head_size**-0.5`, not `n_embd**-0.5`.

</details>

**Definition of done:** `gpt.py` runs top to bottom, trains, prints train/val loss every few hundred steps, saves a 1000-character sample to `sample.txt`, and beats your lesson-29 LSTM's val loss at comparable parameter count — with the comparison numbers recorded in the file.

## Project 3 — Ablation lab

**Goal:** break your GPT on purpose, one component at a time. Anyone can copy an architecture; you will *know* it, because you have watched each part fail. (An **ablation** is exactly this: remove one component and measure the damage.)

**Milestones**

- [ ] Set up: copy `gpt.py` to `ablations.py` and add simple flags (e.g. `USE_RESIDUAL`, `USE_POS_EMB`, `N_HEAD` constants at the top). Fix the seed (`torch.manual_seed(1337)`), pick a config that trains in ~10 minutes on your machine, and first record the **baseline** val loss and a sample.
- [ ] **Ablation 1 — remove residual connections** (`x = self.sa(self.ln1(x))` instead of `x = x + self.sa(self.ln1(x))`). This is lesson 25's lesson made vivid: without the gradient highway, a 4–6 layer stack barely trains. Checkpoint: val loss stalls far above baseline (typically stuck above 2.3) or bounces around instead of descending. Write one paragraph: what happened, and why residuals fix it.
- [ ] **Ablation 2 — remove positional embeddings** (feed token embeddings only). Attention is permutation-invariant — Project 1 never used position — so the model becomes a bag-of-recent-characters. Checkpoint: loss is modestly worse than baseline, but the *samples* tell the real story: word-salad with much weaker word and line structure. One paragraph.
- [ ] **Ablation 3 — 1 head vs 6 heads** at equal total width (1 head of size 384 vs 6 heads of size 64, or scaled to your config). Checkpoint: same parameter count within ~1%, but the multi-head model reaches a noticeably lower val loss. One paragraph on why several small specialized lookups beat one big one.
- [ ] Optional 4th — remove the `1/sqrt(head_size)` scaling. Connect what you see to your Project 1 one-hot softmax experiment and lesson 25's saturation.
- [ ] Collect results in `work/31-attention-build-gpt/ablations.md`: a small table (variant, params, final val loss) and your four paragraphs. This file is your proof of understanding — you will reread it before every ML interview for the rest of your career.

<details><summary>Hints</summary>

- Keep everything else identical between runs: same seed, same steps, same eval interval. You are changing one variable at a time — that is the whole scientific method of this lesson.
- If the no-residual run diverges instead of stalling, lower the learning rate for that run only and note it in the finding — instability *is* a finding.
- For fair 1-vs-6-head comparison, print parameter counts for both before training and check they roughly match.

</details>

**Definition of done:** `ablations.md` exists with the results table, one honest paragraph per ablation, and loss numbers a stranger could reproduce from your flags and seed.

## Stretch goals

- **See attention:** save the `(T, T)` attention weight matrix from one head and `plt.imshow` it for a real text snippet. Look for heads that attend to the previous character, to spaces, to the start of the line.
- **Read nanoGPT:** go through the [nanoGPT repo](https://github.com/karpathy/nanoGPT) (`model.py` first) and list five concrete differences from your `gpt.py` (weight tying, initialization details, flash attention, ...). This is professional-grade code for exactly what you just built.
- **Your own corpus:** train on a public-domain book (search Project Gutenberg) or your own writing. Watching your GPT imitate a different voice makes the "it models the training distribution" idea land.
- **Sinusoidal positions:** replace the learned positional table with the fixed sin/cos encodings from the original "Attention Is All You Need" paper and compare losses.

## If you get stuck

- **Shape errors** (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`): print `.shape` after every line in `forward`. Know your target shapes cold: input `(B, T)`, embeddings `(B, T, C)`, attention weights `(B, T, T)`, head output `(B, T, head_size)`.
- **Loss is `nan`:** almost always the mask — `-inf` must go in before softmax, and check you didn't softmax over the wrong dimension (`dim=-1`).
- **Loss won't drop below ~2.5:** your model may be accidentally bigram — verify attention weights are not all uniform, and that `y` really is `x` shifted by one.
- **Painfully slow:** shrink `block_size` and `n_layer` first; a small model that finishes teaches more than a big one you abandon.
- Standing advice: read error tracebacks bottom-up; print shapes and a few actual values, not just types; when stuck 30+ minutes, ask an AI assistant for a **hint** ("what concept am I missing?"), never the solution; and type all code yourself — this lesson above all others.

## Resources

- [Karpathy — Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Project 2 follows video 7, "Let's build GPT: from scratch, in code, spelled out", minute by minute.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the visual companion; read after Project 1, when every diagram maps to a matrix you have printed.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — read *after* Project 2: the grown-up version of your `gpt.py`, and your source for the tiny Shakespeare download URL.
- [PyTorch docs](https://pytorch.org/docs/) — reference for `nn.Module`, `masked_fill`, `register_buffer`, `nn.LayerNorm`.

## Skills unlocked

- [ ] I can explain queries, keys and values as a soft dictionary lookup, without notes.
- [ ] I can write scaled dot-product attention in NumPy from memory and say why the scaling factor is there.
- [ ] I can explain causal masking and implement it with a triangular matrix and `-inf`.
- [ ] I can assemble a full transformer block and state in one sentence what attention, the MLP, residuals and layernorm each contribute.
- [ ] I can train a character-level GPT that beats my LSTM at the same parameter count, and I have the loss numbers to show it.
- [ ] I can predict what breaks when you remove residuals, positional embeddings or extra heads — because I broke them and wrote it down.
- [ ] I can explain why transformers train in parallel and RNNs cannot, and why that mattered for the history of AI.

## Next up

Your GPT reads characters; real GPTs read *tokens* — next you build the very tokenizer that GPT uses, byte-pair encoding from scratch: [32 · Tokenization: Build the GPT Tokenizer (Karpathy 8)](32-tokenization-bpe.md).
