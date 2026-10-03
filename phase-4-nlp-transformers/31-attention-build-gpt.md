# 31 · Attention and the Transformer: Let's Build GPT (Karpathy 7)

**Phase 4 — NLP and Transformers** · Estimated time: 2 weeks · Prerequisites: [25 · Training Deep Nets](../phase-3-deep-learning/25-training-deep-nets.md), [26 · Convolutional Neural Networks](../phase-3-deep-learning/26-cnns-computer-vision.md), [28 · makemore](28-language-models-makemore.md), [29 · RNNs and LSTMs](29-rnn-lstm-char-generation.md), [30 · Embeddings](30-embeddings-word2vec.md)

> This lesson is the summit of the from-scratch track: it builds a transformer, typed line by line, and trains it to write Shakespeare-flavored text. The transformer is the architecture behind GPT, ChatGPT, Claude, and essentially all modern AI. Every ingredient is already familiar from earlier lessons: matrix multiplication (lesson 08), gradients and backprop (09, 22), embeddings (28, 30), normalization layers (25), and residual (skip) connections (26). The lesson puts them together with one new idea called *attention*. Plan for two focused weeks.

## What this lesson builds

- **Project 1 — Attention on paper:** a NumPy notebook that computes queries, keys, values, attention weights and outputs by hand for a 4-token toy example, printing every intermediate matrix.
- **Project 2 — A hand-built GPT:** a working GPT (`gpt.py`), built step by step alongside Karpathy's "Let's build GPT" video and trained on tiny Shakespeare, with generated samples and a loss log that beats the lesson-29 LSTM.
- **Project 3 — Ablation lab:** a set of deliberately broken GPT variants (no residuals, no positional embeddings, 1 head vs 6) plus a written finding for each, as proof of understanding *why* each piece exists.

## Concepts covered

- The RNN bottleneck: why squeezing a whole sequence through one fixed-size memory vector limits what the network can remember.
- Queries, keys, values: attention as a *soft dictionary lookup* where every token asks a question and every token offers an answer.
- Scaled dot-product attention: the one equation that changed the field.
- Causal masking: mechanically preventing tokens from peeking at the future.
- Multi-head attention: running several small attention lookups in parallel.
- Positional embeddings: telling the model *where* each token sits, since attention alone has no sense of order.
- The transformer block: attention + MLP + residuals + layernorm (every piece already familiar from lessons 22–26).
- Why transformers train in parallel across all positions while RNNs must crawl one step at a time.

## Before starting

Check the prerequisites honestly. Lessons 25 and 26 should be finished, with a clear idea of what normalization layers (lesson 25's BatchNorm; layernorm is its close relative) and skip connections (lesson 26) are *for*. Lesson 28 should be finished too, with a trained character-level language model and a sense of what "loss ≈ 2.5" feels like. Lesson 29 should have left an LSTM with its loss written down, because this lesson needs that number.

Activate the venv from lesson 01, check that PyTorch (installed in lesson 24) imports and prints its version, and create the work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"
mkdir -p work/31-attention-build-gpt
cd work/31-attention-build-gpt
```

For the data, reuse the tiny Shakespeare file from lesson 29:

```bash
cp ../29-rnn-lstm/input.txt .
wc -c input.txt
```

`wc -c` counts the file's bytes and should print about 1115394, roughly 1.1 MB of Shakespeare. If the file no longer exists, download it again the same way lesson 29 did:

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**A note on hardware:** the final scaled-up model in the video trains on a GPU. On a CPU-only computer, everything up to the scale-up step runs fine in minutes. For the big config, either shrink it (instructions in Project 2) or use a free cloud GPU (search for "Google Colab", then upload `gpt.py` and `input.txt` there). An Apple Silicon Mac on macOS 14 or later can also speed training up on its built-in GPU, which PyTorch calls the `mps` device. To use it, where the video's code sets `device`, type this line instead: `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`. It picks an NVIDIA GPU if there is one, else a Mac's GPU, else the CPU, so the same script runs on any computer. On an Intel Mac the `import torch` check above fails, because current PyTorch cannot be installed there at all: do Projects 2 and 3 in Colab (Project 1 needs only NumPy).

## Project 1 — Attention on paper first

**Goal:** demystify attention completely by computing it once with matrices small enough to read, before any PyTorch. Later, when the line `wei = q @ k.transpose(-2, -1)` is typed, it will be clear exactly what every number is.

**Milestones**

- [ ] Create a notebook `attention_by_hand.ipynb`. Make a toy "sentence" of 4 tokens and give each an 8-dimensional embedding: `x = np.random.randn(4, 8)` with a fixed seed (`np.random.seed(42)`). This is the `(T=4, C=8)` input: 4 positions, 8 channels. Print it.
- [ ] The big idea, in code: each token emits a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what will I hand over if someone attends to me?"). Create three random weight matrices `Wq, Wk, Wv` of shape `(8, 4)` with `np.random.randn(8, 4) / np.sqrt(8)` (dividing by the square root of the input width keeps the entries of Q and K near unit size, which the scaling step below relies on) and compute `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`. Checkpoint: all three have shape `(4, 4)`, for 4 tokens and head size 4.
- [ ] Compute the raw **attention scores**: `scores = Q @ K.T`, shape `(4, 4)`. Entry `[i, j]` is the dot product of token i's query with token j's key. It is a similarity score: "how relevant is token j to what token i is looking for?" Print it and pick one entry; verify it by hand with `np.dot(Q[i], K[j])`.
- [ ] Scale: divide scores by `np.sqrt(4)` (square root of head size). To see why, multiply the scores by 8 and softmax each row. The rows collapse to nearly one-hot, meaning each token attends to only one other token and gradients die (lesson 25's saturation problem, back again). Checkpoint: the exaggerated version puts nearly all of each row's weight on one or two tokens; the scaled version stays spread out.
- [ ] Apply the **causal mask**: a token may only attend to itself and earlier tokens, because at generation time the future does not exist yet. Use `np.tril` to build a lower-triangular mask and set the upper triangle of `scores` to `-np.inf` *before* softmax. Checkpoint: after softmax (row-wise), every row sums to 1.0, the upper triangle is exactly 0, and row 0 is `[1, 0, 0, 0]` (the first token can only attend to itself).
- [ ] Compute the output: `out = weights @ V`, shape `(4, 4)`. Each output row is a weighted average of value vectors. This is a *soft dictionary lookup*: instead of returning one entry, it returns a blend, weighted by relevance. Checkpoint: `out[0]` equals `V[0]` exactly (why?).
- [ ] Prove causality: change token 3's embedding (`x[3] += 100`), rerun the whole computation, and diff the outputs. Checkpoint: `out[0]`, `out[1]`, `out[2]` are unchanged to the last decimal; only `out[3]` moved.
- [ ] Now read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Every diagram should map onto one of the matrices just printed.

<details><summary>Hints</summary>

- Keep everything tiny and printed. The whole point is that a `(4, 4)` matrix is small enough to keep in mind; resist the urge to make it "realistic".
- For masking: `scores[np.triu(np.ones((4, 4)), k=1) == 1] = -np.inf` is one way; `np.where(np.tril(np.ones((4,4))) == 0, -np.inf, scores)` is another. The trick is that `-np.inf` becomes exactly 0 after softmax.
- Softmax per row: subtract the row max first for numerical stability (`e = np.exp(s - s.max(axis=1, keepdims=True))`), then divide by the row sum.
- If `out[0] != V[0]`, the mask is applied after softmax instead of before. Order matters.

</details>

**Definition of done:** one notebook where every intermediate (Q, K, V, scores, masked scores, weights, output) is printed and annotated with a one-line comment in the learner's own words, and both checkpoints (rows sum to 1, future edits do not leak backward) pass.

## Project 2 — Build GPT with Karpathy

**Goal:** type along with "Let's build GPT: from scratch, in code, spelled out" (video 7 of [the Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)) and end up with a hand-built, working GPT trained on tiny Shakespeare. Type every line instead of copying and pasting. Pause the video constantly.

**Milestones**

- [ ] **Data pipeline** (this is lesson 28's machinery, so it should feel familiar): load `input.txt`, build the character vocabulary, write `encode`/`decode`, split 90/10 into train/val, and write `get_batch(split)` returning `x, y` of shape `(batch_size, block_size)` where `y` is `x` shifted by one. Checkpoint: vocab size is 65, and `decode(encode("hello"))` returns `"hello"`.
- [ ] **Bigram baseline:** build the same bigram model as lesson 28, but as a PyTorch `nn.Module` with a training loop and a `generate` method. Train a few thousand steps. Checkpoint: val loss around **2.5** and the samples are recognizable letter-soup, not English. Write this loss down: it is the number to beat.
- [ ] **The mathematical trick:** before real attention, replicate Karpathy's averaging demo. Use a lower-triangular matrix and softmax to make each position an average of all previous positions, in one matrix multiply. This is Project 1's masking trick, now batched to shape `(B, T, T)`. Checkpoint: version 2 (matmul) and version 3 (softmax) produce the same tensor as the explicit loop, up to float rounding. Compare with `torch.allclose(xbow, xbow2, atol=1e-6)`: with the default tolerance, current PyTorch can print `False`, because the results differ by about 3e-8.
- [ ] **Single attention head:** implement a `Head` module. It needs `nn.Linear` layers for key/query/value, the scaled dot product, `masked_fill` with the registered `tril` buffer, softmax and a weighted sum. Add token embeddings *plus* **positional embeddings** (a learned `(block_size, n_embd)` table; without it, attention cannot tell "the dog bit the man" from "the man bit the dog", because the math in Project 1 never used position). Train. Checkpoint: val loss drops to roughly **2.4**, which is better than bigram because each character can now consult its past.
- [ ] **Multi-head attention:** run several smaller heads in parallel and concatenate the results. This works like several independent lookups, each free to specialize (one head might track vowels, another quotes). Checkpoint: val loss around **2.28**.
- [ ] **Feedforward layer:** add the per-token MLP (the "think about what was gathered" step after attention's "gather"). Checkpoint: loss improves again, roughly **2.24**.
- [ ] **Blocks, residuals, layernorm:** stack `Block`s of attention + MLP. First watch it *fail*: deep stacks without help train badly (lesson 25 predicted this). Then add residual connections and layernorm (pre-norm, as in the video), plus projection layers. Checkpoint: train loss dips just below **2.0** and val loss lands around **2.05–2.1**. Val loss falls clearly below 2.0 only after the scale-up in the next step.
- [ ] **Scale up:** add dropout, then grow to the video's config (`n_embd=384, n_head=6, n_layer=6, block_size=256, batch_size=64`). On the video's A100 GPU this trains in ~15 min to val loss around **1.48**; a free Colab GPU is slower, so expect it to take noticeably longer. CPU-only? Use `n_embd=128, n_head=4, n_layer=4, block_size=64` and expect ~1.8 in under an hour, which is still dramatically better than bigram.
- [ ] **The showdown:** generate 1000 characters. Then look up the lesson-29 LSTM's val loss and parameter count (`sum(p.numel() for p in model.parameters())`). Checkpoint: at comparable size, the GPT's val loss is lower and its samples look more like a play script (character names, line breaks, dialogue structure). Record both numbers in a comment at the top of `gpt.py`.
- [ ] **Why the parallelism matters:** write a 5-line comment in the learner's own words: the LSTM had to process T characters in T sequential steps (each hidden state waits for the previous one); the transformer computes all T positions in one batched matmul, which is exactly what GPUs are good at. That is why transformers scaled and RNNs did not.

<details><summary>Hints</summary>

- Adopt Karpathy's `B, T, C` naming (batch, time, channels) everywhere and comment the shape after every line while learning: `# (B, T, head_size)`. Nine out of ten bugs in this lesson are shape bugs.
- `self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))` stores the mask on the module without making it a trainable parameter. Slice it to `[:T, :T]` inside `forward`, because at generation time T can be shorter than `block_size`.
- In `generate`, crop the context to the last `block_size` tokens before each forward pass, or the positional embedding table will be indexed out of range.
- If loss hits `nan` after adding attention: check that `masked_fill` with `float('-inf')` comes *before* softmax.
- The video scales the scores by `C**-0.5`, where `C` is `n_embd`. The correct factor is `head_size**-0.5` (Karpathy's repo now writes `k.shape[-1]**-0.5`). The video's version still trains and causes no `nan`; its attention is just a little softer than intended.

</details>

**Definition of done:** `gpt.py` runs top to bottom, trains, prints train/val loss every few hundred steps, saves a 1000-character sample to `sample.txt`, and beats the lesson-29 LSTM's val loss at comparable parameter count, with the comparison numbers recorded in the file.

## Project 3 — Ablation lab

**Goal:** break the GPT on purpose, one component at a time. Anyone can copy an architecture; watching each part fail is how to *know* it. (An **ablation** is exactly this: remove one component and measure the damage.)

**Milestones**

- [ ] Set up: copy `gpt.py` to `ablations.py` and add simple flags (e.g. `USE_RESIDUAL`, `USE_POS_EMB`, `N_HEAD` constants at the top). Fix the seed (`torch.manual_seed(1337)`), pick a config that trains in ~10 minutes on the available hardware, and first record the **baseline** val loss and a sample.
- [ ] **Ablation 1 — remove residual connections** (`x = self.sa(self.ln1(x))` instead of `x = x + self.sa(self.ln1(x))`). This is lesson 26's skip-connection idea made vivid: without the gradient highway, a 4–6 layer stack barely trains. Checkpoint: val loss stalls far above baseline (typically stuck above 2.3) or bounces around instead of descending. Write one paragraph: what happened, and why residuals fix it.
- [ ] **Ablation 2 — remove positional embeddings** (feed token embeddings only). Attention by itself ignores order: shuffle the inputs and the outputs only shuffle along (Project 1 never used position). Only the causal mask leaves a weak hint of order, because each position sees a different number of earlier characters, so the model has to work out order indirectly. Checkpoint: loss is modestly worse than baseline, but the *samples* tell the real story: word-salad with much weaker word and line structure. One paragraph.
- [ ] **Ablation 3 — 1 head vs 6 heads** at equal total width (1 head of size 384 vs 6 heads of size 64, or scaled to the chosen config). Checkpoint: same parameter count within ~1%, but the multi-head model reaches a noticeably lower val loss. One paragraph on why several small specialized lookups beat one big one.
- [ ] Optional 4th — remove the `1/sqrt(head_size)` scaling. Connect the result to the Project 1 one-hot softmax experiment and lesson 25's saturation.
- [ ] Collect results in `work/31-attention-build-gpt/ablations.md`: a small table (variant, params, final val loss) and one paragraph per ablation run. This file is the proof of understanding, worth rereading before every ML interview for years to come.

<details><summary>Hints</summary>

- Keep everything else identical between runs: same seed, same steps, same eval interval. Change one variable at a time; that is the whole scientific method of this lesson.
- If the no-residual run diverges instead of stalling, lower the learning rate for that run only and note it in the finding. Instability *is* a finding.
- For fair 1-vs-6-head comparison, print parameter counts for both before training and check they roughly match.

</details>

**Definition of done:** `ablations.md` exists with the results table, one honest paragraph per ablation, and loss numbers that a stranger could reproduce from the flags and seed.

## Stretch goals

- **See attention:** save the `(T, T)` attention weight matrix from one head and `plt.imshow` it for a real text snippet. Look for heads that attend to the previous character, to spaces, to the start of the line.
- **Read nanoGPT:** go through the [nanoGPT repo](https://github.com/karpathy/nanoGPT) (`model.py` first) and list five concrete differences from this lesson's `gpt.py` (weight tying, initialization details, flash attention, ...). This is professional-grade code for exactly what Project 2 built.
- **A custom corpus:** train on a public-domain book (search Project Gutenberg) or on personal writing. Watching the GPT imitate a different voice makes the "it models the training distribution" idea concrete.
- **Sinusoidal positions:** replace the learned positional table with the fixed sin/cos encodings from the original "Attention Is All You Need" paper and compare losses.

## Getting unstuck

- **Shape errors** (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`): print `.shape` after every line in `forward`. Know the target shapes by heart: input `(B, T)`, embeddings `(B, T, C)`, attention weights `(B, T, T)`, head output `(B, T, head_size)`.
- **Loss is `nan`:** the cause is almost always the mask. The `-inf` must go in before softmax. Also check that softmax runs over the right dimension (`dim=-1`).
- **Loss does not drop below ~2.5:** the model may be accidentally bigram. Verify that the attention weights are not all uniform, and that `y` really is `x` shifted by one.
- **Painfully slow:** shrink `block_size` and `n_layer` first; a small model that finishes teaches more than a big one that is abandoned.
- Standing advice: read error tracebacks bottom-up; print shapes and a few actual values, not just types; when stuck 30+ minutes, ask an AI assistant for a **hint** ("what concept am I missing?"), never the solution; and type all code by hand, in this lesson above all others.

## Resources

- [Karpathy — Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Project 2 follows video 7, "Let's build GPT: from scratch, in code, spelled out", minute by minute.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the visual companion; read after Project 1, when every diagram maps to a matrix already printed.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — read *after* Project 2: the grown-up version of this lesson's `gpt.py`, and the source for the tiny Shakespeare download URL.
- [PyTorch docs](https://pytorch.org/docs/) — reference for `nn.Module`, `masked_fill`, `register_buffer`, `nn.LayerNorm`.

## Skills unlocked

- [ ] I can explain queries, keys and values as a soft dictionary lookup, without notes.
- [ ] I can write scaled dot-product attention in NumPy from memory and say why the scaling factor is there.
- [ ] I can explain causal masking and implement it with a triangular matrix and `-inf`.
- [ ] I can assemble a full transformer block and state in one sentence what attention, the MLP, residuals and layernorm each contribute.
- [ ] I can train a character-level GPT that beats the lesson-29 LSTM at the same parameter count, and I have the loss numbers to show it.
- [ ] I can predict what breaks when residuals, positional embeddings or extra heads are removed, because I broke them and wrote it down.
- [ ] I can explain why transformers train in parallel and RNNs cannot, and why that mattered for the history of AI.

## Next up

The GPT in this lesson reads characters, but real GPTs read *tokens*, so the next lesson builds the very tokenizer that GPT uses (byte-pair encoding) from scratch: [32 · Tokenization: Build the GPT Tokenizer (Karpathy 8)](32-tokenization-bpe.md).
