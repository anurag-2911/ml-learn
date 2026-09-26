# 33 · Train Your Own GPT (nanoGPT, Karpathy 9)

**Phase 4 — NLP & Transformers** · Estimated time: 2 weeks · Prerequisites: [31 · Attention and the Transformer](31-attention-build-gpt.md), [32 · Tokenization: Build the GPT Tokenizer](32-tokenization-bpe.md), [25 · The Dark Arts of Training Deep Networks](../phase-3-deep-learning/25-training-deep-nets.md)

> In lesson 31 you built a GPT that babbles Shakespeare at the character level. That was a toy. This lesson is where you graduate: you will assemble a real corpus of text *you* care about, tokenize it properly, and run a multi-million-parameter training run with everything a professional run has — checkpoints, a learning-rate schedule, a validation curve, and sample generations that are unmistakably in the style of your data. This is also the summit of the Karpathy arc you have been climbing since lesson 22. When you finish, you will understand — end to end, with your own hands — how the engines of modern AI are built.

## What you will build

- **Project 1 — Your corpus, prepared at scale**: a 5–50MB text dataset of your choosing, tokenized once into `train.bin` / `val.bin` memory-mapped files, plus a `prepare.py` script that produced them.
- **Project 2 — A real training run**: your lesson-31 GPT scaled to ~10M parameters, trained on a free cloud GPU with warmup + cosine learning-rate schedule, checkpointing, resume support, a logged validation-loss curve, and a README showing 3 generations at different temperatures.
- **Project 3 — The stretch summit**: watch Karpathy's "Let's reproduce GPT-2 (124M)" and understand every piece; optionally (~$10–40 of rented GPU) attempt the run yourself.

## Concepts you will learn by doing

- **Scaling up** — what changes when a model stops being a toy: mixed precision, gradient accumulation, learning-rate schedules
- **Reading professional code** — the nanoGPT codebase as a reference implementation to compare against your own
- **Dataset preparation at scale** — tokenize once, store as binary, read with `memmap` so data loading is never the bottleneck
- **Managing a training run** — checkpoints, resuming after a crash, tracking validation loss over hours
- **GPU realities** — free tiers (Colab, Kaggle) vs renting (Runpod, Lambda), and what a few dollars buys
- **Sampling strategies** — temperature and top-k, the knobs that turn one model into many voices
- **Pretraining vs finetuning** — what those words actually mean, setting up lesson 36

## Before you start

Check you have: your working GPT from [lesson 31](31-attention-build-gpt.md), your BPE tokenizer from [lesson 32](32-tokenization-bpe.md), and comfort with the training diagnostics from [lesson 25](../phase-3-deep-learning/25-training-deep-nets.md).

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch numpy tiktoken matplotlib
mkdir -p work/33-train-your-own-gpt
cd work/33-train-your-own-gpt
```

Clone nanoGPT next to your work folder — not to run blindly, but to read:

```bash
git clone https://github.com/karpathy/nanoGPT.git
```

You will also need a Google account for Colab (free GPU) or a Kaggle account (free GPU hours per week). Both are free; pick one now and confirm you can open a notebook with a GPU attached (`Runtime → Change runtime type → GPU` in Colab).

No dataset download is prescribed — choosing your corpus **is** Project 1.

## Project 1 — Corpus of your choice

**Goal**: Assemble 5–50MB of raw text you genuinely love, and turn it into two memory-mapped binary token files. A model trained on data you know intimately is one you can actually judge.

**Milestones**

- [ ] Pick your corpus. Ideas: everything by one author from Project Gutenberg (gutenberg.org — plain-text downloads, no account needed), song lyrics, a Wikipedia subset, your favorite out-of-copyright books, or a pile of source code. Target at least 5MB of raw text (check with `du -sh`); under that, a 10M-param model will memorize rather than learn. Checkpoint: `wc -c corpus.txt` shows at least 5,000,000 bytes.
- [ ] Clean it lightly in a `prepare.py` script: strip boilerplate (Gutenberg files have legal headers/footers — remove them programmatically, not by hand), normalize weird whitespace, concatenate everything into one big string. Print the first and last 500 characters to eyeball it. Checkpoint: no license text or HTML junk visible at either end.
- [ ] Tokenize the whole corpus **once**. Use your lesson-32 BPE tokenizer, or `tiktoken` (OpenAI's fast tokenizer library) with the `gpt2` encoding — using tiktoken is not cheating; you built one, so you know exactly what it does. Time it. Checkpoint: you get a list/array of token IDs, and tokens ≈ bytes ÷ 3 to 4 for English text.
- [ ] Split 90/10 into train and validation, then write each as a raw binary file of `uint16` values (every GPT-2 token ID fits in 16 bits — that halves the file vs `uint32`):

```python
import numpy as np
ids = np.array(train_ids, dtype=np.uint16)
ids.tofile("train.bin")
```

- [ ] Write the loading side using `np.memmap` — a NumPy array that lives on disk and is read lazily, so a 50MB (or 50GB) dataset costs almost no RAM. Reopen the memmap inside your batch function to avoid a slow memory leak on some systems. Checkpoint: decoding tokens 1000–1100 from `train.bin` reproduces a readable slice of your corpus exactly.
- [ ] Record the stats in a README: raw size, token count, vocab used, train/val split. Checkpoint: you can state your dataset size in tokens (e.g. "11.2M tokens") — from now on that is how you measure data, like the professionals do.

<details><summary>Hints</summary>

- Gutenberg headers all contain the phrase `*** START OF` and footers `*** END OF` — find those markers and slice between them.
- `tiktoken.get_encoding("gpt2")` gives you `.encode(text)` and `.decode(ids)`. Encode in chunks of a few MB if memory gets tight.
- To read a batch from a memmap: pick random offsets `ix`, then stack `data[i : i+block_size]` slices and convert with `torch.from_numpy(chunk.astype(np.int64))`.
- If your own BPE tokenizer is too slow for the full corpus, that is a real lesson about production code — note it, and switch to tiktoken without guilt.

</details>

**Definition of done**: `train.bin` and `val.bin` exist, a round-trip decode of any slice is correct, and `prepare.py` rebuilds both files from raw text in one command.

## Project 2 — The run

**Goal**: Scale your lesson-31 GPT to ~10M parameters and manage a real multi-hour training run on a free cloud GPU — schedules, checkpoints, logging, and generations that are unmistakably in the style of your corpus.

**Milestones**

- [ ] Read `model.py` and `train.py` in the nanoGPT repo, side by side with your lesson-31 code. It is the same architecture you built — that is the point. List 5 things nanoGPT does that yours doesn't (weight decay grouping, gradient clipping, mixed precision, gradient accumulation, cosine schedule are all in there). You may keep improving your own code or adapt nanoGPT — reading it counts as part of the lesson either way, but type what you borrow. Checkpoint: you can explain in one sentence each what your 5 items are for.
- [ ] Scale the config to roughly 10M parameters: try 6 layers, 6 heads, embedding size 384, block size 256, with your tokenizer's vocab size. Count parameters with `sum(p.numel() for p in model.parameters())`. Checkpoint: parameter count between 8M and 15M.
- [ ] Add a learning-rate schedule: **warmup** (start near zero and ramp up over the first few hundred steps, so early noisy gradients don't wreck the fresh weights) then **cosine decay** (glide smoothly down to ~10% of peak, so late training takes gentle steps). Write `get_lr(step)` as a pure function and plot it with matplotlib before training. Checkpoint: the plot shows a sharp ramp then a long smooth cosine curve down.
- [ ] Add checkpointing: every N steps, evaluate mean loss on `val.bin`, and if it's the best so far, `torch.save` a dict of model weights + optimizer state + step + config to `ckpt.pt`. Add a `--resume` path that loads all of it and continues. Test resume locally on CPU with a tiny config before you burn GPU time. Checkpoint: kill training with Ctrl-C, resume, and the loss continues from where it stopped instead of restarting high.
- [ ] Move to the GPU. Zip your work folder, upload to Colab or Kaggle, and confirm `torch.cuda.is_available()` is `True`. Enable **mixed precision** — computing mostly in 16-bit floats, which roughly doubles speed on modern GPUs with negligible quality loss — via `torch.autocast` (nanoGPT shows exactly how, including the `GradScaler` needed for float16). Checkpoint: a short timed run shows a clear speedup vs full precision.
- [ ] Measure throughput before the real run: tokens/second = batch × block_size × steps ÷ seconds. If the GPU wants bigger batches than memory allows, use **gradient accumulation** — run several small forward/backward passes, summing gradients, and step the optimizer once, simulating a large batch. Checkpoint: you can predict the length of your full run to within ~20% ("about 3 hours for 20k steps").
- [ ] The run itself: train for a few hours, logging `(step, train_loss, val_loss, lr)` to a CSV, generating a small sample every ~1000 steps so you can watch the style emerge, and downloading `ckpt.pt` periodically (free tiers disconnect — your resume code is your insurance). Checkpoint: val loss well below your lesson-31 character model's, and the loss curve plot is smooth with no divergence spikes.
- [ ] Sample properly. Implement **temperature** (divide logits by T before softmax: below 1.0 makes the model conservative and repetitive, above 1.0 makes it wild) and **top-k** (only sample among the k most likely tokens, cutting off the long tail of nonsense). Put 3 generations — T=0.8, T=1.0, T=1.2 or with/without top-k — in your README with the loss curve. Checkpoint: someone who knows your corpus would identify its style from the samples in 5 seconds. Unmistakably in-style generations are the bar.
- [ ] Bridge to lesson 36: what you just did is **pretraining** — teaching a model a domain from scratch by next-token prediction on raw text. **Finetuning** starts from someone else's pretrained checkpoint and nudges it with a small dataset toward a task. Write 3 sentences in your README on why finetuning is usually cheaper. Checkpoint: you can say what GPT's "P" stands for and why it matters.

<details><summary>Hints</summary>

- Cosine schedule shape: for step `t` past warmup, `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))` where `progress` goes 0→1 over the decay phase. Peak LR around 1e-3 with AdamW is a sane start at this scale.
- Checkpoint dict essentials: `model.state_dict()`, `optimizer.state_dict()`, `step`, and the config. Forgetting the optimizer state is the classic bug — resume "works" but the loss jumps because AdamW lost its momentum buffers.
- On Colab, `from google.colab import files` handles small uploads/downloads; mounting Google Drive is steadier for checkpoints on long runs.
- If loss goes to NaN: lower the peak LR first, confirm gradient clipping (`clip_grad_norm_` at 1.0) runs *before* `optimizer.step()`, and check your warmup actually starts near zero.

</details>

**Definition of done**: a completed multi-hour GPU run with a saved best checkpoint, a plotted val-loss curve, and a README containing throughput numbers plus 3 in-style generations at different temperatures.

## Project 3 — Stretch summit: reproduce GPT-2 (124M)

**Goal**: Understand — and if budget allows, execute — the full reproduction of GPT-2 124M, the real model OpenAI shipped in 2019. Watching and understanding the video **completes this lesson**; the run itself is optional.

**Milestones**

- [ ] Watch "Let's reproduce GPT-2 (124M)" from the [Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — it is long, so take it in sittings, notes open. Checkpoint: you can list 5 optimizations from the video that you did not use in Project 2 (candidates: kernel fusion via `torch.compile`, flash attention, vocab-size padding to a power-of-2-friendly number, distributed training across GPUs, weight tying).
- [ ] Read about [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), the open recreation of GPT-2's training data. Compare: your run was tens of millions of tokens; this is ~9 billion. Checkpoint: you can explain why scaling data and parameters *together* matters, in plain words.
- [ ] Decide on the run with a written cost estimate. Realistic paths: ~$10–40 on a rented multi-GPU machine (8×A100 runs roughly $10–15/hour, and the run finishes in a few hours as in the video; Runpod and Lambda are the usual suspects — you already know Runpod exists as your rental option, but note Colab's free tier was enough for everything up to now) or a much longer single-GPU slog. Before launching, do the arithmetic: expected cost = hourly rate × hours predicted from a 100-step throughput test (see the hints) — never trust a duration you haven't measured on the actual machine. **No budget means skip with zero guilt** — the understanding is the summit, not the receipt. Checkpoint: a paragraph in your README: run or skip, and why.
- [ ] (Optional, budget willing) Do it with nanoGPT's train script on the rented machine, applying everything from Project 2: measure throughput first, checkpoint to persistent storage, watch the val curve, and **set a spending alarm before launch**. Checkpoint: val loss on OpenWebText at or below the GPT-2 baseline reported in the video.
- [ ] **MILESTONE — the Karpathy arc is complete.** Look back at what you have personally built from scratch since lesson 22: **an autograd engine, an MLP, a CNN, an LSTM, a BPE tokenizer, and a GPT — trained on data you chose, with your own hands.** You now understand, end to end, how the engines of modern AI are built. Almost nobody — including many people working in AI — can say that. Write the look-back list in `PROGRESS.md` and read it out loud.

<details><summary>Hints</summary>

- The video builds the code live; keep nanoGPT open in VS Code and trace each change he makes to the corresponding file.
- If renting: create the machine, run a 100-step throughput test, multiply the predicted hours by the hourly rate to get your expected bill, and only then commit to the full run. Ten minutes of testing protects hours of budget.
- Rented machines usually bill until you *terminate*, not merely stop — check the provider's billing page before you walk away.

</details>

**Definition of done**: video watched with notes, the 5-optimizations list and run-or-skip paragraph written, and the milestone look-back recorded in `PROGRESS.md`. The run itself is a bonus, never a requirement.

## Stretch goals

- Train two model sizes (e.g. 3M and 10M) on the same data for the same wall-clock time and plot both val curves — a first taste of scaling laws.
- Implement top-p (nucleus) sampling — sample from the smallest set of tokens whose probabilities sum to p — and compare against top-k on your model.
- Add `torch.compile(model)` and measure the throughput change on your GPU.
- Build a tiny CLI: `python generate.py --prompt "..." --temperature 0.8 --top_k 50` that loads your best checkpoint and streams tokens as they generate.

## If you get stuck

- **Loss stuck near `ln(vocab_size)`** (~10.8 for a 50k vocab): that is the loss for uniform guessing — the model is learning nothing. Suspect the data pipeline: decode a training batch and *read it*; if it's garbage, your bug is in Project 1, not the model.
- **Great train loss, flat val loss**: overfitting — your corpus is too small for the model. Shrink the model or grow the data.
- **Colab disconnects mid-run**: this is normal, not failure — it's exactly why you built resume. Checkpoint to Drive and continue.
- **CUDA out of memory**: reduce batch size and compensate with gradient accumulation; if needed, reduce block size.
- Standing advice: read the error traceback bottom-up (the last line is the actual error); print shapes and dtypes at every boundary; ask an AI assistant for a **hint**, not a solution; and type all code yourself — copy-paste teaches nothing.

## Resources

- [Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — "Let's reproduce GPT-2 (124M)" is the finale and the heart of Project 3.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the professional reference codebase; read it, compare it to your lesson-31 code, borrow from it deliberately.
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — the open recreation of GPT-2's training data; the dataset card explains what real pretraining data looks like.
- [PyTorch docs](https://pytorch.org) — look up `torch.autocast`, `GradScaler`, and `clip_grad_norm_` when you wire in mixed precision.
- Project Gutenberg (gutenberg.org) — tens of thousands of public-domain books as plain text, no account needed; the classic corpus source.

## Skills unlocked

- [ ] I can prepare a large text dataset: tokenize once, store as binary, and load it with memmap without exhausting RAM.
- [ ] I can implement a warmup + cosine learning-rate schedule and explain why each phase exists.
- [ ] I can checkpoint and resume a training run, including optimizer state, and have proven it by killing and resuming a run.
- [ ] I can run training on a cloud GPU, use mixed precision and gradient accumulation, and predict a run's duration from measured throughput.
- [ ] I can control generation with temperature and top-k and explain what each does to the token distribution.
- [ ] I can read a professional codebase like nanoGPT and map it onto code I wrote myself.
- [ ] I can explain pretraining vs finetuning and when each is the right tool.
- [ ] I have built, from scratch: an autograd engine, an MLP, a CNN, an LSTM, a tokenizer, and a trained GPT.

## Next up

You have built the engine — now learn to drive the giant ones: [34 · Using LLMs: APIs, Prompting and Structured Output](../phase-5-llms/34-llm-apis-prompting.md).
