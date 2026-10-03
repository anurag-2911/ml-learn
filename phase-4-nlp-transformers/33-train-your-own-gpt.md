# 33 · Train Your Own GPT (nanoGPT, Karpathy 9)

**Phase 4 — NLP & Transformers** · Estimated time: 2 weeks · Prerequisites: [31 · Attention and the Transformer](31-attention-build-gpt.md), [32 · Tokenization: Build the GPT Tokenizer](32-tokenization-bpe.md), [25 · The Dark Arts of Training Deep Networks](../phase-3-deep-learning/25-training-deep-nets.md)

> Lesson 31 built a character-level GPT that babbles in the style of Shakespeare, but it was a toy. This lesson moves past the toy stage. It assembles a real text corpus of genuine personal interest, tokenizes it properly, and carries out a multi-million-parameter training run with everything a professional run has: checkpoints, a learning-rate schedule, a validation curve, and sample generations that are unmistakably in the style of the data. It is also the summit of the Karpathy arc that began in lesson 22. Finishing it gives a hands-on, end-to-end understanding of how the engines of modern AI are built.

## What this lesson builds

- **Project 1 — The corpus, prepared at scale**: a freely chosen 5–50MB text dataset, tokenized once into `train.bin` / `val.bin` memory-mapped files, plus a `prepare.py` script that produced them.
- **Project 2 — A real training run**: the lesson-31 GPT scaled to ~10M parameters, trained on a free cloud GPU with warmup + cosine learning-rate schedule, checkpointing, resume support, a logged validation-loss curve, and a README showing 3 generations at different temperatures.
- **Project 3 — The stretch summit**: watch Karpathy's "Let's reproduce GPT-2 (124M)" and understand every piece; optionally (~$10–40 of rented GPU), attempt the run as well.

## Concepts covered

- **Scaling up**: what changes when a model stops being a toy (mixed precision, gradient accumulation, learning-rate schedules).
- **Reading professional code**: the nanoGPT codebase as a reference implementation to compare against the lesson-31 code.
- **Dataset preparation at scale**: tokenize once, store as binary, and read with `memmap`, so data loading is never the bottleneck.
- **Managing a training run**: checkpoints, resuming after a crash, and tracking validation loss over hours.
- **GPU realities**: free tiers (Colab, Kaggle) vs renting (Runpod, Lambda), and what a few dollars buys.
- **Sampling strategies**: temperature and top-k, the knobs that turn one model into many voices.
- **Pretraining vs finetuning**: what those words actually mean, as groundwork for lesson 36.

## Before starting

Check that these are in place: the working GPT from [lesson 31](31-attention-build-gpt.md), the BPE tokenizer from [lesson 32](32-tokenization-bpe.md), and comfort with the training diagnostics from [lesson 25](../phase-3-deep-learning/25-training-deep-nets.md).

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tiktoken matplotlib
mkdir -p work/33-train-your-own-gpt
cd work/33-train-your-own-gpt
```

The first `pip` line installs PyTorch from its own package index, as in lesson 24 (if torch is already installed, even a GPU build, it changes nothing). The other packages need the separate second line, because that index does not carry tiktoken or matplotlib. On an Intel Mac, current PyTorch cannot be installed at all, so the first line fails there: do the parts of this lesson that need torch in a free Google Colab notebook, as lesson 01 suggested (PyTorch comes preinstalled there).

Clone nanoGPT next to the course repo, outside it (a clone inside the repo would be picked up by `git add .` as a broken embedded repository). The point is to read it, not to run it blindly:

```bash
git clone https://github.com/karpathy/nanoGPT.git ~/ml/nanoGPT
```

The lesson also needs a Google account for Colab (free GPU) or a Kaggle account (free GPU hours per week). Both are free. Pick one now, and open a notebook with a GPU attached to confirm that it works (`Runtime → Change runtime type → GPU` in Colab).

**Keep the big files out of git.** A `ckpt.pt` that includes the optimizer state takes about 12 bytes per parameter, so even a 10M-parameter model's checkpoint is over 120 MB, and GitHub rejects any file over 100 MB. The token files and the raw corpus do not belong in the fork either, and `prepare.py` can rebuild the data at any time. In this folder, run `printf '*.pt\n*.bin\nraw/\n' > .gitignore`. Checkpoint: `git check-ignore ckpt.pt train.bin raw/` prints all three names.

No dataset download is prescribed: choosing the corpus **is** Project 1.

## Project 1 — Corpus of choice

**Goal**: Assemble 5–50MB of raw text that is a genuine favorite, and turn it into two memory-mapped binary token files. A model trained on deeply familiar data is one that can actually be judged.

**Milestones**

- [ ] Pick the corpus. Ideas: everything by one author from Project Gutenberg (gutenberg.org, with plain-text downloads and no account needed), song lyrics, a Wikipedia subset, favorite out-of-copyright books, or a pile of source code. Save the raw files in a `raw/` folder inside the work folder, and target at least 5MB of raw text (check with `du -sh raw`). Below that, a 10M-param model will memorize rather than learn. Checkpoint: `cat raw/* | wc -c` shows at least 5,000,000 bytes.
- [ ] Clean it lightly in a `prepare.py` script: strip boilerplate (Gutenberg files have legal headers and footers; remove them in code, not by hand), normalize odd whitespace, and concatenate everything into one big string. Print the first and last 500 characters to check it by eye. Checkpoint: no license text or HTML junk is visible at either end.
- [ ] Tokenize the whole corpus **once**. Use the lesson-32 BPE tokenizer, or `tiktoken` (OpenAI's fast tokenizer library) with the `gpt2` encoding. Using tiktoken is not cheating: lesson 32 built a BPE tokenizer by hand, so it is already clear exactly what tiktoken does. Time it. Checkpoint: the tokenizer returns a list/array of token IDs, and tokens ≈ bytes ÷ 3 to 4 for English text with tiktoken's `gpt2` encoding (about bytes ÷ 2 with a lesson-32 tokenizer of around 1024 tokens).
- [ ] Split 90/10 into train and validation, then write each as a raw binary file of `uint16` values (every GPT-2 token ID fits in 16 bits, which halves the file size compared with `uint32`):

```python
import numpy as np
ids = np.array(train_ids, dtype=np.uint16)
ids.tofile("train.bin")
```

- [ ] Write the loading side using `np.memmap`, a NumPy array that lives on disk and is read lazily, so a 50MB (or 50GB) dataset costs almost no RAM. Reopen the memmap inside the batch function to avoid a slow memory leak on some systems. Checkpoint: decoding tokens 1000–1100 from `train.bin` reproduces a readable slice of the corpus exactly.
- [ ] Record the stats in a README: raw size, token count, vocab used, train/val split. Checkpoint: state the dataset size in tokens (e.g. "11.2M tokens"). From now on, measure data in tokens, as professionals do.

<details><summary>Hints</summary>

- Gutenberg headers all contain the phrase `*** START OF` and footers `*** END OF`. Find those markers and slice between them.
- `tiktoken.get_encoding("gpt2")` provides `.encode(text)` and `.decode(ids)`. Encode in chunks of a few MB if memory gets tight.
- To read a batch from a memmap: pick random offsets `ix`, then stack `data[i : i+block_size]` slices and convert with `torch.from_numpy(chunk.astype(np.int64))`.
- If the lesson-32 BPE tokenizer is too slow for the full corpus, that is a real lesson about production code. Note it, and switch to tiktoken without guilt.

</details>

**Definition of done**: `train.bin` and `val.bin` exist, a round-trip decode of any slice is correct, and `prepare.py` rebuilds both files from raw text in one command.

## Project 2 — The run

**Goal**: Scale the lesson-31 GPT to ~10M parameters and manage a real multi-hour training run on a free cloud GPU, with schedules, checkpoints, logging, and generations that are unmistakably in the style of the corpus.

**Milestones**

- [ ] Read `model.py` and `train.py` in the nanoGPT repo, side by side with the lesson-31 code. It is the same architecture that lesson 31 built, and that is the point. List 5 things nanoGPT does that the lesson-31 code does not (weight decay grouping, gradient clipping, mixed precision, gradient accumulation and a cosine schedule are all in there). It is fine to keep improving the lesson-31 code or to adapt nanoGPT. Reading nanoGPT counts as part of the lesson either way, but type out anything borrowed from it. Checkpoint: explain in one sentence each what the 5 items are for.
- [ ] Scale the config to roughly 10M parameters: try 6 layers, 6 heads, embedding size 384, block size 256, with the tokenizer's vocab size. Count parameters with `sum(p.numel() for p in model.parameters())`. The token embedding and the output layer each hold vocab × embedding-size numbers, so the vocab sets much of the total: this config is about 11.5M with a 1024-token lesson-32 vocab, but about 49M with tiktoken's 50,257-token `gpt2` vocab. With the `gpt2` vocab, use 4 layers and embedding size 192 instead, and make the output layer share the token embedding's weight matrix (weight tying, as nanoGPT does in `model.py`), for about 11.5M. Checkpoint: parameter count between 8M and 15M.
- [ ] Add a learning-rate schedule: **warmup** (start near zero and ramp up over the first few hundred steps, so early noisy gradients do not wreck the fresh weights) then **cosine decay** (glide smoothly down to ~10% of peak, so late training takes gentle steps). Write `get_lr(step)` as a pure function and plot it with matplotlib before training. Checkpoint: the plot shows a sharp ramp then a long smooth cosine curve down.
- [ ] Add checkpointing: every N steps, evaluate mean loss on `val.bin`, and if it is the best so far, `torch.save` a dict of model weights + optimizer state + step + config to `ckpt.pt`. Add a `--resume` path that loads all of it and continues. Test resume locally on CPU with a tiny config before spending GPU time. Checkpoint: kill training with Ctrl-C, resume, and the loss continues from where it stopped instead of restarting high.
- [ ] Move to the GPU. Pick the device in one line, `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`, and move the model and each batch with `.to(device)`, so that one script runs everywhere: on the NVIDIA GPU that Colab or Kaggle lends out (`cuda`), on an Apple Silicon Mac's built-in GPU (`mps`, macOS 14 or later; handy for faster local tests), or on the CPU. Zip the work folder, upload it to Colab or Kaggle, and confirm that `torch.cuda.is_available()` is `True`. Enable **mixed precision**: computing mostly in 16-bit floats, which roughly doubles speed on modern GPUs with negligible quality loss. It is switched on with `torch.autocast`, and nanoGPT shows exactly how, including the `GradScaler` needed for float16. On Colab's free T4 (and other GPUs older than the A100), use float16 with the `GradScaler`: nanoGPT's `torch.cuda.is_bf16_supported()` test returns `True` there because bfloat16 can be emulated, and emulated bfloat16 runs no faster than full precision. `torch.cuda.is_bf16_supported(including_emulation=False)` reports real hardware support. Checkpoint: a short timed run shows a clear speedup vs full precision.
- [ ] Measure throughput before the real run: tokens/second = batch × block_size × steps ÷ seconds. If the GPU wants bigger batches than memory allows, use **gradient accumulation**: run several small forward/backward passes, summing the gradients, and step the optimizer once. This simulates a large batch. Checkpoint: predict the length of the full run to within ~20% ("about 3 hours for 20k steps").
- [ ] The run itself: train for a few hours, logging `(step, train_loss, val_loss, lr)` to a CSV, generating a small sample every ~1000 steps to watch the style emerge, and downloading `ckpt.pt` periodically (free tiers disconnect, and the resume code is the insurance against that). Checkpoint: val loss ends far below its starting value near `ln(vocab_size)` (about 10.8 for GPT-2's 50,257 tokens), and the loss curve plot is smooth with no divergence spikes. Do not compare the number with lesson 31's 1.48: that loss is per character, this one is per token, and per-token losses from different vocabularies are not comparable (lesson 32).
- [ ] Sample properly. Implement **temperature** (divide logits by T before softmax: below 1.0 makes the model conservative and repetitive, above 1.0 makes it wild) and **top-k** (only sample among the k most likely tokens, cutting off the long tail of nonsense). Put 3 generations (T=0.8, T=1.0, T=1.2 or with/without top-k) in the README with the loss curve. Checkpoint: someone who knows the corpus would identify its style from the samples in 5 seconds. Unmistakably in-style generations are the bar.
- [ ] Bridge to lesson 36: the run just completed is **pretraining**, which means teaching a model a domain from scratch by next-token prediction on raw text. **Finetuning** starts from someone else's pretrained checkpoint and nudges it with a small dataset toward a task. Write 3 sentences in the README on why finetuning is usually cheaper. Checkpoint: say what GPT's "P" stands for and why it matters.

<details><summary>Hints</summary>

- Cosine schedule shape: for step `t` past warmup, `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))` where `progress` goes 0→1 over the decay phase. Peak LR around 1e-3 with AdamW is a sensible start at this scale.
- Checkpoint dict essentials: `model.state_dict()`, `optimizer.state_dict()`, `step`, and the config. Forgetting the optimizer state is the classic bug: resume "works", but the loss jumps because AdamW lost its momentum buffers.
- On Colab, `from google.colab import files` handles small uploads/downloads; mounting Google Drive is steadier for checkpoints on long runs.
- If loss goes to NaN: lower the peak LR first, confirm that gradient clipping (`clip_grad_norm_` at 1.0) runs *before* `optimizer.step()`, and check that the warmup actually starts near zero.

</details>

**Definition of done**: a completed multi-hour GPU run with a saved best checkpoint, a plotted val-loss curve, and a README containing throughput numbers plus 3 in-style generations at different temperatures.

## Project 3 — Stretch summit: reproduce GPT-2 (124M)

**Goal**: Understand (and, if the budget allows, carry out) the full reproduction of GPT-2 124M, the real model OpenAI shipped in 2019. Watching and understanding the video **completes this lesson**; the run itself is optional.

**Milestones**

- [ ] Watch "Let's reproduce GPT-2 (124M)" from the [Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). It is long, so take it in several sittings, with notes open. Checkpoint: list 5 optimizations from the video that Project 2 did not use (candidates: kernel fusion via `torch.compile`, flash attention, vocab-size padding to a power-of-2-friendly number, distributed training across GPUs, weight tying).
- [ ] Read about [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), the open recreation of GPT-2's training data. Compare: the Project 2 corpus was millions of tokens; this is ~9 billion. Checkpoint: explain in plain words why scaling data and parameters *together* matters.
- [ ] Decide on the run with a written cost estimate. Realistic paths: ~$10–40 on a rented multi-GPU machine (8×A100 runs roughly $10–15/hour, and the video's own code trains on 10B tokens of FineWeb-Edu in one to two hours on such a machine; Runpod and Lambda are the usual choices, but note that Colab's free tier was enough for everything up to now), or a much longer single-GPU slog. Before launching, do the arithmetic: expected cost = hourly rate × hours predicted from a 100-step throughput test (see the hints). Never trust a duration that has not been measured on the actual machine. **No budget means skip with zero guilt**: the understanding is the summit, not the receipt. Checkpoint: a paragraph in the README: run or skip, and why.
- [ ] (Optional, budget willing) Do it with the video's code ([build-nanogpt](https://github.com/karpathy/build-nanogpt): `fineweb.py` to prepare the FineWeb-Edu data, then `train_gpt2.py`) on the rented machine, applying everything from Project 2: measure throughput first, checkpoint to persistent storage, watch the val curve, and **set a spending alarm before launch**. nanoGPT's own `config/train_gpt2.py` is a different, much longer run (300B tokens of OpenWebText, about 4 days on 8×A100, roughly $1,000 or more at the rates above), far outside this budget. Checkpoint: val loss on the FineWeb-Edu validation shard at or below the OpenAI GPT-2 (124M) baseline shown in the video (about 3.29).
- [ ] **MILESTONE: the Karpathy arc is complete.** Look back at everything built since lesson 22: **an autograd engine and an MLP from scratch, convolution in NumPy and a trained CNN, an LSTM language model, a BPE tokenizer from scratch, and a GPT trained on self-chosen data.** Together they show, end to end, how the engines of modern AI are built. Write the look-back list in `PROGRESS.md` and read it out loud.

<details><summary>Hints</summary>

- The video builds the code live; keep nanoGPT open in VS Code and trace each change Karpathy makes to the corresponding file.
- If renting: create the machine, run a 100-step throughput test, multiply the predicted hours by the hourly rate to get the expected bill, and only then commit to the full run. Ten minutes of testing protects hours of budget.
- Rented machines usually bill until they are *terminated*, not merely stopped. Check the provider's billing page before walking away.

</details>

**Definition of done**: video watched with notes, the 5-optimizations list and run-or-skip paragraph written, and the milestone look-back recorded in `PROGRESS.md`. The run itself is a bonus, never a requirement.

## Stretch goals

- Train two model sizes (e.g. 3M and 10M) on the same data for the same wall-clock time and plot both val curves. This is a first taste of scaling laws.
- Implement top-p (nucleus) sampling, which samples from the smallest set of tokens whose probabilities sum to p, and compare it against top-k on the trained model.
- Add `torch.compile(model)` and measure the throughput change on the GPU.
- Build a tiny CLI: `python generate.py --prompt "..." --temperature 0.8 --top_k 50` that loads the best checkpoint and streams tokens as they are generated.

## Getting unstuck

- **Loss stuck near `ln(vocab_size)`** (~10.8 for a 50k vocab): that is the loss for uniform guessing. The model is learning nothing. Suspect the data pipeline: decode a training batch and *read it*. If it is garbage, the bug is in Project 1, not in the model.
- **Great train loss, flat val loss**: overfitting. The corpus is too small for the model. Shrink the model or grow the data.
- **Colab disconnects mid-run**: this is normal, not a failure, and it is exactly why resume support was built. Checkpoint to Drive and continue.
- **CUDA out of memory** (or `MPS backend out of memory` on an Apple Silicon Mac): reduce batch size and compensate with gradient accumulation; if needed, reduce block size.
- Standing advice: read the error traceback bottom-up (the last line is the actual error); print shapes and dtypes at every boundary; ask an AI assistant for a **hint**, not a solution; and type all code by hand, because copy-paste teaches nothing.

## Resources

- [Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — "Let's reproduce GPT-2 (124M)" is the finale and the heart of Project 3.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the professional reference codebase; read it, compare it to the lesson-31 code, and borrow from it deliberately.
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — the open recreation of GPT-2's training data; the dataset card explains what real pretraining data looks like.
- [PyTorch docs](https://pytorch.org) — look up `torch.autocast`, `GradScaler`, and `clip_grad_norm_` when wiring in mixed precision.
- Project Gutenberg (gutenberg.org) — tens of thousands of public-domain books as plain text, no account needed; the classic corpus source.

## Skills unlocked

- [ ] I can prepare a large text dataset: tokenize once, store as binary, and load it with memmap without exhausting RAM.
- [ ] I can implement a warmup + cosine learning-rate schedule and explain why each phase exists.
- [ ] I can checkpoint and resume a training run, including optimizer state, and have proven it by killing and resuming a run.
- [ ] I can run training on a cloud GPU, use mixed precision and gradient accumulation, and predict a run's duration from measured throughput.
- [ ] I can control generation with temperature and top-k and explain what each does to the token distribution.
- [ ] I can read a professional codebase like nanoGPT and map it onto code I wrote myself.
- [ ] I can explain pretraining vs finetuning and when each is the right tool.
- [ ] I have built, from scratch: an autograd engine, an MLP, convolution, a tokenizer, and a trained GPT, and I have trained a CNN and an LSTM.

## Next up

With the engine built, the next lesson turns to driving the giant ones: [34 · Using LLMs: APIs, Prompting and Structured Output](../phase-5-llms/34-llm-apis-prompting.md).
