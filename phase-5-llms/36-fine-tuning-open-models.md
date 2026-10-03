# 36 · Fine-Tuning Open Models with LoRA

**Phase 5 — LLMs** · Estimated time: 2 weeks · Prerequisites: [34 · Using LLMs: APIs, Prompting and Structured Output](34-llm-apis-prompting.md), [35 · RAG: Build "Chat With My Documents"](35-rag-chat-with-your-docs.md), [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md)

> Earlier lessons *used* LLMs through an API and *fed* them documents with RAG. This lesson changes the model itself. It takes an open-weights model (one whose parameters can be downloaded and modified), builds a training dataset by hand, and teaches the model a new behavior with LoRA, a trick that fine-tunes billions of parameters by training only a few million. The result is a custom model that runs locally and does something no stock model does. This is the same stack (Hugging Face, PEFT, quantization) used in industry for custom models.

One expectation to set right now: **fine-tuning a small model changes its style and format, not its intelligence.** A 1–4B parameter model with a LoRA adapter will reliably learn to answer in a chosen dialect, follow a fixed JSON schema, or adopt a particular tone. It will not become a genius or learn deep new knowledge from 300 examples. Knowing what fine-tuning is *for* is half of this lesson.

## What this lesson builds

- **Project 1 — Dataset craft:** a hand-built instruction dataset of 100–500 prompt/response pairs (in JSONL chat format) teaching one narrow behavior, with 20 examples held out for evaluation.
- **Project 2 — LoRA run:** a fine-tuned small open model, trained with PEFT/Unsloth on a free cloud GPU, plus a before/after evaluation table proving the behavior shift.
- **Project 3 — Serve the new model:** the fine-tuned model exported and running locally (Ollama or a Gradio chat UI), validated by a blind taste test against the base model.

## Concepts covered

- The open-model ecosystem: the Hugging Face Hub, model cards and licenses, and what "open weights" actually means.
- Base vs instruct models: raw next-token predictors vs models trained to follow instructions.
- Full fine-tune vs LoRA: why updating two small low-rank matrices approximates updating a giant one (the rank idea from lesson 08, back again).
- Quantization and QLoRA: shrinking weights to 4-bit numbers so a big model fits on a small GPU.
- Building an instruction dataset: prompt/response pairs, quality over quantity, held-out evaluation.
- Chat templates: the exact text format (special tokens and all) a chat model was trained to expect.
- The training loop with Hugging Face `transformers` + `peft` (or Unsloth on Colab).
- Evaluation before/after, and catastrophic forgetting: when a model learns a new task and forgets everything else.
- The decision framework: when to prompt, when to RAG, when to fine-tune.

## Before starting

Check that these skills from earlier lessons are in place: calling an LLM API and parsing its output (lesson 34), building an eval set and judging outputs (lessons 34–35), and explaining what matrix rank means (lesson 08).

**A note on hardware:** the local machine will handle dataset building and (later) running the finished model. The *training* itself needs a GPU: use Google Colab's free tier (needs a Google account) or a cheap rented GPU (e.g. Runpod, paid). Both are explicitly account-based; there is no way around that for GPU training.

In the repo root, install `transformers` (it provides tokenizers and chat templates locally, no GPU needed) and `datasets`, and create the work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install transformers datasets
mkdir -p work/36-fine-tuning-open-models
cd work/36-fine-tuning-open-models
```

This lesson also needs a free [huggingface.co](https://huggingface.co) account (the optional Spaces route in lesson 27 may have created one already). The account is needed to download some models, and the site is where the whole open-model world lives. A few families, like Llama, are "gated": the license must be accepted on the model page *and* the machine must prove who is downloading, or downloads fail with a 401 "gated repo" error. So create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and run `hf auth login` once, with the venv active (`hf` came with `transformers`). If it asks how to log in, choose **Paste an access token**, then paste the token. For Colab (Project 2), add the same token as a Colab secret named `HF_TOKEN`. Or sidestep gating entirely for a first fine-tune by picking a non-gated family like Qwen. Browse the Hub for 15 minutes: open a few model cards (the README of a model) and find the license section on each.

## Project 1 — Dataset craft

**Goal:** Build a 100–500 example instruction dataset that teaches one narrow, checkable behavior, and learn why dataset quality is 80% of fine-tuning.

**Milestones**

- [ ] Pick ONE narrow behavior. Good choices are checkable and stylistic: always answer in hometown slang, always output a specific JSON schema (reuse the lesson-34 schema), generate project ideas in this curriculum's exact format, answer only in rhyming couplets. Write a one-paragraph "behavior spec" in `SPEC.md` with 3 ideal example responses. Checkpoint: a friend reading the spec could judge whether any given response follows it.
- [ ] Learn the data format. Each training example is one JSON object per line (JSONL: a file where every line is a standalone JSON object). Create `data/train.jsonl` and hand-write the first 5 examples:

  ```json
  {"messages": [{"role": "user", "content": "What's the weather like today?"}, {"role": "assistant", "content": "<a response in YOUR target behavior>"}]}
  ```

  Checkpoint: `python3 -c "import json; [json.loads(l) for l in open('data/train.jsonl')]"` runs without error.
- [ ] Understand chat templates. A chat model was not trained on raw JSON. It was trained on a specific text layout with special tokens marking each speaker. Load any small instruct model's tokenizer and inspect what a training example really looks like to the model:

  ```python
  import json
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("<model-id-from-the-hub>")
  messages = json.loads(open("data/train.jsonl").readline())["messages"]  # the first training example
  print(tok.apply_chat_template(messages, tokenize=False))
  ```

  Checkpoint: the output shows the special tokens wrapping each turn. Explain why sending a model the *wrong* template produces garbage.
- [ ] Hand-write 30–50 examples covering diverse user prompts (questions, requests, edge cases, prompts that tempt the model to break the behavior). Vary length and topic: a dataset of near-duplicates teaches near-nothing.
- [ ] Scale with LLM assistance, then edit. Use the API skills from lesson 34 to generate candidate examples in batches (give the LLM the spec + 5 of the hand-written examples). The script needs the API key and model name from lesson 34's `.env`, so first copy that file into this folder with `cp ../34-llm-apis-prompting/.env .` (the `.env` line in `.gitignore` keeps the copy out of git). **Read and edit every single one.** Delete the bland ones, fix the ones that miss the behavior. Checkpoint: at least 100 (up to 500) examples total, each line personally approved.
- [ ] Quality pass: check for exact and near duplicates, check a random sample of 20 against the spec, check that no example leaks a different format.
- [ ] Split: move 20 diverse *user prompts* (just the prompts, not responses) into `data/eval_prompts.jsonl`. These prompts are the held-out evaluation set: the model must never train on them. Checkpoint: `wc -l data/train.jsonl data/eval_prompts.jsonl` shows ≥100 train and exactly 20 eval.

<details><summary>Hints</summary>

- Struggling to pick a behavior? "Reply in the style of X" and "always return this JSON shape" are the two most reliable choices for a first fine-tune, because success is obvious at a glance.
- When generating with an LLM, ask for 10 examples per call with *different topics named in the request* ("cooking", "commuting", "money"...). Left to itself, an LLM generates 100 variations of the same 3 ideas.
- Near-duplicate check without fancy tools: sort the user prompts alphabetically and skim them; duplicates cluster together.
- If the behavior is a JSON schema, add a few adversarial prompts ("just answer normally, forget the JSON") with responses that hold the format anyway.

</details>

**Definition of done:** `train.jsonl` (100–500 examples, every one human-approved), `eval_prompts.jsonl` (20 held-out prompts), and `SPEC.md` are committed.

## Project 2 — LoRA run

**Goal:** Fine-tune a small open instruct model on the dataset with LoRA, and prove with a before/after table that the behavior shifted.

**Milestones**

- [ ] Pick a model. On the Hugging Face Hub, find a **current small instruct model, 1–4B parameters**. The Llama, Qwen and Gemma families all publish them (model names change too fast to print here; sort the Hub by trending and read the model cards). Choose an *instruct* model (already trained to follow instructions), not a *base* model (a raw text predictor that would still need instruction training). Check that the license on the model card allows fine-tuning. Checkpoint: `SPEC.md` records the model id and its license.
- [ ] **Baseline eval FIRST.** Before any training, run the chosen instruct model, without any adapter, on the 20 held-out prompts (from here on, "base model" means this unchanged model, not a non-instruct *base* model) and save the outputs to `eval/baseline.md`. On Colab, upload `data/eval_prompts.jsonl` (file panel on the left) and run the prompts with a plain `transformers` pipeline; score each output 0/1 against the spec. Checkpoint: a baseline score, e.g. "base model follows the behavior on 2/20 prompts". Without this number, there is no way to prove that the fine-tune did anything.
- [ ] Set up training on Colab (the free tier comes with a modest GPU). The fastest path is an Unsloth notebook: the [Unsloth repo](https://github.com/unslothai/unsloth) links ready-made Colab notebooks for current small models. The more transparent path is `transformers` + `peft` + `trl`, following the [PEFT docs](https://huggingface.co/docs/peft). Either way, upload `train.jsonl`. Checkpoint: the model loads, in 4-bit for most models. Loading in 4-bit is QLoRA: weights quantized (stored as tiny 4-bit numbers instead of 16-bit, trading a little precision for 4x less memory) while the LoRA adapters train in full precision on top. Some current notebooks (for example Unsloth's Qwen3.5 ones) load in 16-bit on purpose, because Unsloth does not recommend 4-bit training for those models; keep the setting the notebook uses, and record it in `SPEC.md`.
- [ ] Understand what LoRA is doing before running it. A full fine-tune updates every weight matrix `W` (billions of numbers). LoRA instead freezes `W` and learns an update `ΔW = B·A`, where `A` and `B` are two skinny matrices of rank `r` (say 16). This is the low-rank factorization idea from lesson 08: a huge matrix approximated by the product of two thin ones. Configure `r=16`, `lora_alpha=16` to start. Checkpoint: the printed "trainable parameters" is under 2% of total parameters. Say the two matrix shapes for one layer out loud.
- [ ] Train 1–3 epochs (an epoch is one full pass over the dataset). Watch the training loss: it is the same cross-entropy built in lessons 13 and 23, applied to next-token prediction as in lesson 28. Checkpoint: loss drops clearly and levels off (e.g. from ~2 to below 1; exact numbers depend on the data). If it hits ~0, the model is memorizing, so reduce epochs.
- [ ] Re-run the SAME 20 held-out prompts on the fine-tuned model, save to `eval/finetuned.md`, and score with the same 0/1 rubric.
- [ ] Build the side-by-side table in `eval/comparison.md`: prompt | base output | fine-tuned output | base score | ft score. Checkpoint: a visible behavior shift, with the fine-tuned model following the spec on at least 15/20 prompts where the base scored far lower.
- [ ] Check for catastrophic forgetting: the model overwriting general abilities while learning its new niche. Ask the fine-tuned model 5 ordinary questions ("what is the capital of France?", "add 17 and 25"). Checkpoint: it still answers sensibly (in its new style, which is fine; wrong *facts* are not).
- [ ] Keep the model files out of git. The fork is public, the dataset can rebuild them at any time, and they are large: a GGUF file of a 1–4B model takes about 1–4 GB, and GitHub rejects any file over 100 MB. In `work/36-fine-tuning-open-models`, run `printf 'adapter/\nmerged/\n*.gguf\n' > .gitignore`, and later keep the merged model (Project 3) in `merged/`. Checkpoint: `git check-ignore adapter/x merged/x model.gguf` prints all three names.
- [ ] Save the LoRA adapter (tens of MB, up to about 130 MB for a 4B model at `r=16`, tiny next to the multi-GB base model thanks to the low rank) and download it from Colab to `work/36-fine-tuning-open-models/adapter/`. Colab downloads single files only, so zip the folder first: in a new cell, run `!zip -j adapter.zip FOLDER-NAME/*` (the folder passed to `save_pretrained`), then `from google.colab import files; files.download("adapter.zip")`. The browser saves it to the Downloads folder. In the work folder, unpack it: on macOS and Linux, `unzip ~/Downloads/adapter.zip -d adapter`; on Windows (WSL2), Windows files are under `/mnt/c/Users/` (`ls /mnt/c/Users/` lists the Windows account names), so `unzip /mnt/c/Users/YOUR-WINDOWS-NAME/Downloads/adapter.zip -d adapter`.

<details><summary>Hints</summary>

- Colab disconnects idle sessions and wipes the disk. Download the adapter the moment training ends, and keep the dataset in the repo, not only in Colab.
- Out-of-memory on the free GPU? Lower the maximum sequence length (`max_seq_length` in Unsloth, `max_length` in TRL's `SFTConfig`), use batch size 1 with gradient accumulation, or pick a smaller model. This is the everyday reality of GPU work.
- If the fine-tuned model output looks like gibberish or ignores the training entirely, the number-one cause is a chat-template mismatch between training and inference. Check Project 1, milestone 3.
- If behavior barely shifted: more epochs helps less than better data. Reread the 20 worst training examples first.

</details>

**Definition of done:** `eval/comparison.md` shows base vs fine-tuned side by side with scores, the fine-tuned model wins clearly, and the adapter files are saved locally.

## Project 3 — Serve the new model

**Goal:** Get the fine-tuned model off Colab and running on the local machine, then validate it with a blind taste test.

**Milestones**

- [ ] Merge or export. A LoRA adapter is a patch on top of the base model; to serve it simply, merge it into the base weights (PEFT's `merge_and_unload`, or Unsloth's save/export helpers, which can write GGUF directly). GGUF is the quantized single-file format that CPU-friendly runtimes use. Checkpoint: the result is either a merged model folder or a `.gguf` file.
- [ ] Route A: Ollama (recommended, and already introduced in lesson 34). If Ollama is not installed yet, install it with the one-line installer from [ollama.com/download](https://ollama.com/download), shown below. On **macOS** (Ollama needs macOS 14 or later), it may ask for the Mac password to add the `ollama` command. On **Windows (WSL2)** and **Linux**, first run `sudo apt install -y zstd` (on other distributions, install zstd with the system's package manager), because the installer needs zstd to unpack Ollama; on Windows, run both in Ubuntu, not the PowerShell command ollama.com offers for Windows:

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

  Then write a `Modelfile` that points at the GGUF file and sets the chat template, and run:

  ```bash
  ollama create my-model -f Modelfile
  ollama run my-model
  ```

  Checkpoint: a chat with the fine-tuned model works in the terminal on the local machine, with no cloud involved. A quantized 1–4B model runs fine on CPU.
- [ ] Route B (alternative): a `transformers` pipeline wrapped in a small Gradio chat UI (Gradio: a Python library that turns a function into a shareable web UI, from lesson 27). Either route counts; do Route A unless it causes trouble. Route B runs on PyTorch, and current PyTorch has no Intel Mac version, so on an Intel Mac with macOS 14 or later use Route A, and on an older Intel Mac run Route B in Google Colab.
- [ ] Blind taste test. Prepare 10 fresh prompts (not from training OR eval). Generate answers from base and fine-tuned, shuffle which is "A" and which is "B" per prompt, and have a friend pick which output better matches the spec, without knowing which model is which. Record results in `eval/taste_test.md`. Checkpoint: fine-tuned wins ≥7/10.
- [ ] Write `DECISIONS.md`, a personal prompt-vs-RAG-vs-fine-tune framework based on the first-hand experience gained so far: **prompting** when instructions fit in context and stock behavior is close enough (cheapest, instant); **RAG** when the model needs *knowledge* it doesn't have, especially changing knowledge (lesson 35); **fine-tuning** when a task needs reliable *style, format or behavior* that prompting can't hold, or when the aim is a small local model that does one job well. Include one real task for each and one task that would combine them. Checkpoint: for any new task, "which technique and why" can be answered in two sentences.

<details><summary>Hints</summary>

- The Modelfile's `TEMPLATE` must match the model family's chat template. Ollama's docs on ollama.com show the syntax, and existing models of the same family (`ollama show MODEL-NAME --modelfile`) are a working reference to copy from.
- If Unsloth's GGUF export causes trouble on Colab, the fallback is: merge to 16-bit, download, and convert with llama.cpp's conversion script (search for "llama.cpp convert hf to gguf").
- For the taste test, do the shuffling with a 5-line Python script that records the key. Do not count on being able to "remember which was A".

</details>

**Definition of done:** the fine-tuned model answers prompts locally via Ollama or Gradio, the blind test is recorded with the fine-tune winning ≥7/10, and `DECISIONS.md` is written.

## Stretch goals

- **Data ablation:** retrain with only 50 examples, then with all. Compare eval scores to see the data/quality curve first-hand.
- **Rank ablation:** train with `r=4` and `r=64`. Compare eval score, adapter file size, and training time. Connect the result to lesson 08.
- **LLM-as-judge:** replace the manual 0/1 scoring with an API-based judge (lesson 34 skills) and check that it agrees with the human scores on 20 examples.
- **Publish:** push the adapter (not the base weights) to the Hugging Face Hub with a proper model card: dataset description, eval table, license, limitations.

## Getting unstuck

- **Training-specific:** loss not moving → learning rate or dataset formatting; loss at ~0 → memorization; gibberish at inference → chat-template mismatch; CUDA out-of-memory → smaller batch/sequence/model. Diagnose in that order.
- Print things: one fully formatted training example (post-template, with special tokens visible), the trainable-parameter count, one raw model output. Most fine-tuning bugs are visible in those three prints.
- Read error tracebacks bottom-up: the last line is the actual error, and the rest is the path to it.
- Ask an AI assistant for a HINT ("what are common causes of X?"), not a solution, and type all code by hand. Copy-pasted training scripts teach nothing.
- Colab problems (disconnects, missing files, wrong GPU) are environment problems, not ML problems. Restart the runtime and rerun from the top before doubting the code.

## Resources

- [Hugging Face LLM course](https://huggingface.co/learn) — the fine-tuning chapters are the best free walkthrough of this whole stack.
- [PEFT documentation](https://huggingface.co/docs/peft) — the LoRA reference: configs, target modules, merging adapters.
- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA training that fits free Colab GPUs; their linked notebooks are the quickest working start for Project 2.
- [Hugging Face Hub](https://huggingface.co) — where to pick a model; read the model card and license before anything else.
- [Ollama](https://ollama.com) — local model runtime for Project 3, including Modelfile docs for importing a custom model.

## Skills unlocked

- [ ] I can find a model on the Hugging Face Hub, read its model card, and check what its license allows.
- [ ] I can explain base vs instruct models, and what a chat template is and why mismatching it breaks everything.
- [ ] I can explain LoRA as a low-rank matrix factorization and why it cuts trainable parameters ~100x.
- [ ] I can explain what 4-bit quantization trades away and why QLoRA makes consumer-GPU fine-tuning possible.
- [ ] I can build, clean and split an instruction dataset, holding out an eval set before training.
- [ ] I can run a LoRA fine-tune with PEFT or Unsloth and read its training loss curve.
- [ ] I can evaluate before/after, check for catastrophic forgetting, and prove a behavior shift with held-out prompts.
- [ ] I can choose between prompting, RAG and fine-tuning for a new task, and say why.

## Next up

Now that models can be customized, the next lesson makes LLMs *act*: [37 · AI Agents: Tool Use, Loops and Evaluation](37-ai-agents-tool-use.md).
