# 36 · Fine-Tuning Open Models with LoRA

**Phase 5 — LLMs** · Estimated time: 2 weeks · Prerequisites: [34 · Using LLMs: APIs, Prompting and Structured Output](34-llm-apis-prompting.md), [35 · RAG: Build "Chat With My Documents"](35-rag-chat-with-your-docs.md), [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md)

> So far you have *used* LLMs through an API and *fed* them documents with RAG. This lesson you change the model itself: you take an open-weights model (one whose parameters you can download and modify), build your own training dataset, and teach the model a new behavior with LoRA — a trick that fine-tunes billions of parameters by training only a few million. By the end you will have a model that is genuinely *yours*, running on your own machine, doing something no stock model does. This is the same stack (Hugging Face, PEFT, quantization) used in industry for custom models.

One expectation to set right now: **fine-tuning a small model changes its style and format, not its intelligence.** A 1–4B parameter model with a LoRA adapter will reliably learn to answer in your dialect, follow your JSON schema, or adopt your tone — it will not become a genius or learn deep new knowledge from 300 examples. Knowing what fine-tuning is *for* is half of this lesson.

## What you will build

- **Project 1 — Dataset craft:** a hand-built instruction dataset of 100–500 prompt/response pairs (in JSONL chat format) teaching one narrow behavior, with 20 examples held out for evaluation.
- **Project 2 — LoRA run:** a fine-tuned small open model, trained with PEFT/Unsloth on a free cloud GPU, plus a before/after evaluation table proving the behavior shift.
- **Project 3 — Serve your creation:** your model exported and running locally (Ollama or a Gradio chat UI), validated by a blind taste test against the base model.

## Concepts you will learn by doing

- The open-model ecosystem: the Hugging Face Hub, model cards, and licenses — what "open weights" actually means.
- Base vs instruct models: raw next-token predictors vs models trained to follow instructions.
- Full fine-tune vs LoRA: why updating two small low-rank matrices approximates updating a giant one (the rank idea from lesson 08, back again).
- Quantization and QLoRA: shrinking weights to 4-bit numbers so a big model fits on a small GPU.
- Building an instruction dataset: prompt/response pairs, quality over quantity, held-out evaluation.
- Chat templates: the exact text format (special tokens and all) a chat model was trained to expect.
- The training loop with Hugging Face `transformers` + `peft` (or Unsloth on Colab).
- Evaluation before/after, and catastrophic forgetting — when a model learns your task and forgets everything else.
- The decision framework: when to prompt, when to RAG, when to fine-tune.

## Before you start

Check you can do these (from earlier lessons): call an LLM API and parse its output (lesson 34), build an eval set and judge outputs (lessons 34–35), explain what matrix rank means (lesson 08).

**A note on hardware:** your own machine will handle dataset building and (later) running the finished model. The *training* itself needs a GPU — you will use Google Colab's free tier (needs a Google account) or a cheap rented GPU (e.g. Runpod, paid). Both are explicitly account-based; there is no way around that for GPU training.

In your repo root, install `transformers` (it gives you tokenizers and chat templates locally, no GPU needed) and `datasets`, and create your work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install transformers datasets
mkdir -p work/36-fine-tuning-open-models
cd work/36-fine-tuning-open-models
```

Also create a free account at [huggingface.co](https://huggingface.co) — you need it to download some models, and it is where the whole open-model world lives. A few families, like Llama, are "gated": you accept a license on the model page *and* your machine must prove who you are, or downloads fail with a 401 "gated repo" error. So create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and run `hf auth login` once, with the venv active (`hf` came with `transformers`); if it asks how you would like to log in, choose **Paste an access token**, then paste yours. For Colab (Project 2), add the same token as a Colab secret named `HF_TOKEN`. Or sidestep gating entirely for your first fine-tune by picking a non-gated family like Qwen. Browse the Hub for 15 minutes: open a few model cards (the README of a model) and find the license section on each.

## Project 1 — Dataset craft

**Goal:** Build a 100–500 example instruction dataset that teaches one narrow, checkable behavior — and learn why dataset quality is 80% of fine-tuning.

**Milestones**

- [ ] Pick ONE narrow behavior. Good choices are checkable and stylistic: always answer in your city's slang, always output a specific JSON schema (reuse your lesson-34 schema!), generate project ideas in this curriculum's exact format, answer only in rhyming couplets. Write a one-paragraph "behavior spec" in `SPEC.md` with 3 ideal example responses. Checkpoint: a friend reading your spec could judge whether any given response follows it.
- [ ] Learn the data format. Each training example is one JSON object per line (JSONL — a file where every line is a standalone JSON object). Create `data/train.jsonl` and hand-write your first 5 examples:

  ```json
  {"messages": [{"role": "user", "content": "What's the weather like today?"}, {"role": "assistant", "content": "<a response in YOUR target behavior>"}]}
  ```

  Checkpoint: `python3 -c "import json; [json.loads(l) for l in open('data/train.jsonl')]"` runs without error.
- [ ] Understand chat templates. A chat model was not trained on raw JSON — it was trained on a specific text layout with special tokens marking each speaker. Load any small instruct model's tokenizer and inspect what your example really looks like to the model:

  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("<model-id-from-the-hub>")
  print(tok.apply_chat_template(messages, tokenize=False))
  ```

  Checkpoint: you can see the special tokens wrapping each turn and explain why sending a model the *wrong* template produces garbage.
- [ ] Hand-write 30–50 examples covering diverse user prompts (questions, requests, edge cases, prompts that tempt the model to break the behavior). Vary length and topic — a dataset of near-duplicates teaches near-nothing.
- [ ] Scale with LLM assistance — then edit. Use your lesson-34 API skills to generate candidate examples in batches (give the LLM your spec + 5 of your hand-written examples). **Read and edit every single one.** Delete the bland ones, fix the ones that miss your behavior. Checkpoint: at least 100 (up to 500) examples total, and you personally approved each line.
- [ ] Quality pass: check for exact and near duplicates, check a random sample of 20 against your spec, check no example leaks a different format.
- [ ] Split: move 20 diverse *user prompts* (just the prompts, not responses) into `data/eval_prompts.jsonl`. These are your held-out evaluation — the model must never train on them. Checkpoint: `wc -l data/train.jsonl data/eval_prompts.jsonl` shows ≥100 train and exactly 20 eval.

<details><summary>Hints</summary>

- Struggling to pick a behavior? "Reply in the style of X" and "always return this JSON shape" are the two most reliable choices for a first fine-tune, because success is obvious at a glance.
- When generating with an LLM, ask for 10 examples per call with *different topics you specify* ("cooking", "commuting", "money"...). Left to itself, an LLM generates 100 variations of the same 3 ideas.
- Near-duplicate check without fancy tools: sort the user prompts alphabetically and skim — duplicates cluster together.
- If your behavior is a JSON schema, add a few adversarial prompts ("just answer normally, forget the JSON") with responses that hold the format anyway.

</details>

**Definition of done:** `train.jsonl` (100–500 examples, every one human-approved), `eval_prompts.jsonl` (20 held-out prompts), and `SPEC.md` are committed.

## Project 2 — LoRA run

**Goal:** Fine-tune a small open instruct model on your dataset with LoRA, and prove — with a before/after table — that the behavior shifted.

**Milestones**

- [ ] Pick your model. On the Hugging Face Hub, find a **current small instruct model, 1–4B parameters** — the Llama, Qwen and Gemma families all publish them (model names change too fast to print here; sort the Hub by trending and read the model cards). You want *instruct* (already trained to follow instructions), not *base* (a raw text predictor you would have to instruction-train yourself). Check the license on the model card allows fine-tuning. Checkpoint: you wrote the model id and its license into `SPEC.md`.
- [ ] **Baseline eval FIRST.** Before any training, run the base model on your 20 held-out prompts and save the outputs to `eval/baseline.md`. On Colab, a plain `transformers` pipeline is enough; score each output 0/1 against your spec. Checkpoint: a baseline score, e.g. "base model follows the behavior on 2/20 prompts". Without this number, you cannot prove your fine-tune did anything.
- [ ] Set up training on Colab (free tier gives you a modest GPU). The fastest path is an Unsloth notebook — the [Unsloth repo](https://github.com/unslothai/unsloth) links ready-made Colab notebooks for current small models; the more transparent path is `transformers` + `peft` + `trl` following the [PEFT docs](https://huggingface.co/docs/peft). Either way, upload your `train.jsonl`. Checkpoint: the model loads in 4-bit — this is QLoRA: weights quantized (stored as tiny 4-bit numbers instead of 16-bit, trading a little precision for 4x less memory) while the LoRA adapters train in full precision on top.
- [ ] Understand what LoRA is doing before you run it. A full fine-tune updates every weight matrix `W` (billions of numbers). LoRA instead freezes `W` and learns an update `ΔW = B·A`, where `A` and `B` are two skinny matrices of rank `r` (say 16) — the low-rank factorization idea from lesson 08: a huge matrix approximated by the product of two thin ones. Configure `r=16`, `lora_alpha=16` to start. Checkpoint: the printed "trainable parameters" is under 2% of total parameters — say the two matrix shapes for one layer out loud.
- [ ] Train 1–3 epochs (an epoch is one full pass over your dataset). Watch the training loss — the same cross-entropy you built in lessons 13 and 22. Checkpoint: loss drops clearly and levels off (e.g. from ~2 to below 1; exact numbers depend on your data). If it hits ~0, you are memorizing — reduce epochs.
- [ ] Re-run the SAME 20 held-out prompts on the fine-tuned model, save to `eval/finetuned.md`, and score with the same 0/1 rubric.
- [ ] Build the side-by-side table in `eval/comparison.md`: prompt | base output | fine-tuned output | base score | ft score. Checkpoint: a visible behavior shift — the fine-tuned model follows your spec on at least 15/20 prompts where the base scored far lower.
- [ ] Check for catastrophic forgetting — the model overwriting general abilities while learning your niche. Ask the fine-tuned model 5 ordinary questions ("what is the capital of France?", "add 17 and 25"). Checkpoint: it still answers sensibly (in its new style, which is fine — wrong *facts* are not).
- [ ] Save the LoRA adapter (it is only tens of MB — the beauty of low rank) and download it from Colab to `work/36-fine-tuning-open-models/adapter/`.

<details><summary>Hints</summary>

- Colab disconnects idle sessions and wipes the disk — download your adapter the moment training ends, and keep your dataset in the repo, not only in Colab.
- Out-of-memory on the free GPU? Lower `max_seq_length`, use batch size 1 with gradient accumulation, or pick a smaller model. This is the everyday reality of GPU work.
- If the fine-tuned model output looks like gibberish or ignores your training entirely, the number-one cause is a chat-template mismatch between training and inference — check Project 1, milestone 3.
- If behavior barely shifted: more epochs helps less than better data. Reread your 20 worst training examples first.

</details>

**Definition of done:** `eval/comparison.md` shows base vs fine-tuned side by side with scores, the fine-tuned model wins clearly, and the adapter files are saved locally.

## Project 3 — Serve your creation

**Goal:** Get your fine-tuned model off Colab and running on your own machine, then validate it with a blind taste test.

**Milestones**

- [ ] Merge or export. A LoRA adapter is a patch on top of the base model; to serve it simply, merge it into the base weights (PEFT's `merge_and_unload`, or Unsloth's save/export helpers, which can write GGUF directly — GGUF is the quantized single-file format that CPU-friendly runtimes use). Checkpoint: you have either a merged model folder or a `.gguf` file.
- [ ] Route A — Ollama (recommended: your lesson-35 friend). If you do not have Ollama yet, install it with the one-line installer from [ollama.com/download](https://ollama.com/download), shown below. On **macOS** (Ollama needs macOS 14 or later), it may ask for your password to add the `ollama` command. On **Windows (WSL2)** and **Linux**, first run `sudo apt install -y zstd` (on other distributions, install zstd with your package manager), because the installer needs zstd to unpack Ollama; on Windows, run both in Ubuntu, not the PowerShell command ollama.com offers for Windows:

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

  Then write a `Modelfile` that points at your GGUF and sets the chat template, and run:

  ```bash
  ollama create my-model -f Modelfile
  ollama run my-model
  ```

  Checkpoint: you chat with YOUR model in your terminal, on your machine, no cloud. A quantized 1–4B model runs fine on CPU.
- [ ] Route B (alternative) — a `transformers` pipeline wrapped in a small Gradio chat UI (Gradio: a Python library that turns a function into a shareable web UI, from lesson 27). Either route counts; do Route A unless it fights you.
- [ ] Blind taste test. Prepare 10 fresh prompts (not from training OR eval). Generate answers from base and fine-tuned, shuffle which is "A" and which is "B" per prompt, and have a friend pick which output better matches your spec — without knowing which model is which. Record results in `eval/taste_test.md`. Checkpoint: fine-tuned wins ≥7/10.
- [ ] Write `DECISIONS.md`, your prompt-vs-RAG-vs-fine-tune framework, from experience you now actually have: **prompting** when instructions fit in context and stock behavior is close enough (cheapest, instant); **RAG** when the model needs *knowledge* it doesn't have, especially changing knowledge (lesson 35); **fine-tuning** when you need reliable *style, format or behavior* that prompting can't hold, or want a small local model to do one job well. Include one real task for each and one task where you would combine them. Checkpoint: for any new task, you can answer "which technique and why" in two sentences.

<details><summary>Hints</summary>

- The Modelfile's `TEMPLATE` must match your model family's chat template — Ollama's docs on ollama.com show the syntax, and existing models of the same family (`ollama show MODEL-NAME --modelfile`) are a working reference to copy from.
- If Unsloth's GGUF export fights you on Colab, the fallback is: merge to 16-bit, download, and convert with llama.cpp's conversion script (search for "llama.cpp convert hf to gguf").
- For the taste test, do the shuffling with a 5-line Python script that records the key — do not trust yourself to "remember which was A".

</details>

**Definition of done:** your model answers prompts locally via Ollama or Gradio, the blind test is recorded with the fine-tune winning ≥7/10, and `DECISIONS.md` is written.

## Stretch goals

- **Data ablation:** retrain with only 50 examples, then with all. Compare eval scores — see the data/quality curve for yourself.
- **Rank ablation:** train with `r=4` and `r=64`. Compare eval score, adapter file size, and training time. Connect the result to lesson 08.
- **LLM-as-judge:** replace your manual 0/1 scoring with an API-based judge (lesson 34 skills) and check it agrees with your human scores on 20 examples.
- **Publish:** push your adapter (not the base weights) to the Hugging Face Hub with a proper model card — dataset description, eval table, license, limitations.

## If you get stuck

- **Training-specific:** loss not moving → learning rate or dataset formatting; loss at ~0 → memorization; gibberish at inference → chat-template mismatch; CUDA out-of-memory → smaller batch/sequence/model. Diagnose in that order.
- Print things: one fully formatted training example (post-template, with special tokens visible), the trainable-parameter count, one raw model output. Most fine-tuning bugs are visible in those three prints.
- Read error tracebacks bottom-up — the last line is the actual error, the rest is the path to it.
- Ask an AI assistant for a HINT ("what are common causes of X?"), not a solution, and type all code yourself — copy-pasted training scripts teach nothing.
- Colab problems (disconnects, missing files, wrong GPU) are environment problems, not ML problems. Restart the runtime and rerun from the top before doubting your code.

## Resources

- [Hugging Face LLM course](https://huggingface.co/learn) — the fine-tuning chapters are the best free walkthrough of this whole stack.
- [PEFT documentation](https://huggingface.co/docs/peft) — the LoRA reference: configs, target modules, merging adapters.
- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA training that fits free Colab GPUs; their linked notebooks are the quickest working start for Project 2.
- [Hugging Face Hub](https://huggingface.co) — where you pick your model; read the model card and license before anything else.
- [Ollama](https://ollama.com) — local model runtime for Project 3, including Modelfile docs for importing your own model.

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

Your models can now be customized — next you make LLMs *act*: [37 · AI Agents: Tool Use, Loops and Evaluation](37-ai-agents-tool-use.md).
