# 34 · Using LLMs: APIs, Prompting and Structured Output

**Phase 5 — LLMs** · Estimated time: 1 week · Prerequisites: [04 · Classes, Files, JSON and Errors](../phase-0-foundations/04-python-oop-files-errors.md), [29 · RNNs and LSTMs](../phase-4-nlp-transformers/29-rnn-lstm-char-generation.md), [33 · Train Your Own GPT](../phase-4-nlp-transformers/33-train-your-own-gpt.md)

> You just spent three lessons building and training language models yourself. Now flip roles: instead of building the model, you build *with* one. Frontier LLMs are available behind a simple API — a URL you send text to and get text back — and most working AI products today are exactly that: a well-designed prompt, a schema, and error handling wrapped around an API call. This week you ship three small tools: a streaming chat assistant, a text-to-JSON extractor, and a prompt evaluation lab. The third one teaches the habit that separates professionals from dabblers: measuring prompts instead of vibing them.

## What you will build

- **Project 1 — `chat.py`**: a command-line chat assistant with conversation memory, streaming replies, and a `--persona` flag that changes its personality via the system prompt.
- **Project 2 — `extract.py`**: a structured extractor that turns messy text (job postings, emails, recipes) into validated JSON, with automatic retry when the model returns something malformed.
- **Project 3 — `promptlab/`**: a mini evaluation harness that scores 5 prompt variants against 10 test cases and prints a results table — proof of which prompt actually wins.

## Concepts you will learn by doing

- **Chat API anatomy** — system, user and assistant messages; why the API is stateless and *you* keep the history.
- **Tokens** — the billing and context unit; you built a tokenizer in lesson 32, now you pay by it.
- **Temperature** — the randomness dial you implemented yourself in lesson 29, now as an API parameter.
- **API keys and `.env` files** — keeping secrets out of your code and out of git.
- **Prompting techniques that matter** — clear instructions, few-shot examples, asking for step-by-step reasoning.
- **Structured output** — forcing the model to return JSON your code can parse, and validating it.
- **Streaming** — printing the reply token-by-token as it is generated.
- **Cost awareness** — input vs output tokens, and when a smaller model is the right call.
- **Failure modes** — hallucination (confident nonsense) and prompt sensitivity (tiny wording change, big behavior change).

## Before you start

You need lesson 33 done (so you viscerally know what is behind the API) and lesson 04's JSON skills.

Activate the venv at the repo root and install the client libraries:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install anthropic python-dotenv
mkdir -p work/34-llm-apis-prompting && cd work/34-llm-apis-prompting
```

**Pick your provider.** This lesson uses Claude as the worked example, but every provider's chat API has the same shape (messages in, message out), so everything transfers:

- **Claude API** — sign up at [https://docs.claude.com](https://docs.claude.com) (the docs link to the Console where you create an API key). Check the current trial/credit policy when you sign up — if there are no free credits and you don't want to pay a few dollars, do the whole lesson on the Ollama path below. Even paid, this whole lesson costs very little.
- **OpenAI API** — same idea, docs at [https://platform.openai.com/docs](https://platform.openai.com/docs).
- **Fully local and free** — install [Ollama](https://ollama.com) and it serves open models on your own machine at `http://localhost:11434`, no key, no cost. Slower and weaker than frontier models, but everything in this lesson works with it. This is the zero-cost path. On **macOS**, Ollama needs macOS 14 or later. On **Windows (WSL2)** and **Linux**, run `sudo apt install -y zstd` and then the install command on [ollama.com/download/linux](https://ollama.com/download/linux), which needs zstd to unpack Ollama; on Windows, run both inside Ubuntu instead of installing the Windows version: your code runs in WSL, which by default cannot reach Windows programs at `localhost`.

**Which model?** Model names and prices change every few months, so this lesson never hardcodes them. Open your provider's docs, find the current model list, and pick a small/cheap one for experimenting. Put the name in your `.env` (below) so your code never contains it either.

**Set up your secrets.** An API key is a password for your account — anyone who has it can spend your money. It goes in a `.env` file (a plain text file of `NAME=value` lines that stays on your machine), never in code, never in git. Run this in your lesson folder, `work/34-llm-apis-prompting/`:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=paste-your-key-here
MODEL=paste-a-current-model-name-from-the-docs
EOF
echo ".env" >> ~/ml/ml-learn/.gitignore
```

Run `git status` from the repo root — if `.env` shows up as untracked-and-ignored (it should not appear at all in the list), you are safe. Do this check *before* your first commit this week.

**Smoke test** (Claude example — the SDK reads `ANTHROPIC_API_KEY` from the environment automatically):

```python
# smoke.py
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()                      # reads .env into environment variables
client = anthropic.Anthropic()
response = client.messages.create(
    model=os.environ["MODEL"],
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.content[0].text)
print(response.usage)              # tokens in / tokens out — this is what you pay for
```

Checkpoint: you get a greeting back and a usage line showing input and output token counts. Congratulations — you just did in one call what took you all of lesson 33 to build.

## Project 1 — CLI chat assistant

**Goal**: Build `chat.py`, a terminal chatbot with memory, streaming output, and a switchable personality. Along the way you learn the full anatomy of a chat API.

**Milestones**

- [ ] Understand the three message roles by experimenting in a scratch script. A conversation is a list of dicts: `{"role": "user", ...}` is you, `{"role": "assistant", ...}` is the model's previous replies, and the **system prompt** (a separate `system=` parameter in Claude's API) is standing instructions the model treats as its job description. Send the same question with two different system prompts ("You are a pirate" vs "You answer in one word"). Checkpoint: same question, visibly different behavior — that is the system prompt working.
- [ ] Prove the API is **stateless**: send "My name is YOUR-NAME" (with your own name), then in a *separate* call send "What is my name?". The model has no idea. Now send the second call *with the first exchange included in the messages list*. Checkpoint: it knows your name only when you resend the history — memory is your job, not the API's.
- [ ] Build the core loop of `chat.py`: read user input with `input()`, append it to a `messages` list, call the API, print the reply, append the reply as an `{"role": "assistant", ...}` message. Type `quit` to exit. Checkpoint: a multi-turn conversation where the model remembers earlier turns.
- [ ] Add **streaming** — receiving the reply word-by-word as it is generated instead of waiting for the whole thing. Use the SDK's streaming helper:

  ```python
  with client.messages.stream(model=model, max_tokens=1024,
                              system=system_prompt, messages=messages) as stream:
      for text in stream.text_stream:
          print(text, end="", flush=True)
      final = stream.get_final_message()   # complete message, for your history
  ```

  Checkpoint: replies appear progressively like a typewriter, not in one block after a pause.
- [ ] Add a `--persona` flag using `argparse` (Python's standard library for command-line flags — the whole trick is two lines: `parser = argparse.ArgumentParser(); parser.add_argument("--persona", default="default")`, then read `parser.parse_args().persona`). `python chat.py --persona pirate` loads a pirate system prompt; keep a small dict of 3+ personas (e.g. `default`, `pirate`, `strict-tutor`). Checkpoint: each persona gives a recognizably different conversation.
- [ ] Add **temperature** as a `--temp` flag. Temperature scales the randomness of sampling — you *built this exact dial* in lesson 29 when you divided logits by T before softmax. Ask the same creative question ("invent a name for a coffee shop") 3 times at low temperature and 3 times at high. Checkpoint: low temp gives near-identical answers, high temp gives variety.
- [ ] Print a cost line on exit: total input tokens and output tokens accumulated from each response's `usage`, and note that output tokens usually cost several times more than input tokens (exact prices: your provider's docs). Checkpoint: after a 5-turn chat you can say "this conversation cost roughly X tokens in, Y out."

<details><summary>Hints</summary>

- Keep `messages` as a plain list you append to; the system prompt does NOT go in the list for Claude — it is a separate `system=` parameter. (OpenAI puts it in the list as `{"role": "system", ...}` — same concept, different plumbing.)
- Because you resend the whole history every call, long chats cost more each turn — that is why input tokens grow. Notice this in your cost line.
- Wrap the API call in `try/except` and catch the SDK's error types (rate limits, bad key) so a network hiccup doesn't crash the chat. Print the error and let the user retry.
- For Ollama: it exposes an HTTP API on localhost — see [ollama.com](https://ollama.com) docs for the endpoint shape, and adapt your call function. Design your code so the "call the model" part is one function you can swap.

</details>

**Definition of done**: `python chat.py --persona pirate --temp 1.0` holds a streaming, multi-turn conversation that remembers context and reports token usage on exit — and your `.env` is not in git.

## Project 2 — Structured extractor

**Goal**: Build `extract.py`: messy human text in, validated JSON out. This input→schema→validate→retry pattern is the engine inside most real LLM products.

**Milestones**

- [ ] Collect 5 messy sample inputs as text files in `samples/`: e.g. two job postings (copy-paste from anywhere), an email arranging a meeting, a recipe, a receipt-like message ("paid Rs 1,450 to Sharma Electricals on 3rd March for wiring"). Real, imperfect text — that is the point.
- [ ] Define your target **schema** — the exact JSON shape you want, e.g. for job postings: `{"title": str, "company": str, "location": str, "salary_range": str or null, "skills": [str]}`. Write it down as a Python dict of field names and types. Deciding what `null` means (field genuinely absent from the text) is part of the design.
- [ ] Version 1, naive: prompt "Extract the following fields as JSON: ... Text: ...", then `json.loads()` the reply. Run it on all 5 samples. Checkpoint: it works on some, and at least once you get JSON wrapped in prose or markdown code fences — congratulations, you have met the failure mode this project exists to fix.
- [ ] Harden the prompt: demand *only* raw JSON with exactly your keys, and add **one worked example** in the prompt (a sample text plus the exact JSON you want back). Giving examples instead of only instructions is called **few-shot prompting**, and it is the single highest-value prompting technique. Checkpoint: format errors become rare.
- [ ] Check whether your provider has a native **structured output** mode — a request option that constrains the model to emit JSON matching your schema (see [docs.claude.com](https://docs.claude.com) for Claude's, [platform.openai.com/docs](https://platform.openai.com/docs) for OpenAI's). Use it if available; keep the prompt-only version working too, so you understand what the feature saves you.
- [ ] Write `validate(data, schema)` yourself: correct type for every field, no missing keys, no invented extra keys. Return a list of problems, empty list = valid. (You know enough Python from lesson 04 — no libraries needed, though the `pydantic` library does this for a living and is worth a look afterwards.)
- [ ] Add the **retry loop**: if `json.loads` fails or validation returns problems, call the model again — include the previous bad output and the error message, ask it to fix it. Max 3 attempts, then give up loudly with a clear error. Checkpoint: on all 5 samples you either get valid JSON or an honest failure, never a silent crash.
- [ ] Hallucination check: add one sample that is *missing* a field (a job posting with no salary). Does the model invent one? A made-up-but-plausible value is a **hallucination** — the model filling gaps with confident fiction. Fix it in the prompt: "use null for anything not stated in the text — never guess." Checkpoint: absent fields come back as `null`, not fiction.

<details><summary>Hints</summary>

- Strip markdown fences before parsing: if the reply starts with ```` ```json ````, slice them off. Cheap and effective first line of defense.
- Set temperature to 0 (or your provider's minimum) for extraction — you want the most deterministic output, not creativity.
- In the retry prompt, be surgical: "Your previous output failed validation with these errors: {errors}. Return corrected JSON only." Feeding the error back works remarkably well.
- Test your validator on hand-written bad JSON first (wrong type, missing key, extra key) so you trust it before blaming the model.

</details>

**Definition of done**: `python extract.py samples/job1.txt` prints valid, schema-conforming JSON for all your samples, absent fields are `null`, and a deliberately hopeless input fails with a clear message after 3 attempts.

## Project 3 — Prompt lab

**Goal**: Stop guessing which prompt is better — measure it. Build a tiny eval harness: one task, 5 prompt variants, 10 test cases, a scoreboard. This is the most important habit in this lesson.

**Milestones**

- [ ] Pick one task with checkable answers. Good choice: **grading short answers** — given a question, a reference answer and a student answer, output exactly `CORRECT` or `INCORRECT`. (Alternative: article summarization, but grading is easier to score, so start there.)
- [ ] Build `cases.json`: 10 test cases you write yourself, each `{"question": ..., "reference": ..., "student": ..., "expected": "CORRECT" or "INCORRECT"}`. Make at least 3 of them *hard*: a correct answer worded completely differently from the reference, an answer that is close but subtly wrong, an empty answer. Easy test cases are how bad prompts sneak through.
- [ ] Write 5 prompt variants in `prompts/v1.txt` … `v5.txt`, each with a `{question}` / `{reference}` / `{student}` placeholder (fill with Python's `.format()`). Make them genuinely different strategies: (1) one bare instruction, (2) detailed rules for what counts as correct, (3) **few-shot** — rules plus 2 worked examples, (4) **step-by-step reasoning** — "explain your reasoning, then give a verdict on the last line" (asking the model to reason before answering measurably helps on judgment tasks), (5) your own wildcard.
- [ ] Build `run_eval.py`: for each variant × each case, call the model (temperature 0), parse the verdict from the reply, compare with `expected`, score 1 or 0. That is 50 calls — a few minutes and a fraction of your credits. Save every raw reply to `results/` so you can inspect failures. Checkpoint: a results file with 50 graded outcomes.
- [ ] Print the scoreboard — one row per variant: accuracy (you built this metric in lesson 14) and total tokens used. Checkpoint: a table like `v3: 9/10` vs `v1: 6/10`, and the variants are *not* all equal.
- [ ] Read the failures. For the best variant, open the raw replies it got wrong: is the model wrong, is your test case ambiguous, or is your verdict-parsing too strict? All three happen constantly in real eval work. Fix what needs fixing and re-run.
- [ ] **Prompt sensitivity** experiment: take your best variant, change something trivial (reorder two rules, rephrase one sentence), re-run. Checkpoint: the score moves — sometimes by a lot. That is why serious teams re-run evals on every prompt edit, exactly like you re-run tests on every code edit.

<details><summary>Hints</summary>

- Make the output format part of the contract: "the last line of your reply must be exactly CORRECT or INCORRECT" — then parse only the last non-empty line, case-insensitively. Robust parsing is half of eval engineering.
- Cache responses to disk keyed by (variant, case index) so re-running the scoreboard doesn't re-spend tokens on calls you already made.
- 10 cases is enough to learn the method, not enough to trust a 1-point gap: 9/10 vs 8/10 is a coin flip, 9/10 vs 5/10 is signal. Say so in your conclusions.
- Keep the eval harness generic (task = prompt template + cases + scorer) — you will reuse it in lessons 35 and 37.

</details>

**Definition of done**: one command re-runs the whole eval and prints the table; you can name the winning variant, its score, and one concrete failure it still has. You never again say "this prompt feels better" without a number.

## Stretch goals

- Add `/save` and `/load` commands to `chat.py` that persist conversation history to JSON (lesson 04 skills) so a chat survives restarts.
- Point your extractor at 20 inputs and batch them: read a whole folder, write one JSON-lines output file, report a success rate.
- Run your prompt-lab eval against a second model (a smaller/cheaper one, or a local Ollama model) — same table, extra column. Does the best prompt for one model win on the other?
- Build a rough token-cost estimator: using prices from your provider's docs, print the actual currency cost of an eval run.

## If you get stuck

- `401` / authentication errors: your key is not reaching the client. Check `load_dotenv()` runs before the client is created, print `os.environ.get("ANTHROPIC_API_KEY")[:8]` to confirm it loaded, and make sure you are running from the folder containing `.env`.
- `429` rate-limit errors: free tiers allow limited requests per minute. Add `time.sleep(1)` between eval calls and retry with a wait on 429.
- Model ignores your format instructions: it happens; that is prompt sensitivity, not a bug in your code. Add a worked example (few-shot beats instructions), and rely on your retry loop — that is why you built it.
- JSON parse failures: print the raw reply *before* parsing. Nine times out of ten it is markdown fences or a chatty preamble you can strip or prompt away.
- Standing advice: read the error message bottom-up, print intermediate values (here: the raw API response) instead of guessing, ask an AI assistant for a HINT — not a solution — and type all code yourself. Typing it is how it sticks.

## Resources

- [Claude API docs](https://docs.claude.com) — the primary worked example: API reference, current model list, structured output, streaming, and a good prompt-engineering guide.
- [OpenAI API docs](https://platform.openai.com/docs) — same concepts on another provider; skim to see how much transfers.
- [Ollama](https://ollama.com) — run open models locally for free; the zero-cost way to do this entire lesson.

## Skills unlocked

- [ ] I can explain system/user/assistant roles and why the client must resend conversation history.
- [ ] I can keep API keys in `.env`, out of my code and out of git, and verify it.
- [ ] I can stream a model's response and manage conversation memory in a CLI app.
- [ ] I can get validated JSON from an LLM using schema-shaped prompts, few-shot examples and a retry loop.
- [ ] I can explain temperature, tokens, and input-vs-output cost, and estimate what a call costs.
- [ ] I can name the two big failure modes — hallucination and prompt sensitivity — and defend against both.
- [ ] I can build a small eval (cases, scorer, table) and choose between prompts with numbers, not vibes.

## Next up

Your chat assistant knows nothing about *your* documents — next you fix that by building retrieval-augmented generation: [35 · RAG: Build "Chat With My Documents"](35-rag-chat-with-your-docs.md).
