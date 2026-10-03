# 34 · Using LLMs: APIs, Prompting and Structured Output

**Phase 5 — LLMs** · Estimated time: 1 week · Prerequisites: [04 · Classes, Files, JSON and Errors](../phase-0-foundations/04-python-oop-files-errors.md), [29 · RNNs and LSTMs](../phase-4-nlp-transformers/29-rnn-lstm-char-generation.md), [33 · Train Your Own GPT](../phase-4-nlp-transformers/33-train-your-own-gpt.md)

> The previous three lessons built and trained language models. This lesson switches roles: instead of building a model, it builds *with* one. Frontier LLMs are available behind a simple API (a URL that takes text in and sends text back), and most working AI products today are exactly that: a well-designed prompt, a schema, and error handling wrapped around an API call. This week's three projects are small working tools: a streaming chat assistant, a text-to-JSON extractor, and a prompt evaluation lab. The third one teaches the habit that separates professionals from amateurs: measuring prompts instead of judging them by feel.

## What this lesson builds

- **Project 1 — `chat.py`**: a command-line chat assistant with conversation memory, streaming replies, and a `--persona` flag that changes its personality via the system prompt.
- **Project 2 — `extract.py`**: a structured extractor that turns messy text (job postings, emails, recipes) into validated JSON, with automatic retry when the model returns something malformed.
- **Project 3 — `promptlab/`**: a mini evaluation harness that scores 5 prompt variants against 10 test cases and prints a results table: proof of which prompt actually wins.

## Concepts covered

- **Chat API anatomy**: system, user and assistant messages; why the API is stateless and *the client* keeps the history.
- **Tokens**: the billing and context unit. Lesson 32 built a tokenizer; now API calls are paid for by the token.
- **Temperature**: the randomness dial implemented in lesson 29, now as an API parameter.
- **API keys and `.env` files**: keeping secrets out of the code and out of git.
- **Prompting techniques that matter**: clear instructions, few-shot examples, asking for step-by-step reasoning.
- **Structured output**: forcing the model to return JSON that the code can parse, and validating it.
- **Streaming**: printing the reply token-by-token as it is generated.
- **Cost awareness**: input vs output tokens, and when a smaller model is the right choice.
- **Failure modes**: hallucination (confident nonsense) and prompt sensitivity (tiny wording change, big behavior change).

## Before starting

This lesson needs lesson 33 done (it gives a first-hand feel for what is behind the API) and the JSON skills from lesson 04.

Activate the venv at the repo root and install the client libraries:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install anthropic python-dotenv
mkdir -p work/34-llm-apis-prompting && cd work/34-llm-apis-prompting
```

**Pick a provider.** This lesson uses Claude as the worked example, but every provider's chat API has the same shape (messages in, message out), so everything transfers:

- **Claude API**: sign up at [https://docs.claude.com](https://docs.claude.com) (the docs link to the Console, where API keys are created). Check the current trial/credit policy when signing up. If there are no free credits, either pay a few dollars or do the whole lesson on the Ollama path below. Even paid, this whole lesson costs very little.
- **OpenAI API**: same idea, docs at [https://platform.openai.com/docs](https://platform.openai.com/docs).
- **Fully local and free**: install [Ollama](https://ollama.com), and it serves open models on the computer itself at `http://localhost:11434`, with no key and no cost. It is slower and weaker than frontier models, but everything in this lesson works with it. This is the zero-cost path. On **macOS**, Ollama needs macOS 14 or later. On **Windows (WSL2)** and **Linux**, run `sudo apt install -y zstd` and then the install command on [ollama.com/download/linux](https://ollama.com/download/linux), which needs zstd to unpack Ollama. On Windows, run both inside Ubuntu instead of installing the Windows version: the lesson's code runs in WSL, which by default cannot reach Windows programs at `localhost`.

**Which model?** Model names and prices change every few months, so this lesson never hardcodes them. Open the provider's docs, find the current model list, and pick a small/cheap one for experimenting. Put the name in the `.env` file (below) so that the code never contains it either.

**Set up the secrets.** An API key is a password for the account: anyone who has it can spend money on that account. It goes in a `.env` file (a plain text file of `NAME=value` lines that stays on the computer), never in code, never in git. Run this in the lesson folder, `work/34-llm-apis-prompting/`:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=paste-your-key-here
MODEL=paste-a-current-model-name-from-the-docs
EOF
echo ".env" >> ~/ml/ml-learn/.gitignore
```

Check that git ignores the file: in the lesson folder, run `git check-ignore .env`. Checkpoint: it prints `.env`. If it prints nothing, `.env` is not ignored: open `~/ml/ml-learn/.gitignore`, put `.env` on a line of its own, and run the check again. (Plain `git status` cannot show this, because it lists a new folder only by its name, never the files inside it.) Do this check *before* the first commit this week.

**Smoke test** (Claude example; the SDK reads `ANTHROPIC_API_KEY` from the environment automatically):

```python
# smoke.py
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()                      # reads .env into environment variables
client = anthropic.Anthropic()
response = client.messages.create(
    model=os.environ["MODEL"],
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
text = "".join(b.text for b in response.content if b.type == "text")
print(text)
print(response.usage)              # tokens in / tokens out — this is what gets billed
```

Checkpoint: the script prints a greeting and a usage line showing input and output token counts. One call now does what took all of lesson 33 to build.

A reply is a list of content blocks. Newer models can start a reply with a `thinking` block before the text, so the code keeps only the blocks whose `type` is `"text"` instead of reading `content[0]`. Thinking also counts toward `max_tokens`, so keep that limit generous. Use the same pattern in `chat.py`, `extract.py` and `run_eval.py`.

## Project 1 — CLI chat assistant

**Goal**: Build `chat.py`, a terminal chatbot with memory, streaming output, and a switchable personality. Along the way, the project covers the full anatomy of a chat API.

**Milestones**

- [ ] Understand the three message roles by experimenting in a scratch script. A conversation is a list of dicts: `{"role": "user", ...}` is what the user types, `{"role": "assistant", ...}` is the model's previous replies, and the **system prompt** (a separate `system=` parameter in Claude's API) is standing instructions that the model treats as its job description. Send the same question with two different system prompts ("You are a pirate" vs "You answer in one word"). Checkpoint: same question, visibly different behavior. That is the system prompt working.
- [ ] Prove the API is **stateless**: send "My name is YOUR-NAME" (with a real name), then in a *separate* call send "What is my name?". The model has no idea. Now send the second call *with the first exchange included in the messages list*. Checkpoint: the model knows the name only when the history is sent again. Memory is the client's job, not the API's.
- [ ] Build the core loop of `chat.py`: read user input with `input()`, append it to a `messages` list, call the API, print the reply, append the reply as an `{"role": "assistant", ...}` message. Type `quit` to exit. Checkpoint: a multi-turn conversation where the model remembers earlier turns.
- [ ] Add **streaming**: receiving the reply word-by-word as it is generated instead of waiting for the whole thing. Use the SDK's streaming helper:

  ```python
  with client.messages.stream(model=model, max_tokens=1024,
                              system=system_prompt, messages=messages) as stream:
      for text in stream.text_stream:
          print(text, end="", flush=True)
      final = stream.get_final_message()   # complete message, for the history
  ```

  Checkpoint: replies appear progressively like a typewriter, not in one block after a pause.
- [ ] Add a `--persona` flag using `argparse` (Python's standard library for command-line flags; it takes just two lines: `parser = argparse.ArgumentParser(); parser.add_argument("--persona", default="default")`, then read `parser.parse_args().persona`). `python chat.py --persona pirate` loads a pirate system prompt; keep a small dict of 3+ personas (e.g. `default`, `pirate`, `strict-tutor`). Checkpoint: each persona gives a recognizably different conversation.
- [ ] Add **temperature** as a `--temp` flag. Temperature scales the randomness of sampling. Lesson 29 *built this exact dial* by dividing the logits by T before softmax. The `anthropic` SDK 1.x has no `temperature=` argument (passing one raises `TypeError`), so send it in the request body: `extra_body={"temperature": args.temp}`. Claude Opus 4.7 and later reject any `temperature` with a 400 error (even the default, 1.0), and Claude Sonnet 5 and later reject any value but the default, so run this milestone on a model that still accepts it (the provider's model pages say which; Claude Haiku 4.5 and Claude Sonnet 4.6 do) or on Ollama. Ask the same creative question ("invent a name for a coffee shop") 3 times at low temperature and 3 times at high. Checkpoint: low temp gives near-identical answers, high temp gives variety.
- [ ] Print a cost line on exit: total input tokens and output tokens accumulated from each response's `usage`, and note that output tokens usually cost several times more than input tokens (exact prices: the provider's docs). Checkpoint: after a 5-turn chat, the cost line shows enough to say "this conversation cost roughly X tokens in, Y out."

<details><summary>Hints</summary>

- Keep `messages` as a plain list that the loop appends to. For Claude, the system prompt does *not* go in the list; it is a separate `system=` parameter. (OpenAI puts it in the list as `{"role": "system", ...}`: same concept, different plumbing.)
- Because the whole history is sent again with every call, long chats cost more each turn; that is why input tokens grow. Look for this in the cost line.
- Wrap the API call in `try/except` and catch the SDK's error types (rate limits, bad key) so a network hiccup doesn't crash the chat. Print the error and let the user retry.
- For Ollama: it exposes an HTTP API on localhost. See the [ollama.com](https://ollama.com) docs for the endpoint shape, and adapt the call function. Design the code so that the "call the model" part is one function that can be swapped: `chat(prompt)`, which sends one user message and returns the reply text (the chat loop can call a sibling that takes the whole `messages` list). Lesson 35 calls this `chat()` helper.

</details>

**Definition of done**: `python chat.py --persona pirate --temp 1.0` (on a model that accepts temperature) holds a streaming, multi-turn conversation that remembers context and reports token usage on exit, and the `.env` file is not in git.

## Project 2 — Structured extractor

**Goal**: Build `extract.py`: messy human text in, validated JSON out. This input→schema→validate→retry pattern is the engine inside most real LLM products.

**Milestones**

- [ ] Collect 5 messy sample inputs as text files in `samples/`: e.g. two job postings (copy-paste from anywhere), an email arranging a meeting, a recipe, a receipt-like message ("paid Rs 1,450 to Sharma Electricals on 3rd March for wiring"). Real, imperfect text is the point.
- [ ] Define the target **schema**, the exact JSON shape the extractor should return. For job postings, for example: `{"title": str, "company": str, "location": str, "salary_range": str or null, "skills": [str]}`. Write it down as a Python dict of field names and types. Deciding what `null` means (field genuinely absent from the text) is part of the design.
- [ ] Version 1, naive: prompt "Extract the following fields as JSON: ... Text: ...", then `json.loads()` the reply. Run it on all 5 samples. Checkpoint: it works on some, and at least once the reply is JSON wrapped in prose or markdown code fences. That is the failure mode this project exists to fix.
- [ ] Harden the prompt: demand *only* raw JSON with exactly the schema's keys, and add **one worked example** in the prompt (a sample text plus the exact JSON expected back). Giving examples instead of only instructions is called **few-shot prompting**, and it is the single highest-value prompting technique. Checkpoint: format errors become rare.
- [ ] Check whether the provider has a native **structured output** mode: a request option that constrains the model to emit JSON matching a given schema (see [docs.claude.com](https://docs.claude.com) for Claude's, [platform.openai.com/docs](https://platform.openai.com/docs) for OpenAI's). Use it if available; keep the prompt-only version working too, to see what the feature saves.
- [ ] Write `validate(data, schema)` from scratch: correct type for every field, no missing keys, no invented extra keys. Return a list of problems, empty list = valid. (Lesson 04 covers enough Python for this, so no libraries are needed, though the `pydantic` library exists to do exactly this and is worth a look afterwards.)
- [ ] Add the **retry loop**: if `json.loads` fails or validation returns problems, call the model again. Include the previous bad output and the error message, and ask the model to fix it. Max 3 attempts, then give up loudly with a clear error. Checkpoint: on all 5 samples, the result is either valid JSON or an honest failure, never a silent crash.
- [ ] Hallucination check: add one sample that is *missing* a field (a job posting with no salary). Does the model invent one? A made-up-but-plausible value is a **hallucination**: the model filling gaps with confident fiction. Fix it in the prompt: "use null for anything not stated in the text — never guess." Checkpoint: absent fields come back as `null`, not fiction.

<details><summary>Hints</summary>

- Strip markdown fences before parsing: if the reply starts with ```` ```json ````, slice them off. This is a cheap and effective first line of defense.
- Set temperature to 0 for extraction when the model accepts it (through `extra_body`, as in Project 1); on a model that rejects it, leave it out. Extraction needs the most deterministic output, not creativity.
- In the retry prompt, be precise: "Your previous output failed validation with these errors: {errors}. Return corrected JSON only." Feeding the error back works remarkably well.
- Test the validator on hand-written bad JSON first (wrong type, missing key, extra key), to be sure it works before blaming the model.

</details>

**Definition of done**: `python extract.py samples/job1.txt` prints valid, schema-conforming JSON for all the samples, absent fields are `null`, and a deliberately hopeless input fails with a clear message after 3 attempts.

## Project 3 — Prompt lab

**Goal**: Stop guessing which prompt is better, and measure it instead. Build a tiny eval harness: one task, 5 prompt variants, 10 test cases, a scoreboard. This is the most important habit in this lesson.

**Milestones**

- [ ] Pick one task with checkable answers. A good choice is **grading short answers**: given a question, a reference answer and a student answer, output exactly `CORRECT` or `INCORRECT`. (Alternative: article summarization, but grading is easier to score, so start there.)
- [ ] Build `cases.json`: 10 hand-written test cases, each `{"question": ..., "reference": ..., "student": ..., "expected": "CORRECT" or "INCORRECT"}`. Make at least 3 of them *hard*: a correct answer worded completely differently from the reference, an answer that is close but subtly wrong, an empty answer. Easy test cases are how bad prompts sneak through.
- [ ] Write 5 prompt variants in `prompts/v1.txt` … `v5.txt`, each with a `{question}` / `{reference}` / `{student}` placeholder (fill with Python's `.format()`). Make them genuinely different strategies: (1) one bare instruction; (2) detailed rules for what counts as correct; (3) **few-shot**: rules plus 2 worked examples; (4) **step-by-step reasoning**: "explain your reasoning, then give a verdict on the last line" (asking the model to reason before answering measurably helps on judgment tasks); (5) a free-choice wildcard.
- [ ] Build `run_eval.py`: for each variant × each case, call the model (temperature 0 where the model accepts it), parse the verdict from the reply, compare with `expected`, score 1 or 0. That is 50 calls, which take a few minutes and use a fraction of the credits. Save every raw reply to `results/` so that failures can be inspected. Checkpoint: a results file with 50 graded outcomes.
- [ ] Print the scoreboard, with one row per variant showing accuracy (the metric built in lesson 14) and total tokens used. Checkpoint: a table like `v3: 9/10` vs `v1: 6/10`, and the variants are *not* all equal.
- [ ] Read the failures. For the best variant, open the raw replies it got wrong: is the model wrong, is the test case ambiguous, or is the verdict-parsing too strict? All three happen constantly in real eval work. Fix what needs fixing and re-run.
- [ ] **Prompt sensitivity** experiment: take the best variant, change something trivial (reorder two rules, rephrase one sentence), re-run. Checkpoint: the score moves, sometimes by a lot. That is why serious teams re-run evals on every prompt edit, exactly as tests are re-run on every code edit.

<details><summary>Hints</summary>

- Make the output format part of the contract: "the last line of your reply must be exactly CORRECT or INCORRECT". Then parse only the last non-empty line, case-insensitively. Robust parsing is half of eval engineering.
- Cache responses to disk keyed by (variant, case index) so that re-running the scoreboard doesn't re-spend tokens on calls already made.
- 10 cases is enough to learn the method, not enough to trust a 1-point gap: 9/10 vs 8/10 is a coin flip, 9/10 vs 5/10 is signal. Say so in the conclusions.
- Keep the eval harness generic (task = prompt template + cases + scorer). Lessons 35 and 37 build their own evals with the same method: fixed cases, an automatic scorer, a results table.

</details>

**Definition of done**: one command re-runs the whole eval and prints the table; the winning variant can be named, along with its score and one concrete failure it still has. From here on, a claim like "this prompt feels better" always comes with a number.

## Stretch goals

- Add `/save` and `/load` commands to `chat.py` that persist conversation history to JSON (lesson 04 skills) so a chat survives restarts.
- Point the extractor at 20 inputs and batch them: read a whole folder, write one JSON-lines output file, report a success rate.
- Run the prompt-lab eval against a second model (a smaller/cheaper one, or a local Ollama model): same table, extra column. Does the best prompt for one model win on the other?
- Build a rough token-cost estimator: using prices from the provider's docs, print the actual currency cost of an eval run.

## Getting unstuck

- `401` / authentication errors: the key is not reaching the client. Check that `load_dotenv()` runs before the client is created, print `str(os.environ.get("ANTHROPIC_API_KEY"))[:8]` to confirm it loaded (it prints `None` when the key is missing), and make sure `.env` sits in the same folder as the script or in a folder above it: `load_dotenv()` searches upward from the script's own folder, not from the folder the command runs in.
- `429` rate-limit errors: free tiers allow limited requests per minute. Add `time.sleep(1)` between eval calls and retry with a wait on 429.
- Model ignores the format instructions: this happens, and it is prompt sensitivity, not a bug in the code. Add a worked example (few-shot beats instructions), and rely on the retry loop; that is why it was built.
- JSON parse failures: print the raw reply *before* parsing. Nine times out of ten it is markdown fences or a chatty preamble that can be stripped or prompted away.
- Standing advice: read the error message bottom-up, print intermediate values (here: the raw API response) instead of guessing, ask an AI assistant for a hint (not a solution), and type all code by hand. Typing it is how it sticks.

## Resources

- [Claude API docs](https://docs.claude.com) — the primary worked example: API reference, current model list, structured output, streaming, and a good prompt-engineering guide.
- [OpenAI API docs](https://platform.openai.com/docs) — same concepts on another provider; skim to see how much transfers.
- [Ollama](https://ollama.com) — run open models locally for free; the zero-cost way to do this entire lesson.

## Skills unlocked

- [ ] I can explain system/user/assistant roles and why the client must resend conversation history.
- [ ] I can keep API keys in `.env`, out of the code and out of git, and verify it.
- [ ] I can stream a model's response and manage conversation memory in a CLI app.
- [ ] I can get validated JSON from an LLM using schema-shaped prompts, few-shot examples and a retry loop.
- [ ] I can explain temperature, tokens, and input-vs-output cost, and estimate what a call costs.
- [ ] I can name the two big failure modes (hallucination and prompt sensitivity) and defend against both.
- [ ] I can build a small eval (cases, scorer, table) and choose between prompts with numbers, not gut feeling.

## Next up

The chat assistant knows nothing about a user's *own* documents, and the next lesson fixes that by building retrieval-augmented generation: [35 · RAG: Build "Chat With My Documents"](35-rag-chat-with-your-docs.md).
