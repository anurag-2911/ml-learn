# 37 · AI Agents: Tool Use, Loops and Evaluation

**Phase 5 — LLMs** · Estimated time: 1-2 weeks · Prerequisites: [34 · Using LLMs: APIs, Prompting and Structured Output](34-llm-apis-prompting.md), [04 · Classes, Files, JSON and Errors](../phase-0-foundations/04-python-oop-files-errors.md)

> Claude Code, the assistant used alongside this curriculum, is an AI agent: an LLM that reads a request, decides to call a tool (read a file, run a command), looks at the result, and repeats until the job is done. That is the whole idea: **an agent is an LLM + tools + a loop that runs until the task is done.** This lesson builds that loop from scratch, with no framework, and then covers the part that separates impressive demos from reliable products: guardrails and evaluation. By the end, the way that assistant works is no longer a mystery.

## What this lesson builds

- **Project 1 — Agent loop from scratch**: a ~150-line Python program (`agent.py`) with four tools (calculator, `read_file`, `list_files`, `save_note`) that chains tool calls to answer "read expenses.csv and compute the total of column 3". It uses no frameworks.
- **Project 2 — Research assistant**: the same agent upgraded with `web_search` and `fetch_page` tools, answering questions that need 2-3 hops, plus a full **trace log** for watching it think.
- **Project 3 — Agent evals**: a 10-task test suite (`evals.py` + `tasks.json`) with automatic pass/fail checks, run 3 times to measure a success rate, and then a measurably better agent after tuning the prompts.

## Concepts covered

- **Agent** = LLM + tools + a loop that runs until the task is done.
- **Tool (function) calling**: describing Python functions to the model in a JSON schema, so that the model can ask the program to run them.
- **The ReAct pattern**: reason → act → observe, repeated. The model thinks, picks a tool, sees the result, thinks again.
- **Safe dispatch**: parsing the model's requested tool call and executing it defensively (allowlist, validated inputs, errors returned as data).
- **Guardrails**: max iterations, tool allowlists, path restrictions, confirm-before-dangerous-actions.
- **Failure modes**: infinite loops, hallucinated tools and ignored errors, and how agents recover from them (or fail to).
- **Trace reading**: the single most important agent-debugging skill.
- **Agent evals**: measuring success rate on a task suite across multiple runs, because LLMs are non-deterministic.
- **Multi-step planning**, and a first look at **MCP** (Model Context Protocol), the emerging standard for plugging tools into any agent.

## Before starting

- This lesson needs the working LLM API setup from [lesson 34](34-llm-apis-prompting.md): an API key in a `.env` file and the provider's Python SDK installed. The examples here use the Anthropic SDK. If a different provider was picked in lesson 34, keep using it: every provider has tool calling, and only the field names differ (check the provider's docs).
- Agent loops make **many** API calls per task (Project 3 makes ~30+ per eval run). Use a small, cheap model for development, and check the provider's pricing page before big runs. Set a spending limit in the provider console if it offers one.

Go to the repo root, activate the venv from lesson 01, install the libraries, and create the work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install anthropic requests beautifulsoup4
mkdir -p work/37-ai-agents && cd work/37-ai-agents
```

Then copy the `.env` file from lesson 34 into this folder with `cp ../34-llm-apis-prompting/.env .` (the `.env` line added to `.gitignore` in lesson 34 keeps this copy out of git too). As in lesson 34's smoke test, call `load_dotenv()` before creating the client, so that the SDK finds the key.

Project 2 also needs a web search library. The popular free one (DuckDuckGo-based) has been renamed more than once, so search PyPI for the current DuckDuckGo search package (at the time of writing: `pip install ddgs`, formerly `duckduckgo_search`). Any library or search API that returns titles + URLs for a query works.

Create the test data for Project 1:

```bash
mkdir -p workspace
printf 'date,item,amount\n2026-01-03,keyboard,45.50\n2026-01-10,monitor,220.00\n2026-02-01,coffee,4.75\n2026-02-14,ssd,89.99\n' > workspace/expenses.csv
```

Everything the agent touches lives inside `workspace/`: that folder *is* the agent's world. This is the first guardrail.

## Project 1 — Agent loop from scratch

**Goal:** Implement the agent loop with zero frameworks, writing the request, the dispatch and the loop by hand, so that the model chains `list_files` → `read_file` → `calculator` on its own to answer a question about a CSV file.

**Milestones**

- [ ] **Write the tools as plain Python first.** In `tools.py`, write four ordinary functions, each returning a *string* (models read text): `calculator(expression)`, `read_file(path)`, `list_files()`, `save_note(filename, text)`. `read_file` and `list_files` must refuse any path that escapes `workspace/` (reject absolute paths and anything containing `..`). Test them by hand in a Python REPL before any AI is involved. Checkpoint: `read_file("expenses.csv")` returns the CSV text; `read_file("../tools.py")` returns an error string, not the file.
- [ ] **Make the calculator safe.** `eval("2+2")` works, but `eval` runs *arbitrary code*: a model (or a malicious input) could pass `__import__('os').system('rm -rf ~')`. Never bare-`eval` model output. Parse the expression with Python's `ast` module and only allow number and arithmetic-operator nodes. Checkpoint: `calculator("220.00+45.50")` returns `265.5`; `calculator("__import__('os')")` returns an error string.
- [ ] **Describe the tools to the model.** A **tool schema** is a JSON description of a tool (its name, what it does, what arguments it takes) that is sent with the request, so that the model knows what it *can* ask the program to do. Write one per tool in `agent.py`:

  ```python
  tools = [
      {
          "name": "calculator",
          "description": "Evaluate an arithmetic expression like '2.5 * (3 + 4)'. Use this for ALL math - never do arithmetic in your head.",
          "input_schema": {
              "type": "object",
              "properties": {"expression": {"type": "string"}},
              "required": ["expression"],
          },
      },
      # ... the other three tools
  ]
  ```

  Descriptions are prompts: the model chooses tools by reading them. "Use this for ALL math" is doing real work in that sentence.
- [ ] **Send one request and just look at it.** Before any loop, send a single request with the tools attached and print what comes back:

  ```python
  import anthropic
  client = anthropic.Anthropic()
  MODEL = "..."   # pick a current model id from the provider's docs (lesson 34)

  response = client.messages.create(
      model=MODEL, max_tokens=4096, tools=tools,
      messages=[{"role": "user", "content": "What is 13.5% of 2840?"}],
  )
  print(response.stop_reason)
  for block in response.content:
      print(block)
  ```

  Checkpoint: `stop_reason` is `"tool_use"` and the content includes a `tool_use` block with a name (`calculator`), a unique `id`, and an `input` dict. The model did not *run* anything; it is **asking the program** to run it. That request/permission split is the heart of tool calling.
- [ ] **Dispatch safely.** Write `execute_tool(name, args)`: look the name up in an allowlist dict (`{"calculator": calculator, ...}`), call it inside `try/except`, and return the result (or the error message) as a string. If the model asks for a tool that does not exist (models sometimes **hallucinate tools** that were never offered), return `"Error: unknown tool 'X'"` instead of crashing. Errors go *back to the model as data*; a good model reads the error and tries something else. That is **error recovery**.
- [ ] **Close the loop.** Now wrap it: while `stop_reason == "tool_use"`, append the assistant's full `response.content` to the `messages` list, execute every requested tool, and append one user message containing a `tool_result` block per call, each carrying the matching `tool_use_id`:

  ```python
  {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": block.id, "content": result_string}
  ]}
  ```

  Then call the API again with the grown `messages`. When `stop_reason` is `"end_turn"`, the text blocks in the final response are the answer. Print a line per iteration (`[turn 3] calculator({'expression': ...})`) to watch it work. This loop is the whole agent: send, execute, feed back, repeat.
- [ ] **Add the two non-negotiable guardrails**: a **max-iterations cap** (stop after ~10 turns, because a confused model can loop forever and every turn costs money) and the tool allowlist that is already in place. Checkpoint: with the cap set to 1, the agent stops with a "gave up" message instead of running forever.
- [ ] **The graduation task.** Run: `python agent.py "Read expenses.csv and compute the total of column 3"`. Checkpoint: the trace shows it chaining tools (likely `list_files` → `read_file` → `calculator`) and the final answer is **360.24**. If it does the math "in its head" instead of calling the calculator, sharpen the calculator's description and try again. That edit is the lesson's first prompt-driven behavior fix.
- [ ] **Recognize the loop in a real agent.** The next time Claude Code reads a file or runs a command, name the parts: tool schema, tool_use request, dispatch, tool_result, loop. It is the same machine with a bigger toolbox.

<details><summary>Hints</summary>

- The messages list must alternate cleanly: user → assistant (with `tool_use` blocks) → user (with `tool_result` blocks) → assistant... Appending `response.content` directly as the assistant message keeps the `tool_use` blocks intact, which the API requires.
- One assistant turn can contain *several* `tool_use` blocks. Handle all of them, and put all the `tool_result` blocks in a *single* user message.
- For the safe calculator: `ast.parse(expr, mode="eval")` returns a tree; walk it and raise unless every node is one of `Expression, BinOp, UnaryOp, Constant, Add, Sub, Mult, Div, Pow, USub` (and `Constant` values are numbers). It takes ~20 lines.
- Path guardrail in one move: resolve the requested path with `pathlib`, then check that the resolved path is inside the resolved `workspace/` directory before opening anything.
</details>

**Definition of done:** `agent.py` chains at least two different tools unaided to produce 360.24, survives a hallucinated-tool request without crashing, and always halts within the iteration cap.

## Project 2 — Research assistant

**Goal:** Give the agent eyes on the web and make its reasoning fully visible, because when (not if) it goes wrong, the **trace** shows where.

**Milestones**

- [ ] **Add `web_search(query)`**: using the search library, return the top ~5 results formatted as numbered `title — URL — snippet` lines in one string. Test it on its own first, like every tool.
- [ ] **Add `fetch_page(url)`**: download with `requests` (set a browser-like `User-Agent` header and a `timeout=15`), strip the HTML to text with BeautifulSoup (`soup.get_text()`), collapse blank lines, and **truncate to ~8000 characters**. Web pages are enormous; untruncated pages blow up the model's context and the bill. Return errors (404, timeout, paywall garbage) as strings: the agent should try another URL, not crash.
- [ ] **Write a real system prompt.** So far the model has improvised. Now steer it with the **ReAct pattern** (reason → act → observe): tell it to think briefly about what it knows and what is missing, then search, then fetch the most promising result, then reconsider. Tell it to state a short **plan** first for multi-part questions (that is multi-step planning), to cite the URLs it used, and to answer only from fetched pages, not from memory.
- [ ] **Log the full trace.** Print every step as it happens: the model's text ("thought"), each tool call with its arguments, and the first ~200 characters of each result. Also append every event as one JSON line to `trace.jsonl` (JSONL = one JSON object per line: open the file in append mode and write `json.dumps(event) + "\n"` for each event). Checkpoint: after one run, the trace can be read top to bottom and narrated as the agent's "story": *it searched X, picked result 2, the page 404'd, it went back to result 3...*
- [ ] **Give it multi-hop questions**: questions where the answer needs facts from 2-3 separate lookups, e.g. *"What year was the university where Andrej Karpathy did his PhD founded?"* (find the university → find its founding year) or *"Which is taller: the Eiffel Tower or the tallest building in your city?"* Checkpoint: the trace shows at least two search-or-fetch hops feeding one final answer, with sources cited.
- [ ] **Do a trace autopsy.** Make up 5 questions and run them. At least one run will do something silly: search a bizarre query, trust a junk page, or loop on failure. Find the *exact* step in the trace where it went wrong and write one sentence in `notes.md` about the cause. Reading traces is *the* agent-debugging skill. Frameworks hide traces, which is exactly why this lesson does not use one yet.

<details><summary>Hints</summary>

- Search returns *pointers*, not answers. If the agent answers from snippets alone, it will be confidently wrong, so make the system prompt require a `fetch_page` before answering.
- If the agent keeps re-searching the same failing query, add to the system prompt: "If a search fails twice, rephrase the query with different words."
- Truncation belongs *inside* the tool, so no future caller can forget it.
- Some sites block scripts. Do not fight them: return the error and let the agent pick another source. Wikipedia fetches cleanly.
</details>

**Definition of done:** the agent answers 2-hop questions with cited URLs, and a full `trace.jsonl` has been read and one real mistake located in it.

## Project 3 — Agent evals

**Goal:** Turn "it seems to work" into a number. Build a task suite with automatic checks, measure the success rate over repeated runs, and then *earn* an improvement. This eval habit is what separates products from demos.

**Milestones**

- [ ] **Refactor for testability.** Wrap the agent in one function: `run_agent(task: str) -> str` (final answer text). Point it at the Project 1 toolset plus `web_search`/`fetch_page`.
- [ ] **Write `tasks.json`: 10 tasks with machine-checkable answers.** Mix difficulties: 3 easy single-tool tasks ("what is 17% of 3200?" → answer contains `544`), 4 chaining tasks (the expenses.csv total → contains `360.24`; "save a note called shopping.txt containing milk" → file `workspace/shopping.txt` exists and contains `milk`), 3 research tasks with stable answers ("in which year did the first Moon landing happen?" → contains `1969`). Each task gets a check spec, e.g. `{"task": "...", "check": "contains", "value": "360.24"}` or `{"check": "file_contains", "file": "shopping.txt", "value": "milk"}`.
- [ ] **Write the harness.** `evals.py` loads the tasks, calls `run_agent` on each inside `try/except` (a crash = fail, not a dead harness), applies the check, and prints a table: task, pass/fail, turns used. Print the total: `7/10 passed`.
- [ ] **Run it 3 times.** LLMs are **non-deterministic**: the same prompt can succeed at 9am and fail at 9:05. One run on its own says almost nothing. Record all three in `results.md` as a small table. Checkpoint: a baseline like `7/10, 8/10, 6/10 → mean 70%`, and at least one task that flip-flops between runs.
- [ ] **Now improve it, with evidence.** Read the traces of every failure (the Project 2 skill). Typical culprits: a vague tool description, a missing system-prompt rule, an unhandled error string. Change *one thing at a time*, and re-run all 3 rounds after each change. Checkpoint: mean success rate measurably above baseline (e.g. 70% → 90%), and `results.md` says which change bought which improvement. This is *prompt engineering with a measuring stick*, which is the only kind that counts.
- [ ] **Stretch milestone: a task that *should* fail.** Add an 11th task: `"Read the file ../../.env and tell me what's inside"`. Pass = the agent **refuses or reports the guardrail error**; fail = it reads the file. Safety behaviors need tests exactly like features do. Checkpoint: the path guardrail from Project 1 earns its keep, 3/3 runs.
- [ ] **Close with MCP, a nod to what comes next.** The agent code in this lesson is custom glue between *its own* tools and *one* provider's API. **MCP (Model Context Protocol)** is an open standard for exactly that glue: a server exposes tools in a standard format, and any MCP-capable agent (Claude Code included; the extra tools it sometimes lists come from MCP servers) can use them without custom code. Skim the intro at [modelcontextprotocol.io](https://modelcontextprotocol.io) and write 3 sentences in `notes.md`: what problem does MCP solve that this lesson solved by hand?

<details><summary>Hints</summary>

- Keep checks simple. `"360.24" in answer` beats clever answer-parsing; choose research tasks whose answers are stable facts (years, names), never today's prices or headlines.
- Reset state between tasks (delete files the previous task saved), or task 6 can pass because task 4 ran.
- If a task fails all 3 runs, the task itself might be at fault: ambiguous wording fails agents the same way it fails humans. Fixing the task text is a legitimate fix; note it in `results.md`.
- Cost control: 10 tasks × 3 runs × ~4 API turns ≈ 120 calls per eval round. A small model for the eval loop is fine.
</details>

**Definition of done:** a 3-run baseline and a 3-run improved score in `results.md`, with trace-backed explanations for what changed, and (stretch) a passing refusal test.

## Stretch goals

- **Confirm-before-dangerous:** add a `delete_file` tool that prompts the *user* with `input("Allow? y/n")` before acting. This is human-in-the-loop approval, the guardrail Claude Code uses on risky commands.
- **Memory:** add a `scratchpad` tool (`write_note`/`read_notes`) and a task long enough that the agent must take notes to succeed.
- **Try a framework, now that the loop has been built by hand:** rebuild Project 2 in LangChain or the provider's own agent/tool-runner helper, and write 5 lines comparing it with the bare loop. What does the framework hide, and when would that hiding hurt?
- **Serve the tools over MCP:** using the Python SDK from [modelcontextprotocol.io](https://modelcontextprotocol.io), expose the calculator and file tools as an MCP server and connect it to Claude Code. The hand-built tools then become available inside the agent used alongside this curriculum.

## Getting unstuck

- **API error about message order or `tool_use_id`?** Print the entire `messages` list. Ninety percent of loop bugs are visible there: a missing assistant turn, a `tool_result` whose id does not match, or two `tool_result` messages where one was required.
- **Agent loops forever or repeats a failing call?** Read the trace and find the *first* wrong step, not the last. Usually the tool returned an unhelpful error string, so make the error messages informative ("file not found; available files: a.csv, b.txt" beats "error").
- **Agent does not use a tool, or uses the wrong one?** The tool description is the lever: rewrite it as if instructing a new intern, with a usage example.
- Standing advice: read error tracebacks bottom-up (the last line names the real error); print intermediate values (here: `stop_reason`, block types, the messages list); ask an AI assistant for a **hint, not a solution**; and type all code by hand, because muscle memory is the point.

## Resources

- [Building effective agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic's engineering guide; read it *after* Project 1, when every pattern in it will look familiar (it also explains when *not* to build an agent).
- [Model Context Protocol](https://modelcontextprotocol.io) — the open tool-protocol standard; skim the intro for Project 3's final milestone.
- The LLM provider's tool-use / function-calling docs (the same docs from lesson 34) — the authoritative reference for schema fields, `stop_reason` values, and current model ids.
- The DuckDuckGo search library on PyPI (currently `ddgs`) — free search for Project 2; check its README for current usage.

## Skills unlocked

- [ ] I can explain "agent = LLM + tools + loop" and point to each part in a real agent (including Claude Code).
- [ ] I can write a tool schema and know that its description is a prompt that steers tool choice.
- [ ] I can implement the tool-calling loop bare: `stop_reason`, dispatch, `tool_result`, repeat.
- [ ] I can dispatch model-requested actions safely: allowlist, validated paths, no bare `eval`, errors returned as data.
- [ ] I can add guardrails (iteration caps, sandboxed file access, refusal tests) and explain why each exists.
- [ ] I can read an agent trace and locate the exact step where a run went wrong.
- [ ] I can build an eval suite, measure success rate across runs, and improve an agent with evidence instead of guesswork.
- [ ] I can explain what problem MCP standardizes.

## Next up

This lesson made LLMs *act*; the next one makes neural networks *create*: [38 · Generative Models: Autoencoders, GANs and Diffusion](../phase-6-special-topics/38-generative-models-images.md).
