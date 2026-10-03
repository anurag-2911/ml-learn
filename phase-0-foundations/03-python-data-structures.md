# 03 · Data Structures: Lists, Dicts and Real Text

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [01 · Set Up Your ML Workshop](01-environment-setup.md), [02 · Python Basics by Building Games](02-python-basics.md)

> Almost all data work — and machine learning is data work — boils down to: load a pile of stuff, organize it, count it, transform it. Python's containers (lists, dictionaries, sets, tuples) are how you hold and reshape that pile, and this week you graduate from single variables to processing an entire novel. You will count every word in *Alice's Adventures in Wonderland*, build a to-do app that remembers your tasks between runs, and crack encrypted messages by brute force. Without knowing it, you will also build your first tiny piece of real natural language processing. More on that at the end.

## What you will build

- **Project 1 — Word frequency counter** (`wordcount.py`): reads a whole public-domain book and prints the 20 most common words with their counts.
- **Project 2 — To-do list CLI** (`todo.py`): add, list, complete and delete tasks from the terminal; tasks survive restarts because they are saved to a file.
- **Project 3 — Caesar cipher toolkit** (`cipher.py`): encrypts and decrypts secret messages, plus a brute-force cracker that tries all 26 possible keys and figures out the answer itself.

## Concepts you will learn by doing

- **Lists**: ordered collections — indexing, slicing, `append`, sorting
- **Dictionaries (dicts)**: key → value lookups — `.get`, `.items`, the counting pattern
- **Sets**: collections of unique items with instant "is X in here?" membership tests
- **Tuples and unpacking**: fixed pairs like `("the", 1802)` and pulling them apart in one line
- **List and dict comprehensions**: building a new collection in a single readable line
- **Reading text files** and cleaning strings with `.split`, `.strip`, `.lower`

## Before you start

- You finished lessons 01 and 02: you can write and run a `.py` file, and you are comfortable with variables, `if`/`else`, loops, functions and `input()`.
- No new installs — everything this week is built into Python.

Open a terminal, activate your venv, make a work folder, and download your book from [Project Gutenberg](https://www.gutenberg.org) (a free library of public-domain books):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
mkdir -p work/03-python-data-structures
cd work/03-python-data-structures
curl -L -o alice.txt https://www.gutenberg.org/cache/epub/11/pg11.txt
head -n 20 alice.txt
```

`curl` downloads a file from a URL (`-L` follows redirects to wherever the file has moved, and `-o alice.txt` names the saved file), and `head` shows its first lines — you should see "The Project Gutenberg eBook of Alice's Adventures in Wonderland". If the URL ever stops working, go to gutenberg.org, search for "Alice's Adventures in Wonderland" (ebook #11), and copy the "Plain Text" link (under "Other formats & older devices") instead.

Checkpoint: `wc -w alice.txt` (word count) prints roughly **29,500 words**.

## Project 1 — Word frequency counter

**Goal:** Write `wordcount.py`, a program that reads `alice.txt` and prints the 20 most frequent words with their counts. This is the exact shape of a thousand real data tasks: load → clean → count → rank.

**Milestones**

- [ ] Read the whole book into one string. A *string* is text; a 164,000-character string is still just text. Start with:

  ```python
  with open("alice.txt", encoding="utf-8") as f:
      text = f.read()
  print(len(text), "characters")
  ```

  The `with open(...)` pattern opens a file and closes it automatically when the block ends. Checkpoint: you should see about **164,000 characters**.
- [ ] Lowercase everything with `text = text.lower()` so `The` and `the` count as the same word. Then split the string into a **list** of words with `words = text.split()` — `.split()` with no arguments cuts a string wherever there is whitespace. Checkpoint: `len(words)` is around **29,500**, `words[0]` is `"the"` (indexing: position 0 is the first item), and `words[:10]` (slicing: the first ten items) shows the opening words of the Gutenberg header.
- [ ] Clean the punctuation. Right now `"hearts."` and `"hearts"` are different words. `.strip(chars)` removes any of the given characters from both ends of a string (only the ends — that is what you want, so `don't` keeps its apostrophe). Use a **list comprehension**, which builds a new list from an old one in one line:

  ```python
  punct = '.,;:!?"\'()_-*[]'
  words = [w.strip(punct) for w in words]
  words = [w for w in words if w]   # drop empty strings left over
  ```

  Read the first line out loud: "a list of `w.strip(punct)` for each `w` in `words`". Checkpoint: `"alice" in words` prints `True`.
- [ ] Count with a **dict**. A dict maps keys to values, like a phone book maps names to numbers — here, word → how many times you have seen it. The counting pattern uses `.get(key, default)`, which looks up a key but returns the default instead of crashing when the key is missing:

  ```python
  counts = {}
  for w in words:
      counts[w] = counts.get(w, 0) + 1
  ```

  Checkpoint: `counts["alice"]` is about **383**, and `len(counts)` (the number of *unique* words) is around **4,200**.
- [ ] Rank and print the top 20. `counts.items()` gives you the dict as pairs like `("alice", 383)` — each pair is a **tuple**, a small immutable bundle of values. Sort those pairs by count, biggest first, using `sorted(..., key=..., reverse=True)`, then slice the top 20 and print them with **tuple unpacking**: `for word, count in top20:`. Checkpoint: `"the"` is #1 with a count around **1,800**, followed by "and", "to", "a", "of".
- [ ] The top 20 is mostly boring glue words. Filter them with a **set** — a collection of unique items whose whole job is fast membership tests: `stopwords = {"the", "and", "to", "a", "of", "she", "it", "in", "was", "i", ...}` (build your own from your boring top words; in NLP these are literally called *stopwords*). Skip any word that is `in stopwords` before ranking. Checkpoint: now "alice", "said", "queen", "king" and "turtle" appear in your top 20 — you can see what the book is about.

<details><summary>Hints</summary>

- For sorting pairs by count: `sorted(counts.items(), key=lambda pair: pair[1], reverse=True)`. A `lambda` is a tiny unnamed function; `pair[1]` is the count part of the tuple.
- `sorted()` returns a new list — it does not change the dict. Top 20 is just a slice: `[:20]`.
- If your counts look off, print `words[100:120]` and stare at it. Cleaning bugs are always visible in the data itself.
- `.strip` versus `.replace`: strip only touches the ends of each word; replace would delete characters everywhere. You want strip here.

</details>

**Definition of done:** `python wordcount.py` prints a tidy top-20 table with counts, "the" ranks #1 (~1,800) without the stopword filter, and "alice" is at or near the top with the filter on.

## Project 2 — To-do list CLI

**Goal:** Build `todo.py`, an interactive terminal app: it shows a menu, lets you add/list/complete/delete tasks, and — the important part — saves tasks to a file so they are still there tomorrow. This is your first program with *state that persists*.

**Milestones**

- [ ] Design the data first (a habit that will serve you forever): one task = one dict, like `{"title": "buy milk", "done": False}`, and all tasks = a **list of dicts**. This list-of-dicts shape is everywhere in data work — it is essentially a table, one dict per row.
- [ ] Write the menu loop: forever, print the commands (`add`, `list`, `done`, `delete`, `quit`), read one with `input()`, and dispatch with `if`/`elif`. `quit` breaks the loop. Checkpoint: you can type commands round and round without the program exiting.
- [ ] Implement `add` (ask for a title, `append` a new task dict) and `list` — print each task numbered from 1, with `[x]` or `[ ]` depending on `done`. Use `enumerate(tasks, start=1)`, which hands you *both* the number and the task via tuple unpacking: `for i, task in enumerate(tasks, start=1):`. Checkpoint: adding three tasks then `list` shows them numbered 1, 2, 3.
- [ ] Implement `done` and `delete`: ask which number, convert with `int()`, subtract 1 to get the list index (lists count from 0, humans from 1 — this off-by-one will bite you all year, meet it now). Set `task["done"] = True`, or remove with `tasks.pop(index)`. Reject numbers that are out of range instead of crashing. Checkpoint: completing task 2 shows `[x]` next to it; deleting it renumbers the rest.
- [ ] Persistence. Write a `save(tasks)` function that writes one line per task to `todo.txt` in a format you invent, e.g. `1|buy milk` where the `1`/`0` means done/not-done. Open with `open("todo.txt", "w")` and write lines ending in `"\n"`. Call `save` after every change. Checkpoint: `cat todo.txt` in another terminal shows your tasks.
- [ ] Write `load()`, called once at startup: if the file exists, read it line by line, `.strip()` each line (removes the invisible newline character — forgetting this is a classic bug), split on `"|"`, and rebuild the list of dicts. Checkpoint: add tasks, `quit`, run `python todo.py` again — **your tasks are back**.

<details><summary>Hints</summary>

- Split each saved line with `line.split("|", 1)` — the `1` means "split only at the first pipe", so a title containing `|` will not explode into extra pieces.
- Everything read from a file is a string: turn `"1"` back into a boolean with something like `done = (part == "1")`.
- To check the file exists before loading: `import os` then `os.path.exists("todo.txt")`. (Proper error handling arrives in lesson 04.)
- Guard user input with `choice.isdigit()` before calling `int()` on it.

</details>

**Definition of done:** Add three tasks, complete one, delete one, quit, rerun — the remaining tasks reappear with the correct done-marks.

## Project 3 — Caesar cipher toolkit

**Goal:** Build `cipher.py`: encrypt and decrypt messages with a shift cipher, then write a cracker that breaks any such message with no key at all. A *Caesar cipher* shifts every letter forward by a fixed amount (shift 3: a→d, b→e, ... x→a — it wraps around), and it is how Julius Caesar actually sent orders.

**Milestones**

- [ ] Play with character codes in the REPL (run `python3` with no filename to get the interactive `>>>` prompt; ctrl+d exits). `ord(ch)` gives a letter's number, `chr(n)` goes back. Checkpoint: `ord("a")` is **97**, `chr(97 + 1)` is `"b"`.
- [ ] Write `encrypt(message, shift)`. For each character: if it is a letter (`ch.isalpha()`), shift it and wrap around the alphabet with `%` (modulo — the remainder operator, which turns 26 back into 0); otherwise keep it unchanged (spaces stay spaces). Work in lowercase to keep it simple. The core move for one letter:

  ```python
  chr((ord(ch) - 97 + shift) % 26 + 97)
  ```

  Build the result as a list of characters and glue it with `"".join(...)` — or flex your new comprehension muscles and do the whole thing in one line. Checkpoint: `encrypt("abc xyz", 3)` returns `"def abc"`.
- [ ] Write `decrypt(message, shift)`. Insight: decrypting is just shifting the other way, so it is one line that reuses `encrypt`. Checkpoint: `decrypt(encrypt("hello world", 7), 7)` returns `"hello world"` for *every* shift you try.
- [ ] Wrap it in a small CLI: ask encrypt/decrypt/crack, the message, and (if needed) the shift. Trade secret messages with yourself.
- [ ] The cracker, part 1 — brute force. There are only 26 possible shifts, so try them all: loop `for shift in range(26):` and print each shift next to its decryption. Crack this message, which I encrypted for you with a secret shift: `ypp gsdr drosb roknc`. Checkpoint: exactly one of the 26 lines is English, and it is a Queen of Hearts quote.
- [ ] The cracker, part 2 — make the computer pick the winner. Build a set of common English words, e.g. `common = {"the", "and", "with", "you", "that", "have", "this", "off", "their"}`, and for each candidate decryption count how many of its words are in that set (split the candidate, test membership). Print the candidate with the highest score as THE answer. Checkpoint: your program prints the correct plaintext on its own, and tells you the secret shift was **10**. That moment — you gave it no key, it *figured it out* by scoring guesses against data — is the entire spirit of machine learning in miniature.

<details><summary>Hints</summary>

- Python's `%` handles negative numbers gracefully: `(-3) % 26` is `23`. So `decrypt` can simply call `encrypt` with a negative or `26 - shift` shift.
- Only shift letters. If you shift spaces and punctuation too, the output becomes soup and the cracker's word-matching breaks.
- For scoring, a comprehension plus `sum()` is elegant: `sum(1 for w in candidate.split() if w in common)`.

</details>

**Definition of done:** Round-trip encryption works for all 26 shifts, and `crack` recovers `ypp gsdr drosb roknc` automatically, reporting both plaintext and shift.

## Stretch goals

- **Bigrams:** in `wordcount.py`, count pairs of consecutive words ("said the", "the queen") instead of single words. Hint: `zip(words, words[1:])` pairs each word with its neighbor. What is the most common pair?
- **Compare two books:** download a second Gutenberg book, build a set of each book's vocabulary, and use set operations — `a & b` (words in both), `a - b` (words only in the first) — to compare their vocabularies.
- **To-do upgrades:** add a priority field (high/medium/low) to each task dict and a `sort` command that lists tasks by priority using `sorted` with a `key`.
- **Frequency-analysis cracking:** crack a cipher without a word list, using Project 1's skill — in English, "e" is the most common letter, so find the most common letter in the ciphertext and infer the shift.

## If you get stuck

- `KeyError: 'alice'` means you asked a dict for a key it does not have — usually a cleaning problem (the key is really `"alice."` or `"Alice"`). Print `list(counts.keys())[:20]` and look.
- `IndexError: list index out of range` is almost always the off-by-one: humans count from 1, lists from 0. Print the index and `len(tasks)` right before the failing line.
- When text processing misbehaves, print a small sample of the actual data (`words[:20]`, `repr(line)`) — `repr` reveals invisible characters like `'\n'` that `print` hides.
- Read error messages from the bottom up: the last line says what went wrong, the lines above say where.
- Ask your AI assistant for a **hint, not a solution** ("why would `.strip()` leave punctuation in the middle of a word?"), and type every line yourself — copy-paste teaches your clipboard, not you.

## Resources

- [Project Gutenberg](https://www.gutenberg.org) — 70,000+ free public-domain books; your source for real text now and for training data later.
- [Python tutorial: Data Structures](https://docs.python.org/3/tutorial/datastructures.html) — the official chapter on lists, dicts, sets, tuples and comprehensions; skim after the projects to fill gaps.

## Skills unlocked

- [ ] I can read a text file into a string and clean it with `.lower`, `.strip` and `.split`.
- [ ] I can index and slice a list, and explain why `words[0]` is the first item.
- [ ] I can count anything with a dict using the `.get(key, 0) + 1` pattern.
- [ ] I can sort `(key, value)` tuples by value and take the top N.
- [ ] I can use a set for fast membership tests and explain when to prefer it over a list.
- [ ] I can write a list comprehension with a condition and read one out loud.
- [ ] I can design a record as a dict, keep records in a list, and save/load them from a file.
- [ ] I can unpack tuples in a `for` loop (`for word, count in ...`).

## Next up

One reveal before you go: that word-count dict from Project 1 is called a **bag of words** — a real, foundational technique in natural language processing. In lesson [17](../phase-2-classical-ml/17-naive-bayes-spam-svm.md) you will use exactly it to build a spam filter, so congratulations: you just did baby NLP. Next, your programs get more structure — classes, JSON and proper error handling in [04 · Classes, Files, JSON and Errors](04-python-oop-files-errors.md).
