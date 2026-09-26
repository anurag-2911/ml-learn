# 04 · Classes, Files, JSON and Errors

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [02 · Python Basics by Building Games](02-python-basics.md), [03 · Data Structures: Lists, Dicts and Real Text](03-python-data-structures.md)

> Every ML library you will ever use — scikit-learn, PyTorch, Hugging Face — hands you *objects*: things like `model.fit(data)` and `model.predict(x)`. That dot-syntax is object-oriented programming (OOP), and after this week it will stop looking like magic. You will also learn how to save data to disk as JSON, and how to read Python error messages calmly instead of panicking. You'll build a contact book that survives restarts, a flashcard trainer with a baby spaced-repetition brain, and a `Vector2D` class that is your on-ramp to the math phase. This is the last "pure Python" lesson — after this, the ML ecosystem opens up.

## What you will build

- **Project 1 — Contact book**: a command-line app built from a `Contact` class and a `ContactBook` class, with add/search/delete, saved to a JSON file so your contacts survive between runs.
- **Project 2 — Flashcard trainer**: a quiz app that stores cards in JSON, tracks a score per card, and shows you the cards you keep missing more often — a tiny spaced-repetition algorithm.
- **Project 3 — Vector2D class**: a small math class with add, subtract, scale, dot product and length methods — the exact shape of thing you'll rebuild at scale in [lesson 08](../phase-1-math/08-linear-algebra-by-code.md).

## Concepts you will learn by doing

- **Class** — a blueprint for making objects that bundle data and functions together.
- **`__init__` and `self`** — how an object gets built, and how it refers to its own data.
- **Methods** — functions that live on an object (`book.add(...)` instead of `add(book, ...)`).
- **Why ML libraries are class-based** — the `model = Model(); model.fit(X); model.predict(x)` pattern you'll use for years.
- **JSON** — a text format for saving lists/dicts to a file and loading them back.
- **try/except** — running code that might fail without crashing, and handling the failure.
- **Reading tracebacks** — decoding Python's error messages bottom-up, calmly.
- **Modules, imports, and pip** — splitting code across files and installing other people's packages.

## Before you start

1. You finished lesson 03 and are comfortable with lists, dicts, loops and functions. If `{"name": "Ada", "tags": ["math", "hero"]}` looks readable to you, you're ready.
2. Open your repo in VS Code and activate your virtual environment (the isolated Python sandbox you made in lesson 01):

   ```bash
   cd ~/ml/ml-learn
   source .venv/bin/activate
   ```

3. Create this lesson's work folder:

   ```bash
   mkdir -p work/04-python-oop-files-errors
   cd work/04-python-oop-files-errors
   ```

4. Practice your first **pip install** (pip is Python's package installer — it downloads third-party code from the internet into your venv). We'll install `rich`, a popular package for pretty terminal output, and use it as an optional garnish later:

   ```bash
   pip install rich
   python3 -c "from rich import print; print('[bold green]pip works![/bold green]')"
   ```

   If you see bold green text, you just used someone else's code — that's 90% of real ML work.

No datasets needed this week. Everything is data you create.

## Project 1 — Contact book

**Goal** — Build a contact book app from two classes, then make it persist to disk with JSON, then make it survive a missing or corrupted file. This is the full life cycle of real-world code: works → saves → loads → handles disaster.

**Milestones**

- [ ] **Write your first class.** In `contacts.py`, define a `Contact` class. A class is a blueprint: it describes what data and behavior every contact will have. Here's the anatomy — type it, run it, poke at it:

  ```python
  class Contact:
      def __init__(self, name, phone, email):
          self.name = name      # self = "this particular contact"
          self.phone = phone
          self.email = email

  c = Contact("Ada Lovelace", "555-0101", "ada@example.com")
  print(c.name)   # -> Ada Lovelace
  ```

  `__init__` is the setup function Python runs automatically when you write `Contact(...)`. `self` is how the object refers to itself — `self.name = name` means "store this name on *me*". Checkpoint: you can create two different contacts and print each one's name, and they don't interfere with each other.
- [ ] **Add a method.** Give `Contact` a method `describe(self)` that returns a one-line string like `"Ada Lovelace | 555-0101 | ada@example.com"`. A method is just a function defined inside a class — you call it as `c.describe()`. Checkpoint: `print(c.describe())` shows the formatted line.
- [ ] **Build the `ContactBook` class.** It should hold a list of contacts (`self.contacts = []` in `__init__`) and have three methods: `add(contact)`, `search(term)` (returns every contact whose name contains the term, ignoring case), and `delete(name)`. Pause and notice the pattern: `book.add(...)`, `book.search(...)` — this is *exactly* the shape of `model.fit(...)`, `model.predict(...)` in scikit-learn. ML models are objects that hold data (learned parameters) and expose methods that use it. Checkpoint: you can add 3 contacts, search for one by a partial name, and delete one — all in memory.
- [ ] **Make it an interactive app.** Add a loop that shows a menu (add / search / delete / list / quit) and calls the right method — you built loops like this in lesson 02. Checkpoint: the app runs and all five options work, but everything vanishes when you quit. That's the problem we fix next.
- [ ] **Save to JSON.** JSON (JavaScript Object Notation) is a plain-text format that looks almost exactly like Python dicts and lists, which makes it the standard way to save structured data. Give `ContactBook` a `save(self, filename)` method:

  ```python
  import json

  data = [{"name": c.name, "phone": c.phone, "email": c.email}
          for c in self.contacts]
  with open(filename, "w") as f:
      json.dump(data, f, indent=2)
  ```

  Call it after every change. Checkpoint: open `contacts.json` in VS Code — you can read your contacts as human-readable text.
- [ ] **Load on startup.** Write a `load(self, filename)` method that does the reverse: `json.load(f)` gives you back a list of dicts, and you rebuild `Contact` objects from them. Call it once when the app starts. Checkpoint: add a contact, quit, rerun the app, and the contact is still there.
- [ ] **Break it on purpose, then handle it.** Delete `contacts.json` and run the app — it crashes. Read the traceback (Python's error report) *bottom-up*: the last line names the error (`FileNotFoundError`), the lines above show where it happened. Now open `contacts.json`, type garbage into it, run again — a different crash (`json.JSONDecodeError`). Wrap the loading code in try/except, which means "attempt this; if a specific error occurs, do this instead of crashing":

  ```python
  try:
      with open(filename) as f:
          data = json.load(f)
  except FileNotFoundError:
      data = []   # first run ever - start empty
  except json.JSONDecodeError:
      print("Warning: contacts.json is corrupted, starting fresh.")
      data = []
  ```

  Checkpoint: the app starts cleanly with no file, and with a corrupted file, and never crashes on startup.

<details><summary>Hints</summary>

- If Python complains `TypeError: describe() takes 0 positional arguments but 1 was given`, you forgot `self` as the first parameter of the method — Python passes the object in automatically.
- For case-insensitive search, compare `term.lower() in c.name.lower()`.
- `json.dump` can't save `Contact` objects directly — that's why you convert to plain dicts first. Going dict → object on load: `Contact(d["name"], d["phone"], d["email"])`.
- Save inside `add` and `delete` (call `self.save(...)` at the end of each) so you can never forget.

</details>

**Definition of done** — Two classes, a working menu app, contacts persist across runs, and startup survives both a missing and a corrupted JSON file.

## Project 2 — Flashcard trainer

**Goal** — Build a quiz app whose cards live in a JSON file, where every card tracks how often you get it right — and cards you miss come back more often. This is a real (tiny) adaptive algorithm, and a taste of the "update a score based on feedback" loop that is the heartbeat of machine learning.

**Milestones**

- [ ] **Design the data first.** Create `cards.json` by hand in VS Code: a list of card dicts, each with `"question"`, `"answer"`, and `"score"` (start every score at 0). Put in 8–10 cards about things from lessons 01–03 (e.g. Q: "What symbol starts a Python comment?" A: "#"). Checkpoint: `python3 -c "import json; print(len(json.load(open('cards.json'))))"` prints your card count.
- [ ] **Write a `Card` class and a `Deck` class.** `Card` holds question, answer, score. `Deck` loads cards from JSON in `__init__` (reuse your try/except pattern from Project 1) and has a `save()` method. Checkpoint: you can load the deck and print every question.
- [ ] **Build the quiz loop.** Pick a random card (`import random`, then `random.choice(...)` — you met this in lesson 02), show the question, `input()` the answer, compare ignoring case and surrounding spaces (`.strip().lower()`). Right answer: score + 1. Wrong: score − 1, and show the correct answer. Save after every card. Checkpoint: play 5 cards, quit, open `cards.json` — the scores changed.
- [ ] **Make missed cards come back more often.** Replace `random.choice` with *weighted* choice: give each card a weight like `max(1, 5 - card.score)`, so a card with score −3 (weight 8) is 4× more likely to appear than one with score 3 (weight 2). `random.choices(cards, weights=weights)[0]` does the weighted pick. Congratulations — you wrote a baby spaced-repetition algorithm, the idea behind apps like Anki. Checkpoint: deliberately fail one card 5 times, then play 10 rounds — that card should show up noticeably more than any other.
- [ ] **Add a stats command.** Typing `stats` during the quiz prints every card with its score, worst first (sort with `sorted(..., key=...)` from lesson 03). Optional garnish: print it as a table with the `rich` package you installed. Checkpoint: stats shows your weakest cards at the top.

<details><summary>Hints</summary>

- Structure tip: `Deck.pick_card()` returns a card, the quiz loop stays outside the class. Classes hold data + rules; the loop is just the driver.
- If every card seems equally likely, print the weights list right before choosing — seeing the numbers usually reveals the bug instantly (a habit that will save you constantly in ML).
- `random.choices` (plural, with an s) takes weights; `random.choice` (singular) does not.
- Don't let weights hit zero or go negative — that's what the `max(1, ...)` guard is for.

</details>

**Definition of done** — Quiz runs, scores persist to JSON between sessions, missed cards demonstrably appear more often, and `stats` ranks your weak spots.

## Project 3 — Vector2D class

**Goal** — Build a small class representing a 2D vector (an arrow with an x and y component). In [lesson 08](../phase-1-math/08-linear-algebra-by-code.md) you will discover that vectors are the native language of ML — this class is your bridge there.

**Milestones**

- [ ] **The basics.** In `vector2d.py`, write a `Vector2D` class with `__init__(self, x, y)`. Add a `__repr__(self)` method returning `"Vector2D(3, 4)"` — `__repr__` is a special method Python calls automatically when you `print` the object, so you see something useful instead of `<__main__.Vector2D object at 0x7f...>`. Checkpoint: `print(Vector2D(3, 4))` shows `Vector2D(3, 4)`.
- [ ] **Add and subtract.** `add(other)` and `subtract(other)` should each return a *new* `Vector2D` (add the x's, add the y's) rather than modifying anything. Checkpoint: `Vector2D(1, 2).add(Vector2D(3, 4))` prints `Vector2D(4, 6)`.
- [ ] **Scale, dot product, length.** `scale(k)` multiplies both components by a number. `dot(other)` returns `self.x * other.x + self.y * other.y` — a single number measuring how much two vectors point the same way (this one operation powers most of deep learning; for now just make it work). `length()` returns `math.sqrt(self.dot(self))` — note a method calling another method on the same object via `self`. Checkpoint: `Vector2D(3, 4).length()` is exactly `5.0`, and `Vector2D(1, 0).dot(Vector2D(0, 1))` is `0` (perpendicular arrows share no direction).
- [ ] **Make it a module and import it.** A module is just a `.py` file you import from. Create a second file `demo.py` in the same folder with `from vector2d import Vector2D` at the top, and use the class from there. This is how all Python code is organized — `from sklearn.neighbors import KNeighborsClassifier` (which you'll type in lesson 11) is the same move at a bigger scale. Checkpoint: `python3 demo.py` runs your vector demo with zero vector code inside `demo.py`.
- [ ] **Test it like you mean it.** In `demo.py`, write 5+ checks using `assert`, which crashes loudly if a condition is false — e.g. `assert Vector2D(3, 4).length() == 5.0`. Checkpoint: `python3 demo.py` prints "all tests passed" and asserting something false produces an `AssertionError` traceback you can now read bottom-up without flinching.

<details><summary>Hints</summary>

- Inside `add`, build the result with `return Vector2D(self.x + other.x, self.y + other.y)` — yes, a class can create new instances of itself.
- `__repr__` must *return* a string, not print one. Use an f-string: `f"Vector2D({self.x}, {self.y})"`.
- `import math` at the top of the file gives you `math.sqrt`.

</details>

**Definition of done** — All five methods plus `__repr__` work, the class lives in its own module imported by `demo.py`, and your asserts all pass.

## Stretch goals

- **Operator overloading**: rename `add` to the special method `__add__` and `subtract` to `__sub__`, so `v1 + v2` just works. (Search the Python classes tutorial below for "special methods".) NumPy arrays do exactly this — next lesson you'll use it constantly.
- **Contact book edit command**: add an `edit` option that finds a contact and updates one field, with a friendly message when the name isn't found.
- **Flashcard categories**: give cards a `"topic"` field and let the user quiz a single topic.
- **A `fit`/`predict` toy**: write a class `MeanPredictor` with `fit(self, numbers)` (stores the average) and `predict(self)` (returns it). Ten lines, and you've written your first scikit-learn-shaped model.

## If you get stuck

- **Read the traceback bottom-up.** The last line is the error type and message; the lines above trace where it happened, most recent call last. `AttributeError: 'Contact' object has no attribute 'phone'` usually means a typo in `__init__` (`self.phone` vs `self.phon`).
- **`NameError: name 'self' is not defined`** means you used `self` outside a class, or forgot it in a method's parameter list.
- **JSON acting weird?** Print the data right before `json.dump` and right after `json.load`, and look at it. JSON only holds dicts, lists, strings, numbers, booleans and `null` — never your objects directly.
- **Print early, print often.** When behavior confuses you, print the values and types in play (`print(type(x), x)`). This habit scales all the way to debugging neural networks.
- Ask an AI assistant for a **hint**, not a solution — e.g. "I'm getting this traceback in my ContactBook.load method, give me a hint but don't write the code." And type all code yourself; muscle memory is half the learning.

## Resources

- [Python classes tutorial](https://docs.python.org/3/tutorial/classes.html) — the official walkthrough of classes, `self`, and special methods; skim after Project 1 to consolidate.
- [Automate the Boring Stuff](https://automatetheboringstuff.com/) — free book; its chapters on files, JSON and debugging are excellent backup reading if any project milestone feels shaky.

## Skills unlocked

- [ ] I can write a class with `__init__`, attributes and methods, and explain what `self` means.
- [ ] I can look at `model.fit(X)` and say what kind of thing `model` is and what the dot does.
- [ ] I can save Python data to a JSON file and load it back into objects.
- [ ] I can wrap risky code in try/except and handle specific error types.
- [ ] I can read a traceback bottom-up and locate the failing line.
- [ ] I can split code into modules and import between files.
- [ ] I can pip install a package into my venv and use it.

## Next up

Time to trade hand-rolled loops for the workhorse of all ML computation — arrays: [05 · NumPy: Thinking in Arrays](05-numpy.md).
