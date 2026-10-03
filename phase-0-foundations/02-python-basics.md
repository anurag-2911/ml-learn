# 02 · Python Basics by Building Games

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [01 · Set Up Your ML Workshop](01-environment-setup.md)

> This lesson is where real programming starts, and it teaches Python the way the whole curriculum works: by building. Instead of chapters to read about Python, there are three small games to make. They run in the terminal, and every concept (variables, loops, functions) appears exactly when it is needed to make a game work. These same building blocks are what all machine learning code is made of: a training loop is just a `while` loop, and a model is just functions calling functions. One rule holds for the whole week: **type every line by hand, never paste**, because the typing is the learning.

## What this lesson builds

- **Project 1 — Number guessing game** (`guess.py`): the computer picks a secret number from 1 to 100, the player guesses, and the computer answers "higher" or "lower", counts the attempts, and offers a rematch.
- **Project 2 — Unit converter** (`convert.py`): a menu-driven tool that converts km↔miles, kg↔lb, and °C↔°F, built from six small reusable functions.
- **Project 3 — Rock paper scissors** (`rps.py`): a best-of-5 match against the computer with live score tracking.

## Concepts covered

- **Variables**: named boxes that store a value, like `score = 0`.
- **Types**: the kinds of values. Whole numbers are `int`, decimals are `float`, text is `str`, and true/false values are `bool`.
- **`print()` and `input()`**: showing text to the player and reading what they type.
- **f-strings**: a way to slot values into text, like `f"You took {attempts} tries"`.
- **`if` / `elif` / `else`**: letting the program choose between different paths.
- **`while` and `for` loops**: repeating steps until something happens, or once per item.
- **Functions**: named, reusable chunks of code that take inputs (parameters) and hand back a result (return value).
- **`import random`**: borrowing ready-made code (a *module*) from Python's built-in library.

## Before starting

This lesson needs the setup from [lesson 01](01-environment-setup.md): the terminal, VS Code, and the virtual environment (the private Python installation) at the repo root. Nothing new needs to be installed this week; plain Python is enough.

Open a terminal and run:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
mkdir -p work/02-python-basics
code .
cd work/02-python-basics
```

`mkdir -p` creates the work folder for this lesson, and `code .` opens the whole repo in VS Code. Always open the repo root, as in lesson 01, so that VS Code can find the venv. The last line moves the terminal into the work folder. All three game files live in `work/02-python-basics/`. To create one, right-click that folder in VS Code's Explorer panel and choose **New File**.

Quick smoke test: in VS Code, create a file named `hello.py` in `work/02-python-basics/` containing exactly one line, `print("hello, world")`, then run it from the terminal:

```bash
python3 hello.py
```

If the terminal prints `hello, world`, everything is ready. All week, the work follows that two-step rhythm: edit the file in VS Code, then run it in the terminal.

## Project 1 — Number guessing game

**Goal:** the computer secretly picks a number from 1 to 100; the player guesses until they find it, and the program says how many attempts it took. This one project teaches variables, types, `input()`, `if`, and `while`.

**Milestones**

- [ ] Create `guess.py`. On the first line write `import random`. This loads Python's *random module*, a collection of ready-made functions for randomness. On the next line write `secret = random.randint(1, 100)`. This is a *variable*: the name `secret` now stores whatever number `randint` picked. Temporarily add `print(secret)` and run the file a few times. **Checkpoint: each run prints a different number between 1 and 100.**
- [ ] Read one guess from the player: `guess = input("Your guess: ")`. Be careful: `input()` always gives back text (a `str`), even if the player types `42`. The text `"42"` and the number `42` are different *types* and cannot be compared. Convert it: `guess = int(input("Your guess: "))`. Add `print(type(guess))` once to confirm, then remove it. **Checkpoint: running the file asks for a guess and `type(guess)` prints `<class 'int'>`.**
- [ ] Compare guess to secret with an `if` / `elif` / `else` chain: print "Too low!", "Too high!", or "You got it!". Note that *equal to* in Python is `==` (double equals); a single `=` means "store this value". **Checkpoint: run three times; each of the three messages is reachable.**
- [ ] One guess is not a game. Wrap the ask-and-compare part in a `while` loop (a block that repeats as long as its condition holds), so that the program keeps asking until the guess is right. Indentation (the 4 spaces at the start of a line) is how Python knows which lines are inside the loop; it is grammar, not decoration. **Checkpoint: the game keeps asking until the player wins, then stops.**
- [ ] Count attempts: create `attempts = 0` before the loop, and inside it `attempts = attempts + 1` (or the shorthand `attempts += 1`). On a win, print an *f-string*, which is text with variables slotted in: `print(f"Got it in {attempts} attempts!")`. **Checkpoint: guess by always splitting the range in half (50, then 75 or 25, …). Guessing this way wins in 7 attempts or fewer. That halving trick is called binary search.**
- [ ] Replay option: after a win, ask `again = input("Play again? (y/n) ")` and wrap the entire game in an outer `while` loop that continues while the answer is `"y"`. Delete the leftover `print(secret)`, since it gives the answer away. **Checkpoint: after a game ends, answering `y` starts a fresh round with a new secret number; `n` ends the program politely.**

<details><summary>Hints</summary>

- Comparing text to a number (`"50" > 42`) crashes with a `TypeError`. The fix is the `int(...)` conversion around `input(...)`.
- A clean loop shape: `while guess != secret:` ask again inside the loop. To make it run at least once, either ask once before the loop, or start with `guess = 0` (an impossible value).
- For the replay: pick a **new** secret inside the outer loop; otherwise round two has the same answer as round one.
- If the player types letters instead of a number, the program crashes. That is fine for now; [lesson 04](04-python-oop-files-errors.md) shows how to handle crashes properly.

</details>

**Definition of done:** a full game with higher/lower feedback, an attempt count in the win message, and a working replay that uses a fresh secret number.

## Project 2 — Unit converter

**Goal:** a menu-driven converter for km↔miles, kg↔lb, and °C↔°F. The real lesson here is *functions*: small named machines that are built once and reused, which is exactly how all larger programs (and ML libraries) are organized.

**Milestones**

- [ ] Create `convert.py` and write the first function:

  ```python
  def km_to_miles(km):
      return km * 0.621371
  ```

  `def` defines a function named `km_to_miles`; `km` is its *parameter* (the input slot); `return` sends the result back to whoever called it. Defining a function runs nothing; the function has to be *called*. Add `print(km_to_miles(5))` at the bottom and run the file. **Checkpoint: the terminal prints `3.106855`.**
- [ ] That number has too many decimal places. Print it through an f-string with a format rule: `print(f"{km_to_miles(5):.2f} miles")`. The `:.2f` means "show 2 decimal places". **Checkpoint: the terminal prints `3.11 miles`.**
- [ ] Write the other five functions, one per conversion: `miles_to_km` (÷ 0.621371), `kg_to_lb` (× 2.20462), `lb_to_kg`, `c_to_f` (× 9/5, + 32), `f_to_c`. Test each one with a call whose result can be checked without a calculator. **Checkpoint: `c_to_f(100)` returns `212.0` and `kg_to_lb(1)` returns about `2.20`.**
- [ ] Print a numbered menu (options 1–6 plus `q` to quit) using one `print()` per line, and read the player's pick with `input()`. No conversion logic here yet: just show the menu and print back what they chose.
- [ ] Wire it up: an `if`/`elif` chain on the choice; each branch asks for a value with `input()`, converts it with `float(...)` (like `int(...)` but for decimals: `float("3.5")` works, `int("3.5")` crashes), calls the matching function, and prints the result to 2 decimal places. **Checkpoint: choosing km→miles and entering `10` prints `6.21`.**
- [ ] Wrap menu-and-convert in a `while` loop that repeats until the player enters `q`. **Checkpoint: three different conversions run in a row, and then the program quits cleanly.**

<details><summary>Hints</summary>

- Keep a strict division of labor: functions **return** numbers and never print; the menu code prints and never calculates. This separation is a habit that will matter for the rest of the curriculum.
- Everything from `input()` is a `str`. Convert the menu choice never (compare to `"1"`, `"2"`, …, `"q"`) and the measurement always (`float(...)`).
- Getting `1.6666`-ish from `c_to_f(100)`? Check operator order: multiply by 9, divide by 5, **then** add 32. Parentheses make it unambiguous.

</details>

**Definition of done:** all six conversions work through the looping menu, results show 2 decimals, and `q` exits without an error.

## Project 3 — Rock paper scissors

**Goal:** best-of-5 against the computer with score tracking. This combines everything: random choices, string handling, a function with real decision logic, and loops inside loops.

**Milestones**

- [ ] Create `rps.py`. Make a list of the moves, `moves = ["rock", "paper", "scissors"]` (a *list* is an ordered collection in square brackets; lesson 03 explores lists fully, and today only this one is needed), and let the computer pick with `random.choice(moves)`. **Checkpoint: running the file prints a random move, different across runs.**
- [ ] Read the player's move with `input()` and clean it up: `player = input("rock, paper or scissors? ").lower().strip()`. Those are *string methods* (functions attached to text with a dot) that lowercase it and trim spaces, so `" Rock "` still counts. If the cleaned move is not in the list (`if player not in moves:`), say so and ask again.
- [ ] Write `def decide_winner(player, computer):` returning exactly one of `"player"`, `"computer"`, or `"tie"`. Three rules to encode: rock beats scissors, scissors beats paper, paper beats rock. Test it exhaustively with a `for` loop (a loop that runs once per item of a list) nested inside another:

  ```python
  for p in moves:
      for c in moves:
          print(p, "vs", c, "->", decide_winner(p, c))
  ```

  **Checkpoint: exactly 9 lines, exactly 3 of them ties (the diagonal: rock-rock, paper-paper, scissors-scissors).** Delete the test loop once it passes.
- [ ] Build the match: `player_score = 0`, `computer_score = 0`, and a `while` loop that keeps playing rounds until someone reaches 3 wins (that is what best-of-5 means). Each round: get both moves, call `decide_winner`, update the right score, and print the state with an f-string like `f"You {player_score} - {computer_score} Computer"`. Ties replay the round and change no score. **Checkpoint: a full match always ends with one side on exactly 3 wins.**
- [ ] Finish it: announce the match winner, and add a play-again loop like in Project 1. Scores must reset to 0 for each new match. **Checkpoint: two matches in a row, and the second one starts 0–0.**

<details><summary>Hints</summary>

- Inside `decide_winner`, first handle `player == computer` (tie). After that, only the three ways the *player* wins need to be spelled out; `else` covers every computer win.
- A condition can combine two comparisons with `and`: `if player == "rock" and computer == "scissors":`.
- The "ask again on invalid input" part fits in its own small `while` loop around the `input()` line.

</details>

**Definition of done:** a best-of-5 match with input cleaning, correct scorekeeping, a declared winner, and replay with reset scores.

## Stretch goals

- **A computer that learns the player.** Track how often the player has picked each move using three counter variables (`rock_count`, `paper_count`, `scissors_count`). After a few rounds, make the computer play the counter to the player's most frequent move instead of a random one. Predicting future behavior from counted past behavior is the first genuinely ML-flavored idea in the course.
- **Reverse the guessing game.** The player thinks of a number; the computer guesses by always proposing the middle of the remaining range and narrowing it from the player's `h`/`l`/`c` answers. It should never need more than 7 guesses.
- **Converter, but smarter.** Add two more conversion pairs of any kind (hours↔minutes, GB↔MB, …) and print `"That doesn't look like a number"` instead of crashing when the value input is not numeric (hint: strings have an `.isdigit()` method, though it rejects decimals; finding a way around that is part of the challenge).
- **Guess history.** In Project 1, build up a string of all guesses (`history = history + f"{guess} "`) and show it when the game ends.

## Getting unstuck

- **Read the error from the bottom up.** Python prints a *traceback* when it crashes; the last line names the problem (`NameError`, `TypeError`, …) and the line just above it shows where. The most common errors in the first week are `SyntaxError` (often a missing `:` at the end of an `if`/`while`/`def` line), `IndentationError` (spaces at the start of a line do not match), `NameError` (a typo: `atempts` is not `attempts`), and `TypeError` (mixing text and numbers: an `int()` or `float()` is missing).
- **Print values to find the truth.** Unsure what a variable holds? Add `print(guess, type(guess))` right before the crashing line. Debugging by printing is a respected professional technique, not a beginner hack.
- **Loop never ends?** Press `Ctrl+C` (the Control key, not Command, even on a Mac) to stop the program, then check: does anything inside the loop ever change the loop's condition?
- **Ask an AI assistant for a hint, not a solution.** Say "give me a hint, don't write the code". A pasted solution today is a debt repaid with interest in lesson 22.
- **Type all code by hand.** This includes the snippets in this file. Muscle memory is real.

## Resources

- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) — free online book; chapters 1–4 cover exactly this lesson's ground with more examples, in the same practical spirit.
- [The official Python tutorial](https://docs.python.org/3/tutorial/) — chapters 3 ("An Informal Introduction") and 4 ("More Control Flow Tools") as a second angle on numbers, strings, `if`, loops, and functions.
- On the same site, the *Library Reference* page for the `random` module lists everything `random` can do beyond `randint` and `choice`.

## Skills unlocked

- [ ] I can create a variable and explain the difference between `int`, `float`, `str`, and `bool`.
- [ ] I can read user input, convert it to the right type, and print results with an f-string.
- [ ] I can branch a program with `if` / `elif` / `else`, including conditions joined with `and`.
- [ ] I can repeat work with `while` and `for` loops, and I know that `Ctrl+C` (Control, not Command, even on a Mac) stops an endless one.
- [ ] I can write a function with parameters and a return value, and I keep calculation and printing separate.
- [ ] I can use `import` to pull in a module and call its functions, like `random.randint`.
- [ ] I can read a traceback from the bottom up and fix the four classic beginner errors without help.

## Next up

The games in this lesson handled one value at a time; the next lesson handles thousands, with lists and dictionaries applied to real text: [03 · Data Structures: Lists, Dicts and Real Text](03-python-data-structures.md).
