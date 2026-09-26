# 01 · Set Up Your ML Workshop

**Phase 0 — Foundations** · Estimated time: 2-4 days · Prerequisites: none — this is day one

> Every machine learning expert works out of the same basic workshop: a terminal, Python, a code editor, and git. Today you build that workshop on your own machine, and by the end of it you will have written and run your first Python program, made your first chart, and pushed your first code to GitHub. Nothing here is throwaway — this exact setup is what you will use for all 42 lessons, from printing a multiplication table today to training your own GPT later. Go slowly, type everything yourself, and celebrate each checkpoint.

## What you will build

- **Project 1 — Workshop setup:** a working Python environment inside WSL2 Ubuntu, with a virtual environment at the repo root and the core ML libraries installed and verified.
- **Project 2 — Hello, data:** `hello.py`, a small Python script that prints a multiplication table, which you run from the terminal.
- **Project 3 — First notebook:** a Jupyter notebook that plots the curve y = x², run inside VS Code — then committed and pushed so it is visible on GitHub.

## Concepts you will learn by doing

- The **terminal**: a text window where you type commands to control your computer (`cd`, `ls`, `mkdir`).
- **WSL2**: a full Ubuntu Linux system running inside your Windows machine — everything in this curriculum happens there.
- **Python** and **pip**: the programming language of ML, and its tool for installing extra packages.
- **Virtual environments (venv)**: a private folder of Python packages for one project, so different projects never fight over versions.
- **VS Code with the WSL, Python and Jupyter extensions**: your editor, connected into Linux.
- **git basics**: `add`, `commit`, `push` — saving snapshots of your work and backing them up to GitHub.
- **This repo's layout**: numbered phase folders hold the lessons (read them, don't edit them); a `work/` folder holds all the code you write.

## Before you start

You need:

- A Windows PC with **WSL2 Ubuntu** installed and **VS Code** on Windows (you have both — confirmed).
- This repository cloned at `/home/anurag/ml/ml-learn` (you have this too).

Two rules for the whole curriculum:

1. **Every command runs inside WSL** (the Ubuntu terminal), never in Windows PowerShell or CMD. Open Ubuntu from the Start menu, or a WSL terminal inside VS Code.
2. **All your code lives in `work/`**, one subfolder per lesson. Your folder for this lesson will be `work/01-environment-setup/` — you will create it in Project 2. The lesson files like this one are your instructions; the `work/` folder is your workbench.

No pip installs are needed before Project 1 — installing things *is* Project 1.

## Project 1 — Workshop setup

**Goal:** Install Python 3, pip and venv on Ubuntu, create a virtual environment at the repo root, install the four core libraries, and prove they all import. This is the foundation every later lesson stands on.

**Milestones**

- [ ] Open an Ubuntu (WSL) terminal. Practice moving around: `pwd` prints the folder you are in ("print working directory"), `ls` lists what is in it, `cd foldername` moves into a folder, `cd ..` moves up one, and `mkdir name` makes a new folder. Navigate to the repo: `cd ~/ml/ml-learn` (the `~` means your home folder, `/home/anurag`). Checkpoint: `pwd` prints `/home/anurag/ml/ml-learn` and `ls` shows the phase folders.
- [ ] Update Ubuntu's package list and install Python. `sudo` runs a command as administrator and will ask for your Ubuntu password:

  ```bash
  sudo apt update
  sudo apt install -y python3 python3-pip python3-venv
  ```

  Checkpoint: `python3 --version` prints something like `Python 3.10` or higher, and `pip3 --version` prints a version too.
- [ ] Create a virtual environment at the repo root. A virtual environment is a private folder of Python packages just for this project, so nothing you install here can break anything else on your machine. From `~/ml/ml-learn`:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

  `source ... activate` switches your terminal into the venv. Checkpoint: your prompt now starts with `(.venv)`. You will run `source .venv/bin/activate` at the start of *every* work session for the rest of the curriculum — make it a reflex.
- [ ] Tell git to ignore the venv (it is machine-specific and huge, so it should never be uploaded). Create a file named `.gitignore` at the repo root containing at least these two lines:

  ```
  .venv/
  __pycache__/
  ```

  You can create it with `nano .gitignore` (a simple terminal text editor: type, then Ctrl+O Enter to save, Ctrl+X to exit). Checkpoint: `cat .gitignore` prints the two lines.
- [ ] With the venv active, install the core libraries. `pip` downloads packages from the internet; this can take a few minutes:

  ```bash
  pip install jupyter numpy pandas matplotlib
  ```

  Checkpoint: the install finishes with a "Successfully installed ..." line and no red error text.
- [ ] Verify everything imports. This one-liner runs a tiny Python program directly from the terminal:

  ```bash
  python -c "import numpy, pandas, matplotlib; print('workshop ready')"
  ```

  Checkpoint: it prints `workshop ready` and nothing else.
- [ ] Connect VS Code to WSL. On the Windows side, open VS Code and install the **WSL** extension (search "WSL" in the Extensions panel). Then, in your Ubuntu terminal at `~/ml/ml-learn`, run `code .` — VS Code opens with a green "WSL: Ubuntu" badge in the bottom-left corner. Inside this WSL window, install the **Python** and **Jupyter** extensions (they install "in WSL"). Checkpoint: the bottom-left badge says WSL, and opening any `.md` file in the repo shows it nicely.

<details><summary>Hints</summary>

- If `sudo apt update` complains about a lock file, another update is running in the background — wait a minute and retry.
- If your prompt does not show `(.venv)`, you probably ran `activate` from the wrong folder. `cd ~/ml/ml-learn` first, then `source .venv/bin/activate`.
- If `code .` says "command not found", close and reopen the Ubuntu terminal after installing the WSL extension in VS Code — the extension adds the command to your path on first connect.
- Hidden files and folders in Linux start with a dot, which is why `ls` does not show `.venv` or `.gitignore`. Use `ls -a` to see them.

</details>

**Definition of done:** the import one-liner prints `workshop ready`, and VS Code opens the repo with the WSL badge showing.

## Project 2 — Hello, data

**Goal:** Write your first real Python file — a script that prints a small multiplication table using a loop — and run it from the terminal. A **script** is just a text file of Python commands that run top to bottom.

**Milestones**

- [ ] Create your work folder and move into it:

  ```bash
  cd ~/ml/ml-learn
  mkdir -p work/01-environment-setup
  cd work/01-environment-setup
  ```

  (`-p` means "create parent folders as needed".) Checkpoint: `pwd` ends in `work/01-environment-setup`.
- [ ] In VS Code, create a new file `hello.py` in that folder. Start tiny — make it print one line:

  ```python
  print("hello, data")
  ```

  Run it from the terminal (venv active) with `python hello.py`. Checkpoint: you see `hello, data`. You have now run your first Python program.
- [ ] Learn the loop you need. A **for loop** repeats a block of code once per item; `range(1, 6)` gives the numbers 1 through 5 (the end number is not included). An **f-string** is a string that fills in values for you. Try this in your file and run it:

  ```python
  for i in range(1, 6):
      print(f"3 x {i} = {3 * i}")
  ```

  The four spaces of indentation are how Python knows what is inside the loop — they are required. Checkpoint: five lines, `3 x 1 = 3` through `3 x 5 = 15`.
- [ ] Now build the real thing yourself: replace that code so the script prints the full multiplication table for the numbers 1 to 5 — every combination, so 25 lines like `4 x 3 = 12`. You will need a loop inside a loop. Checkpoint: `python hello.py` prints 25 correct lines, starting `1 x 1 = 1` and ending `5 x 5 = 25`.
- [ ] Make one small change for style: print a blank line between each number's block (so the 1-times block, blank line, 2-times block, and so on). Checkpoint: the output reads as 5 tidy blocks.

<details><summary>Hints</summary>

- A loop inside a loop looks like: an outer `for i in range(...):`, and, indented inside it, another `for j in range(...):` with the `print` indented further still.
- `print()` with nothing inside prints a blank line. Where you put it (inside the outer loop, after the inner loop) controls where blanks appear.
- `IndentationError` means your spacing is inconsistent — use exactly 4 spaces per level and never mix tabs and spaces. VS Code handles this for you if you let it.

</details>

**Definition of done:** `python hello.py` prints the full 1-to-5 multiplication table, 25 lines in 5 blocks, with no errors.

## Project 3 — First notebook

**Goal:** Create a Jupyter notebook, plot y = x² with matplotlib inside VS Code, then use git to snapshot everything and push it to GitHub. A **notebook** is an interactive document mixing runnable code cells and their outputs — the standard tool for ML experiments.

**Milestones**

- [ ] In VS Code, create `first-plot.ipynb` inside `work/01-environment-setup/` (File → New File → Jupyter Notebook, or just create a file with the `.ipynb` ending). When you run the first cell, VS Code asks which "kernel" (Python) to use — pick the one at `.venv/bin/python`. Checkpoint: a cell containing `1 + 1` runs with Shift+Enter and shows `2` beneath it.
- [ ] Make a cell that plots y = x². Here is the shape of it — you fill in the y values:

  ```python
  import matplotlib.pyplot as plt

  xs = list(range(-10, 11))       # the numbers -10 to 10
  ys = ...                        # your job: y = x*x for each x (a loop that builds a list)
  plt.plot(xs, ys)
  plt.title("y = x squared")
  plt.show()
  ```

  Checkpoint: the plot appears right below the cell and shows a clean U-shape, lowest at x = 0.
- [ ] Add one markdown cell above the plot (cell type "Markdown") with a title and one sentence about what the notebook shows. Checkpoint: the rendered text appears as a heading, not as code.
- [ ] Time to save your work to git. First tell git who you are (once per machine, and use the email of your GitHub account):

  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"
  ```

  Then from the repo root, look at what changed: `git status`. Checkpoint: it lists `.gitignore` and your `work/01-environment-setup/` files as untracked, and does **not** list `.venv`.
- [ ] Stage and commit. **Staging** (`git add`) selects which changes go in the snapshot; a **commit** is the snapshot itself, with a message:

  ```bash
  git add .
  git commit -m "Lesson 01: workshop setup, hello.py and first notebook"
  ```

  Checkpoint: `git status` now says "nothing to commit, working tree clean".
- [ ] Push the commit to GitHub — **push** uploads your commits to the copy of the repo on GitHub's servers: `git push`. The first push may ask you to sign in; GitHub uses personal access tokens instead of passwords for this — the GitHub git guide in Resources walks you through it. Checkpoint: `git push` reports success.
- [ ] Open your repo on github.com in a browser and click into `work/01-environment-setup/`. Checkpoint: `hello.py` and `first-plot.ipynb` are there, and GitHub even renders the notebook with your U-shaped plot.

<details><summary>Hints</summary>

- To build `ys`: start with an empty list `ys = []`, loop over `xs`, and `ys.append(...)` each squared value. (In Python, x squared is `x * x` or `x ** 2`.)
- If VS Code cannot find a kernel, make sure the Jupyter extension is installed **in WSL** and that you installed jupyter inside the venv in Project 1 — then click "Select Kernel" at the top right of the notebook.
- If `git push` is rejected with an authentication error, follow "About remote repositories" and the authentication pages in the GitHub docs linked below — you need to create a token once and use it as your password.

</details>

**Definition of done:** the notebook renders a U-shaped parabola inside VS Code, and all of today's files are visible on github.com.

## Stretch goals

- Make `hello.py` take the table size from the user with `input()`, so `python hello.py` asks "Up to what number?" and prints a table that big.
- In the notebook, plot y = x² and y = x³ on the same chart and add a legend (search matplotlib's docs for `plt.legend`).
- Learn 3 more terminal commands and try each: `cp` (copy), `mv` (move/rename), `rm` (delete — careful, there is no recycle bin).
- Run `git log` and read your commit history; then make a tiny change, commit it with a good message, and push again. Two commits on day one is a great habit.

## If you get stuck

- **Read error messages from the bottom up.** The last line usually names the actual problem (`ModuleNotFoundError: No module named 'numpy'` means the package is missing *in the Python you are running*).
- **"Module not found" almost always means the venv is not active.** Check for `(.venv)` in your prompt; if it is missing, `source .venv/bin/activate` from the repo root.
- **"Command not found" means the terminal cannot find the program** — either it is not installed, or you need a new terminal window after installing it.
- **Print things.** When code misbehaves, add `print(...)` lines to see what values your variables really hold.
- **Ask an AI assistant for a HINT, not a solution.** Paste the error and ask "what does this mean?" — never "write this for me". The struggle is where the learning is.
- **Type all code yourself.** No copy-pasting solutions, ever. Copying commands like `pip install ...` is fine; copying project code defeats the whole curriculum.

## Resources

- [The official Python tutorial](https://docs.python.org/3/tutorial/) — skim sections 1-3 for a feel of the language; lessons 02-04 cover it project-by-project.
- [VS Code WSL documentation](https://code.visualstudio.com/docs/remote/wsl) — the definitive guide if VS Code will not connect to Ubuntu.
- [GitHub's Using Git guide](https://docs.github.com/en/get-started/using-git) — covers add/commit/push and the authentication setup for your first push.

## Skills unlocked

- [ ] I can move around the Linux filesystem with `cd`, `ls`, `pwd` and `mkdir`.
- [ ] I can explain in one sentence what a virtual environment is and why I use one.
- [ ] I can activate the venv and install packages into it with pip.
- [ ] I can write a Python file with a loop in it and run it from the terminal.
- [ ] I can create and run a Jupyter notebook inside VS Code, on the right kernel.
- [ ] I can make a basic matplotlib plot.
- [ ] I can stage, commit and push my work to GitHub, and find it there in the browser.

## Next up

Your workshop is built — now learn the language of the trade by programming small games: [02 · Python Basics by Building Games](02-python-basics.md).
