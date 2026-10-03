# 01 · Set Up Your ML Workshop

**Phase 0 — Foundations** · Estimated time: 2-4 days · Prerequisites: none — this is day one

> Every machine learning expert works out of the same basic workshop: a terminal, Python, a code editor, and git. Today you build that workshop on your own machine — a Mac, a Windows PC or a Linux PC all work — and by the end of it you will have written and run your first Python program, made your first chart, and pushed your first code to GitHub. Nothing here is throwaway — this exact setup is what you will use for all 42 lessons, from printing a multiplication table today to training your own GPT later. Go slowly, type everything yourself, and celebrate each checkpoint.

## What you will build

- **Project 1 — Workshop setup:** git and Python 3.13 installed on your computer, your own copy of this repository at `~/ml/ml-learn`, a virtual environment at the repo root with the core ML libraries installed and verified, and VS Code ready to edit it all.
- **Project 2 — Hello, data:** `hello.py`, a small Python script that prints a multiplication table, which you run from the terminal.
- **Project 3 — First notebook:** a Jupyter notebook that plots the curve y = x², run inside VS Code — then committed and pushed so it is visible on GitHub.

## Concepts you will learn by doing

- The **terminal**: a text window where you type commands to control your computer (`cd`, `ls`, `mkdir`).
- **One workshop on any computer**: macOS and Linux have a Unix-style terminal built in; on Windows you use **WSL2**, a real Ubuntu Linux system running inside Windows. Once setup is done, almost every command in this curriculum is the same on all three.
- **Package managers**: tools that install software for you from the terminal — **Homebrew** (`brew`) on macOS and **apt** on Ubuntu, including Ubuntu inside WSL. (Intel Macs, and Macs still on macOS 14 or older, use Python's official installer instead.)
- **Python** and **pip**: the programming language of ML, and its tool for installing extra packages.
- **Virtual environments (venv)**: a private folder of Python packages for one project, so different projects never fight over versions.
- **VS Code with the Python and Jupyter extensions**: your editor (on Windows, plus the WSL extension that connects it into Linux).
- **git and GitHub basics**: forking and cloning a repository, SSH keys, and `add`, `commit`, `push` — saving snapshots of your work and backing them up to GitHub.
- **This repo's layout**: numbered phase folders hold the lessons (read them, don't edit them); a `work/` folder holds all the code you write.

## Before you start

You need:

- A computer running **macOS**, **Windows 11** (or Windows 10 version 22H2), or **Linux**, with an internet connection and an administrator account, so you can install software. The Linux commands in this curriculum are written for Ubuntu 22.04 or newer; other distributions work with their own package manager.
- A free **GitHub account** with a verified email address — sign up at [github.com](https://github.com) if you do not have one yet.

Wherever the steps differ between systems, you will see labeled instructions for **macOS**, **Windows (WSL2)** and **Linux** — follow only the one for your computer. Everything without a label is identical everywhere.

Two rules for the whole curriculum:

1. **Every command runs in your terminal**: the Terminal app on macOS, the Ubuntu (WSL) terminal on Windows, your usual terminal on Linux — or the terminal built into VS Code (Terminal → New Terminal) once it is set up. On Windows, never type curriculum commands into PowerShell or CMD; the one exception is installing WSL itself in Project 1.
2. **All your code lives in `work/`**, one subfolder per lesson. Your folder for this lesson will be `work/01-environment-setup/` — you will create it in Project 2. The lesson files like this one are your instructions; the `work/` folder is your workbench.

No pip installs are needed before Project 1 — installing things *is* Project 1.

## Project 1 — Workshop setup

**Goal:** Open a terminal, install git and Python 3.13, put your own copy of this repository at `~/ml/ml-learn`, create a virtual environment at the repo root, install the four core libraries, prove they all import, and connect VS Code. This is the foundation every later lesson stands on.

**Milestones**

- [ ] Open a terminal.

  - **macOS:** press Cmd+Space, type `Terminal` and press Return (the app lives in Applications → Utilities). Your prompt ends in `%` — that is zsh, the Mac's default shell. Run `echo 'setopt interactivecomments' >> ~/.zshrc` once: some commands in later lessons end with a note after a `#`, and this makes zsh skip such notes the way Linux terminals do (it takes effect in new Terminal windows).
  - **Windows (WSL2):** install WSL once (already had WSL before today? skip to the note after step 4):
    1. Right-click the Start button and choose **Terminal (Admin)** (or search for PowerShell and choose **Run as administrator**), then run `wsl --install`.
    2. If it ends by saying the changes will not be effective until the system is rebooted, restart your PC, open an administrator window the same way and run `wsl --install` again. This time it downloads Ubuntu and starts it right in that window. (If Ubuntu starts straight away the first time, skip the restart. A "Welcome to Windows Subsystem for Linux" window may also open; you can close it.)
    3. Ubuntu asks for a Linux username. It suggests one based on your Windows name: press Enter to keep it, or erase it with Backspace and type a short, lowercase one. Then choose a password — nothing appears on screen while you type it, and that is normal. Remember it, because `sudo` will ask for it. If Ubuntu then asks whether to share anonymous "platform metrics" with Canonical, either answer is fine.
    4. Close the administrator window. From now on, "the terminal" means **Ubuntu**, opened from the Start menu.

    Already had WSL before today? Open Ubuntu and run `lsb_release -d`: Ubuntu 22.04, 24.04 or 26.04 is fine. Ubuntu 20.04 or older cannot get Python 3.13, so in an administrator window run `wsl --update`, then `wsl --install -d Ubuntu-26.04`, and from now on open **Ubuntu-26.04** from the Start menu.
  - **Linux:** open your terminal app (Ctrl+Alt+T on Ubuntu). Linux terminals copy with Ctrl+Shift+C and paste with Ctrl+Shift+V; plain Ctrl+V does not paste there, and Ctrl+C stops the running command instead of copying.

  Practice moving around: `pwd` prints the folder you are in ("print working directory"), `ls` lists what is in it, `cd foldername` moves into a folder, `cd ..` moves up one, and `mkdir name` makes a new folder. `cd ~` takes you home — `~` is shorthand for your home folder, `/Users/yourname` on a Mac and `/home/yourname` on Linux and WSL. Checkpoint: after `cd ~`, `pwd` prints your home folder.

- [ ] Install git and Python 3.13. Why 3.13 and not the newest Python? Brand-new Python versions take months to be supported by every ML library, and 3.13 works with every library this curriculum uses. Each system installs software its own way. Mac users, check which Mac you have under Apple menu → **About This Mac**: "Chip" means Apple Silicon, "Processor" means Intel, and the same window shows your macOS version number — 15 or any higher number (such as 26) counts as macOS 15 or later.

  - **macOS on Apple Silicon (M1 or newer) with macOS 15 or later:** install **Homebrew**, the Mac's package manager, with the command from [brew.sh](https://brew.sh):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    Press Return when it asks, and type your Mac login password (nothing appears while you type — that is normal). It also installs Apple's Command Line Tools, which include git; this can take a while. When it finishes, it prints **Next steps** with commands that put `brew` on your PATH. Run them — they are not optional. They look like this:

    ```bash
    echo >> ~/.zprofile
    echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv zsh)"
    ```

    Then close Terminal, open it again, and install Python:

    ```bash
    brew install python@3.13
    ```

  - **macOS on an Intel Mac, or on macOS 14 and older:** Homebrew no longer supports these Macs, so use Python's official installer. On [python.org/downloads/macos](https://www.python.org/downloads/macos/), pick the newest **Python 3.13** release (not 3.14 or later) and download its macOS installer. Run it, then open the new **Python 3.13** folder in Applications and double-click **Install Certificates.command** — without that, Python cannot download data over HTTPS. For git, run `xcode-select --install` in Terminal and click **Install**. (Intel Macs handle the early lessons fine, but from lesson 16 on some libraries no longer work on them: XGBoost needs Homebrew's OpenMP library, and current PyTorch has no Intel Mac version. Plan on a free cloud notebook such as Google Colab for those lessons. An Apple Silicon Mac on an old macOS can instead update macOS for free and use Homebrew.)
  - **Windows (WSL2) and Ubuntu Linux:** these steps need Ubuntu 22.04, 24.04 or 26.04 (on an older Ubuntu PC, upgrade Ubuntu first). `sudo` runs a command as administrator and asks for your Linux password (typing may not show on screen — that is normal). Ubuntu comes with its own Python, but depending on your Ubuntu version it is too old or too new for this curriculum, so add the widely used **deadsnakes** package source and install 3.13 alongside it — Ubuntu's own `python3` stays untouched. `curl` downloads files and `unzip` unpacks .zip archives; later lessons use both:

    ```bash
    sudo apt update
    sudo apt install -y software-properties-common git curl unzip
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt install -y python3.13 python3.13-venv
    ```

  - **Other Linux distributions:** install Python 3.13 (with its venv module) and git with your package manager — on Fedora, `sudo dnf install -y python3.13 git`; on Debian 13, `sudo apt install -y python3.13-venv git curl unzip`.

  Checkpoint: `python3.13 --version` prints `Python 3.13.` followed by a number, and `git --version` prints a version.

- [ ] Get your own copy of this repository. In Project 3 you will push your work to GitHub, and you can only push to a repository you own — so first **fork** this one, which copies it into your GitHub account: open [github.com/anurag-2911/ml-learn](https://github.com/anurag-2911/ml-learn), click **Fork**, then **Create fork**. Then **clone** your fork — download it onto your computer — into `~/ml/ml-learn`, the path every lesson uses. Replace `YOUR-USERNAME` with your GitHub username:

  ```bash
  git clone https://github.com/YOUR-USERNAME/ml-learn.git ~/ml/ml-learn
  cd ~/ml/ml-learn
  ```

  Checkpoint: `pwd` prints a path ending in `/ml/ml-learn`, `ls` shows `README.md`, `PROGRESS.md` and the phase folders, and `git remote -v` shows your username, not `anurag-2911`.

- [ ] Create a virtual environment at the repo root. A virtual environment is a private folder of Python packages just for this project, so nothing you install here can break anything else on your computer. From `~/ml/ml-learn`:

  ```bash
  python3.13 -m venv .venv
  source .venv/bin/activate
  ```

  `source ... activate` switches your terminal into the venv. Inside it, `python`, `python3` and `pip` all mean the venv's own Python 3.13 and its pip — later lessons use both `python` and `python3`, and inside the venv they are the same. Checkpoint: your prompt now starts with `(.venv)`, `python --version` prints Python 3.13, and `which python` prints a path ending in `ml-learn/.venv/bin/python`. You will run `source .venv/bin/activate` at the start of *every* work session for the rest of the curriculum — make it a reflex.

- [ ] Tell git to ignore files that should never be uploaded. Create a file named `.gitignore` at the repo root containing at least these five lines (if the file already exists, just make sure they are in it):

  ```
  .venv/
  __pycache__/
  .ipynb_checkpoints/
  .DS_Store
  *:Zone.Identifier
  ```

  The venv is machine-specific and huge; `__pycache__/` and `.ipynb_checkpoints/` are scratch folders that Python and Jupyter create; `.DS_Store` files are clutter that macOS leaves in every folder you open in Finder, and `Zone.Identifier` files are clutter that Windows adds when you drag downloaded files into Ubuntu. You can create the file with `nano .gitignore` (a simple terminal text editor: type the lines, then Ctrl+O and Enter to save, Ctrl+X to exit — that is the Control key, not Command, even on a Mac). Checkpoint: `cat .gitignore` prints the five lines.

- [ ] With the venv active, install the core libraries. `pip` downloads packages from the internet; the first command updates pip itself, and the second can take a few minutes:

  ```bash
  pip install --upgrade pip
  pip install jupyter numpy pandas matplotlib
  ```

  Checkpoint: the install finishes with a "Successfully installed ..." line (or only "Requirement already satisfied" lines, if you ran it before) and no line starting with `ERROR`.

- [ ] Verify everything imports. This one-liner runs a tiny Python program directly from the terminal:

  ```bash
  python -c "import numpy, pandas, matplotlib; print('workshop ready')"
  ```

  Checkpoint: it prints `workshop ready` and nothing else.

- [ ] Set up VS Code, your code editor:

  - **macOS:** download VS Code from [code.visualstudio.com](https://code.visualstudio.com), open the downloaded file and drag **Visual Studio Code** into Applications, then open it from there. Press Cmd+Shift+P, run **Shell Command: Install 'code' command in PATH** (it asks for your password), and open a new Terminal window.
  - **Windows (WSL2):** install VS Code on the *Windows* side from [code.visualstudio.com](https://code.visualstudio.com) — never inside Ubuntu — and leave the installer's **Add to PATH** option ticked. Open it, install the **WSL** extension (search "WSL" in the Extensions panel), then close and reopen your Ubuntu terminal.
  - **Linux:** on Ubuntu, run `sudo snap install --classic code`. On other distributions, or on an ARM computer (where that snap does not exist), download the `.deb` (Debian, Ubuntu) or `.rpm` (Fedora) package from [code.visualstudio.com](https://code.visualstudio.com) — on an ARM computer, click the small **Arm64** link instead of the big button — and install it with `sudo apt install -y ~/Downloads/code_*.deb` (if it asks whether to add Microsoft's repository, answer Yes, so VS Code keeps itself updated; a last line starting `N: Download is performed unsandboxed` is harmless) or `sudo dnf install -y ~/Downloads/code-*.rpm`.

  Then run `cd ~/ml/ml-learn` and `source .venv/bin/activate` (a new terminal window always starts in your home folder with the venv off), and then `code .` (the dot means "this folder"). When VS Code asks whether you trust the authors of the files in this folder, choose **Yes, I trust the authors** — it is your own copy. On Windows, the bottom-left corner of the window now shows **WSL: Ubuntu** (your distribution's name), meaning VS Code is working inside Linux. Open the Extensions panel (Cmd+Shift+X on a Mac, Ctrl+Shift+X elsewhere) and install **Python** and **Jupyter**, both published by Microsoft — the Python extension does not include Jupyter. On Windows, if a button says **Install in WSL: Ubuntu**, click it. Checkpoint: open this lesson file in VS Code and press Cmd+Shift+V on a Mac (Ctrl+Shift+V elsewhere) — you see it nicely formatted, like on GitHub.

<details><summary>Hints</summary>

- **Mac:** if running `git` or `python3` pops up an offer to install "command line developer tools", click **Install**, wait, and try again.
- **Mac:** `zsh: command not found: brew` means the Homebrew **Next steps** were skipped. Run those three lines, then open a new Terminal window.
- **Windows:** if `wsl --install` fails with an error about virtualization, it must be switched on in your PC's firmware (BIOS/UEFI) settings — search the web for your PC model plus "enable virtualization". If it says "Cannot create a file when that file already exists", Ubuntu is already installed: open it from the Start menu and check its version as described under "Already had WSL before today?" in the first milestone.
- **Windows:** keep all your curriculum files inside Ubuntu (under `~`), never under `/mnt/c/...` — Linux tools are much slower on Windows files. To see your Ubuntu files in Windows Explorer, run `explorer.exe .` (with the dot).
- **Windows and Ubuntu:** if `sudo apt update` complains about a lock file, another update is running in the background — wait a minute and retry. `E: Unable to locate package python3.13` means your Ubuntu is older than 22.04 (`lsb_release -d` shows it): on Windows, add Ubuntu-26.04 as described in the first milestone; on a Linux PC, upgrade Ubuntu first.
- **Debian:** if `sudo` says you are not in the sudoers file (or `sudo: command not found`), you set a root password while installing Debian, so your account cannot use sudo yet. Run `su - -c "apt install -y sudo && usermod -aG sudo $USER"`, type the root password, then restart your computer and run the install command again.
- `git clone` asks for a username or password, or fails with "Authentication failed" or "Repository not found": the address is wrong, because cloning your own public fork never needs a password. Press Ctrl+C, check that your fork exists at `https://github.com/YOUR-USERNAME/ml-learn` and that you typed your own username in place of `YOUR-USERNAME`, then clone again. "destination path already exists": `~/ml/ml-learn` is already there — `cd` into it instead of cloning again.
- `git remote -v` shows `anurag-2911`: you cloned the original instead of your fork. Fork it as described above, then run `git remote set-url origin https://github.com/YOUR-USERNAME/ml-learn.git`.
- If your prompt does not show `(.venv)`, the venv is not active in *this* terminal window — activation lasts only for the window you ran it in. Run `cd ~/ml/ml-learn` and then `source .venv/bin/activate`.
- `error: externally-managed-environment`, `command not found: python` or `pip`, or (on Ubuntu and WSL) "Command 'python' not found" or "Command 'pip' not found": same cause — the venv is not active. Activate it instead of installing what Ubuntu suggests (`python-is-python3` or `python3-pip`): those make `python` and `pip` run Ubuntu's own Python whenever the venv is off, which hides this mistake. Never work around it with `sudo pip`.
- Created the venv with the wrong Python? `python --version` inside the venv must say 3.13. If not, run `deactivate`, delete the venv with `rm -rf .venv`, and create it again with `python3.13 -m venv .venv`.
- **Intel Mac:** if `pip install` fails while building a package, add `--prefer-binary` so pip picks a ready-made version: `pip install --prefer-binary jupyter numpy pandas matplotlib`.
- If `code .` says "command not found": on a Mac, repeat the **Shell Command** step, then open a new Terminal window; on Windows, close every Ubuntu window and open a new one (if it still fails, reinstall VS Code with **Add to PATH** ticked — and ignore Ubuntu's suggestion to `snap install code`, which would put a second VS Code inside Linux); on Linux, open a new terminal.
- Hidden files and folders on macOS and Linux start with a dot, which is why `ls` does not show `.venv` or `.gitignore`. Use `ls -a` to see them. (In a Mac's Finder, Cmd+Shift+. shows and hides them.)

</details>

**Definition of done:** the import one-liner prints `workshop ready`, and `code .` opens the repo in VS Code with the Python and Jupyter extensions installed (and, on Windows, **WSL** showing in the bottom-left corner).

## Project 2 — Hello, data

**Goal:** Write your first real Python file — a script that prints a small multiplication table using a loop — and run it from the terminal. A **script** is just a text file of Python commands that run top to bottom.

**Milestones**

- [ ] Create your work folder and move into it, with the venv active:

  ```bash
  cd ~/ml/ml-learn
  source .venv/bin/activate
  mkdir -p work/01-environment-setup
  cd work/01-environment-setup
  ```

  (`-p` means "create parent folders as needed".) Checkpoint: your prompt starts with `(.venv)` and `pwd` ends in `work/01-environment-setup`.
- [ ] In VS Code, create a new file `hello.py` in that folder (in the Explorer panel, right-click `work/01-environment-setup` and choose **New File**). Start tiny — make it print one line:

  ```python
  print("hello, data")
  ```

  Save it with Cmd+S on a Mac or Ctrl+S elsewhere — VS Code does not save automatically, and a dot on the file's tab means unsaved changes. Run it from the terminal (venv active) with `python hello.py`. Checkpoint: you see `hello, data`. You have now run your first Python program.
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
- Output did not change after an edit? You probably forgot to save — check the file's tab for a dot.

</details>

**Definition of done:** `python hello.py` prints the full 1-to-5 multiplication table, 25 lines in 5 blocks, with no errors.

## Project 3 — First notebook

**Goal:** Create a Jupyter notebook, plot y = x² with matplotlib inside VS Code, then use git to snapshot everything and push it to GitHub. A **notebook** is an interactive document mixing runnable code cells and their outputs — the standard tool for ML experiments.

**Milestones**

- [ ] In VS Code, create `first-plot.ipynb` inside `work/01-environment-setup/` (right-click the folder in the Explorer, choose **New File**, and type the name — the `.ipynb` ending makes it a notebook). Keep the repo root, `~/ml/ml-learn`, as the folder open in VS Code, so it can find your venv. When you run the first cell, VS Code asks which "kernel" (Python) to use — choose **Python Environments**, then `.venv` (its path ends in `.venv/bin/python`). Checkpoint: a cell containing `1 + 1` runs with Shift+Enter and shows `2` beneath it.
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
- [ ] Add one markdown cell above the plot (hover over the top edge of the plot cell and click **+ Markdown**) with a title line starting with `# ` and one sentence about what the notebook shows, then press Shift+Enter. Save the notebook (Cmd+S or Ctrl+S). Checkpoint: the rendered text appears as a heading, not as code.
- [ ] Time to save your work to git. First tell git who you are — once per computer (on Windows, inside Ubuntu). Your fork is public, and so is the email address in every commit, so keep your real address out of it: on github.com open **Settings → Emails**, make sure **Keep my email addresses private** is ticked (this also covers commits GitHub makes for you, such as when you click **Sync fork**), and copy the "noreply" address shown there (it looks like `12345678+YOUR-USERNAME@users.noreply.github.com`). Then give git your name and that address:

  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "12345678+YOUR-USERNAME@users.noreply.github.com"
  ```

  Then go back to the repo root and look at what changed:

  ```bash
  cd ~/ml/ml-learn
  git status
  ```

  Checkpoint: `git config user.email` prints your own noreply address (your number and username, not `12345678+YOUR-USERNAME`), and `git status` lists `.gitignore` and `work/` under "Untracked files" but not `.venv/` or `.DS_Store`. (`git status -u` shows every file inside `work/` too.)
- [ ] Stage and commit. **Staging** (`git add`) selects which changes go in the snapshot; a **commit** is the snapshot itself, with a message:

  ```bash
  git add .
  git commit -m "Lesson 01: workshop setup, hello.py and first notebook"
  ```

  Checkpoint: `git status` now says "nothing to commit, working tree clean".
- [ ] Give GitHub a key to your computer — once per computer (on Windows, inside Ubuntu). GitHub does not accept your password for pushing; instead you make an **SSH key**, a pair of files: a private half that never leaves your computer and a public half that you give to GitHub. Create one, pressing Enter at each question to accept the defaults (a passphrase is optional — if you set one, git will ask for it when you push; if it asks to overwrite an existing key, answer `n` and use the key you already have):

  ```bash
  ssh-keygen -t ed25519 -C "you@example.com"
  cat ~/.ssh/id_ed25519.pub
  ```

  The second command prints the public half: one line starting with `ssh-ed25519`. Select it with the mouse and copy it (Cmd+C on a Mac, Ctrl+Shift+C in Windows and Linux terminals). On github.com, click your profile picture → **Settings** → **SSH and GPG keys** → **New SSH key**, paste the line into **Key**, give it a title such as your computer's name, and click **Add SSH key**. Then test the connection:

  ```bash
  ssh -T git@github.com
  ```

  The first time, ssh shows a key fingerprint and asks `Are you sure you want to continue connecting (yes/no/[fingerprint])?`. Check that the fingerprint matches the ED25519 one listed on GitHub's [SSH key fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) page, and only then type `yes`; if it does not match, type `no` and stop. Checkpoint: the reply starts with `Hi` and your username, followed by `You've successfully authenticated`.
- [ ] Switch your clone from GitHub's HTTPS address to its SSH address, so that pushes use your key:

  ```bash
  git remote set-url origin git@github.com:YOUR-USERNAME/ml-learn.git
  ```

  Checkpoint: `git remote -v` shows `git@github.com:` followed by your own username and `/ml-learn.git` on both lines — not `YOUR-USERNAME` or `anurag-2911`.
- [ ] Push the commit to GitHub — **push** uploads your commits to your copy of the repo on GitHub's servers: `git push`. Checkpoint: the output ends with a line like `main -> main` and no error.
- [ ] Open your fork at `https://github.com/YOUR-USERNAME/ml-learn` in a browser (your copy, not the original) and click into `work/01-environment-setup/`. Checkpoint: `hello.py` and `first-plot.ipynb` are there, and GitHub even renders the notebook with your U-shaped plot.

<details><summary>Hints</summary>

- To build `ys`: start with an empty list `ys = []`, loop over `xs`, and `ys.append(...)` each squared value. (In Python, x squared is `x * x` or `x ** 2`.)
- If VS Code cannot find a kernel: make sure VS Code has the repo root open (run `code .` from `~/ml/ml-learn`), that the Python and Jupyter extensions are installed (on Windows, in WSL), and that you installed jupyter inside the venv in Project 1 — then click **Select Kernel** at the top right of the notebook.
- `ssh -T` says `Permission denied (publickey)`: GitHub does not have this computer's public key yet — repeat the **New SSH key** step, pasting the line that `cat ~/.ssh/id_ed25519.pub` prints. If `ssh -T` just hangs, your network may block SSH; GitHub's guide to using SSH over the HTTPS port (linked below) fixes that.
- `git push` asks for a username for `https://github.com`: your clone still uses the HTTPS address — run the `git remote set-url` command above, then push again. `ERROR: Repository not found`: the address you gave `git remote set-url` is wrong, usually because `YOUR-USERNAME` was not replaced — run it again with your exact username.
- `git push` says "Permission to anurag-2911/ml-learn.git denied" or "ERROR: Permission ... denied": your clone points at the original repository instead of your fork. Run the `git remote set-url` command above with your own username, then push again.
- `git push` fails with "GH007: Your push would publish a private email address": set the noreply address with `git config --global user.email ...` as above, run `git commit --amend --reset-author --no-edit`, and push again.
- `git push` is rejected with `(fetch first)`, or `git pull` stops with `fatal: Need to specify how to reconcile divergent branches`: your fork on GitHub and your computer each have commits the other does not (usually after **Sync fork**, below). Run `git pull --no-rebase --no-edit`, then `git push`.
- Later, when the original repository gets new or updated lessons: first commit and push your own work, then on your fork's GitHub page click **Sync fork**, then **Update branch**, and run `git pull`. If GitHub says the branch has conflicts, do not click **Discard commits** — that deletes your own work from your fork; ask for help instead.

</details>

**Definition of done:** the notebook renders a U-shaped parabola inside VS Code, and all of today's files are visible in your fork on github.com.

## Stretch goals

- Make `hello.py` take the table size from the user with `input()`, so `python hello.py` asks "Up to what number?" and prints a table that big.
- In the notebook, plot y = x² and y = x³ on the same chart and add a legend (search matplotlib's docs for `plt.legend`).
- Learn 3 more terminal commands and try each: `cp` (copy), `mv` (move/rename), `rm` (delete — careful, there is no recycle bin).
- Run `git log` and read your commit history (press `q` to get back to the prompt); then make a tiny change, commit it with a good message, and push again. Two commits on day one is a great habit.

## If you get stuck

- **Read error messages from the bottom up.** The last line usually names the actual problem (`ModuleNotFoundError: No module named 'numpy'` means the package is missing *in the Python you are running*).
- **"Module not found" almost always means the venv is not active.** Check for `(.venv)` in your prompt; if it is missing, `source .venv/bin/activate` from the repo root.
- **"Command not found" means the terminal cannot find the program** — either it is not installed, or you need a new terminal window after installing it.
- **A later lesson's command fails only on your system?** A few later lessons still assume Windows with WSL: some download files with `wget`, which Macs do not have (use `curl -L -o FILE URL` instead), and some copy from a Windows folder under `/mnt/c/...` (use a folder on your own computer). Ask an AI assistant for your system's equivalent.
- **Print things.** When code misbehaves, add `print(...)` lines to see what values your variables really hold.
- **Ask an AI assistant for a HINT, not a solution.** Paste the error and ask "what does this mean?" — never "write this for me". The struggle is where the learning is.
- **Type all code yourself.** No copy-pasting solutions, ever. Copying commands like `pip install ...` is fine; copying project code defeats the whole curriculum.

## Resources

- [The official Python tutorial](https://docs.python.org/3/tutorial/) — skim sections 1-3 for a feel of the language; lessons 02-04 cover it project-by-project.
- [Homebrew](https://brew.sh) and [Python releases for macOS](https://www.python.org/downloads/macos/) — the two ways to install Python on a Mac.
- [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) — Microsoft's official guide, if `wsl --install` gives you trouble.
- VS Code setup guides for [macOS](https://code.visualstudio.com/docs/setup/mac), [Linux](https://code.visualstudio.com/docs/setup/linux) and [Windows with WSL](https://code.visualstudio.com/docs/remote/wsl), plus [Jupyter notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
- GitHub's guides to [forking a repository](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo), [syncing a fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/syncing-a-fork), [connecting with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and [using SSH over the HTTPS port](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port).

## Skills unlocked

- [ ] I can move around my computer's folders from the terminal with `cd`, `ls`, `pwd` and `mkdir`.
- [ ] I can install developer tools on my computer — with a package manager, or an official installer on an older Mac.
- [ ] I can explain in one sentence what a virtual environment is and why I use one.
- [ ] I can activate the venv and install packages into it with pip.
- [ ] I can write a Python file with a loop in it and run it from the terminal.
- [ ] I can create and run a Jupyter notebook inside VS Code, on the right kernel.
- [ ] I can make a basic matplotlib plot.
- [ ] I can fork and clone a repository, connect to GitHub with an SSH key, and stage, commit and push my work — and find it on github.com.

## Next up

Your workshop is built — now learn the language of the trade by programming small games: [02 · Python Basics by Building Games](02-python-basics.md).
