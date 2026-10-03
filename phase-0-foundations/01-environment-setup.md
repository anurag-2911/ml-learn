# 01 · Set Up Your ML Workshop

**Phase 0 — Foundations** · Estimated time: 2-4 days · Prerequisites: none

> Machine learning work needs four basic tools: a terminal, Python, a code editor and git. This lesson sets them up on a Mac, a Windows PC or a Linux PC. It then uses them to write and run a first Python program, draw a first chart, and push the first code to GitHub. The same setup is used in all 42 lessons, from the multiplication table in Project 2 to the GPT model in lesson 33. Go slowly, and type the Python code by hand.

## What this lesson builds

- **Project 1 — Workshop setup:** git and Python 3.13, a fork of this repository cloned to `~/ml/ml-learn`, a virtual environment at the repo root with the core ML libraries installed and tested, and VS Code ready to edit it all.
- **Project 2 — Hello, data:** `hello.py`, a small Python script that prints a multiplication table when run from the terminal.
- **Project 3 — First notebook:** a Jupyter notebook in VS Code that plots the curve y = x², committed and pushed to GitHub.

## Concepts covered

- The **terminal**: a text window where typed commands control the computer (`cd`, `ls`, `mkdir`).
- **One workshop on any computer**: macOS and Linux have a Unix-style terminal built in. Windows gets one through **WSL2**, a real Ubuntu Linux system that runs inside Windows. Once setup is done, almost every command in this curriculum is the same on all three.
- **Package managers**: terminal tools that install software, such as **Homebrew** (`brew`) on macOS and **apt** on Ubuntu, including Ubuntu inside WSL. (Intel Macs, and Macs still on macOS 14 or older, use Python's official installer instead.)
- **Python** and **pip**: the programming language of ML, and its tool for installing extra packages.
- **Virtual environments (venv)**: a private folder of Python packages for one project, so that different projects never clash over package versions.
- **VS Code with the Python and Jupyter extensions**: the code editor (on Windows, together with the WSL extension that connects it to Linux).
- **git and GitHub basics**: forking and cloning a repository, SSH keys, and `add`, `commit` and `push`, which save snapshots of the work and back them up to GitHub.
- **The repository layout**: numbered phase folders hold the lessons (read them, but do not edit them), and a `work/` folder holds all the code written during the course.

## Before starting

Requirements:

- A computer running **macOS**, **Windows 11** (or Windows 10 version 22H2) or **Linux**, with an internet connection and an administrator account for installing software. The Linux commands in this curriculum are written for Ubuntu 22.04 or newer; other distributions work with their own package manager.
- A free **GitHub account** with a verified email address (sign up at [github.com](https://github.com)).

Where the steps differ between systems, they are labeled **macOS**, **Windows (WSL2)** and **Linux**. Follow only the label that matches the computer. Steps without a label are the same everywhere.

Two rules apply to the whole curriculum:

1. **Every command runs in the terminal**: the Terminal app on macOS, the Ubuntu (WSL) terminal on Windows, the usual terminal app on Linux, or the terminal built into VS Code (Terminal → New Terminal) once it is set up. On Windows, curriculum commands never go into PowerShell or CMD. The one exception is installing WSL itself in Project 1.
2. **All code lives in `work/`**, in one subfolder per lesson. This lesson's folder is `work/01-environment-setup/`, created in Project 2. Lesson files like this one are the instructions; the `work/` folder is the workbench.

Nothing needs to be installed in advance. Installing things is Project 1.

## Project 1 — Workshop setup

**Goal:** Open a terminal, install git and Python 3.13, fork this repository and clone the fork to `~/ml/ml-learn`, create a virtual environment at the repo root, install the four core libraries, check that they import, and connect VS Code. Every later lesson builds on this setup.

**Milestones**

- [ ] Open a terminal.

  - **macOS:** press Cmd+Space, type `Terminal` and press Return (the app is in Applications → Utilities). The prompt ends in `%`: that is zsh, the default shell on a Mac. Run `echo 'setopt interactivecomments' >> ~/.zshrc` once. Some commands in later lessons end with a note after a `#`, and this setting makes zsh skip such notes the way Linux terminals do (it takes effect in new Terminal windows).
  - **Windows (WSL2):** install WSL once (if WSL is already installed, skip to the note after step 4):
    1. Right-click the Start button and choose **Terminal (Admin)** (or search for PowerShell and choose **Run as administrator**), then run `wsl --install`.
    2. If it ends by saying the changes will not be effective until the system is rebooted, restart the PC, open an administrator window the same way and run `wsl --install` again. This time it downloads Ubuntu and starts it in that window. (If Ubuntu starts straight away the first time, skip the restart. A "Welcome to Windows Subsystem for Linux" window may also open; it can be closed.)
    3. Ubuntu asks for a Linux username and suggests one based on the Windows account name. Press Enter to keep it, or erase it with Backspace and type a short, lowercase name. Then choose a password. Nothing appears on screen while the password is typed; that is normal. Remember it, because `sudo` asks for it. If Ubuntu then asks whether to share anonymous "platform metrics" with Canonical, either answer is fine.
    4. Close the administrator window. From now on, "the terminal" means **Ubuntu**, opened from the Start menu.

    If WSL was installed before, open Ubuntu and run `lsb_release -d`. Ubuntu 22.04, 24.04 or 26.04 is fine. Ubuntu 20.04 or older cannot get Python 3.13, so in an administrator window run `wsl --update`, then `wsl --install -d Ubuntu-26.04`, and from then on open **Ubuntu-26.04** from the Start menu.
  - **Linux:** open the terminal app (Ctrl+Alt+T on Ubuntu). Linux terminals copy with Ctrl+Shift+C and paste with Ctrl+Shift+V. Plain Ctrl+V does not paste there, and Ctrl+C stops the running command instead of copying.

  Practice moving around. `pwd` prints the current folder ("print working directory"), `ls` lists what is in it, `cd foldername` moves into a folder, `cd ..` moves up one level, and `mkdir name` makes a new folder. `cd ~` goes back to the home folder. `~` is short for the home folder's path, which looks like `/Users/ada` on a Mac and `/home/ada` on Linux and WSL. Checkpoint: after `cd ~`, `pwd` prints the home folder.

- [ ] Install git and Python 3.13. Why 3.13 and not the newest Python? ML libraries take months to support each new Python version, and 3.13 works with every library this curriculum uses. Each system installs software its own way. On a Mac, first open Apple menu → **About This Mac**: "Chip" means Apple Silicon and "Processor" means Intel. The same window shows the macOS version: 15 or any higher number (such as 26) counts as macOS 15 or later.

  - **macOS on Apple Silicon (M1 or newer) with macOS 15 or later:** install **Homebrew**, the package manager for macOS, with the command from [brew.sh](https://brew.sh):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    Press Return when asked, then type the Mac login password. Nothing appears on screen while the password is typed; that is normal. The script also installs Apple's Command Line Tools, which include git, so it can take a while. At the end it prints **Next steps**: commands that put `brew` on the PATH. Run them; they are not optional. They look like this:

    ```bash
    echo >> ~/.zprofile
    echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv zsh)"
    ```

    Then close Terminal, open it again, and install Python:

    ```bash
    brew install python@3.13
    ```

  - **macOS on an Intel Mac, or on macOS 14 and older:** Homebrew no longer supports these Macs, so use Python's official installer. On [python.org/downloads/macos](https://www.python.org/downloads/macos/), pick the newest **Python 3.13** release (not 3.14 or later) and download its macOS installer. Run it, then open the new **Python 3.13** folder in Applications and double-click **Install Certificates.command**. Without this step, Python cannot download data over HTTPS. For git, run `xcode-select --install` in Terminal and click **Install**. (Intel Macs handle the early lessons well, but from lesson 16 on, some libraries no longer work on them: XGBoost needs Homebrew's OpenMP library, and current PyTorch has no Intel Mac version. Plan to use a free cloud notebook such as Google Colab for those lessons. An Apple Silicon Mac on an old macOS can instead update macOS for free and use Homebrew.)
  - **Windows (WSL2) and Ubuntu Linux:** these steps need Ubuntu 22.04, 24.04 or 26.04 (on an older Ubuntu PC, upgrade Ubuntu first). `sudo` runs a command as administrator and asks for the Linux password (the typing may not show on screen; that is normal). Ubuntu has its own Python, but depending on the Ubuntu version it is too old or too new for this curriculum. So add **deadsnakes**, a widely used extra package source, and install Python 3.13 next to Ubuntu's own `python3`, which stays untouched. `curl` downloads files and `unzip` unpacks .zip archives; later lessons use both:

    ```bash
    sudo apt update
    sudo apt install -y software-properties-common git curl unzip
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt install -y python3.13 python3.13-venv
    ```

  - **Other Linux distributions:** install Python 3.13 (with its venv module) and git with the system's package manager. On Fedora, run `sudo dnf install -y python3.13 git`; on Debian 13, run `sudo apt install -y python3.13-venv git curl unzip`.

  Checkpoint: `python3.13 --version` prints `Python 3.13.` followed by a number, and `git --version` prints a version.

- [ ] Fork and clone this repository. Project 3 pushes work to GitHub, and GitHub only accepts pushes to a repository that the account owns. So first **fork** this repository, which copies it into the GitHub account: open [github.com/anurag-2911/ml-learn](https://github.com/anurag-2911/ml-learn), click **Fork**, then **Create fork**. Then **clone** the fork, which downloads it to the computer, into `~/ml/ml-learn`, the path every lesson uses. Replace `YOUR-USERNAME` with the GitHub username:

  ```bash
  git clone https://github.com/YOUR-USERNAME/ml-learn.git ~/ml/ml-learn
  cd ~/ml/ml-learn
  ```

  Checkpoint: `pwd` prints a path ending in `/ml/ml-learn`, `ls` shows `README.md`, `PROGRESS.md` and the phase folders, and `git remote -v` shows the GitHub username, not `anurag-2911`.

- [ ] Create a virtual environment at the repo root. A virtual environment is a private folder of Python packages for this project only, so nothing installed in it can break anything else on the computer. From `~/ml/ml-learn`, run:

  ```bash
  python3.13 -m venv .venv
  source .venv/bin/activate
  ```

  `source ... activate` switches the terminal into the venv. Inside it, `python`, `python3` and `pip` all mean the venv's own Python 3.13 and its pip. Later lessons use both `python` and `python3`; inside the venv they are the same. Checkpoint: the prompt now starts with `(.venv)`, `python --version` prints Python 3.13, and `which python` prints a path ending in `ml-learn/.venv/bin/python`. Every work session for the rest of the curriculum starts with `source .venv/bin/activate`, so make it a habit.

- [ ] Tell git to ignore files that should never be uploaded. Create a file named `.gitignore` at the repo root containing at least these five lines (if the file already exists, make sure they are in it):

  ```
  .venv/
  __pycache__/
  .ipynb_checkpoints/
  .DS_Store
  *:Zone.Identifier
  ```

  The venv is large and only works on the computer that made it. `__pycache__/` and `.ipynb_checkpoints/` are scratch folders that Python and Jupyter create. `.DS_Store` files are clutter that macOS leaves in every folder opened in Finder, and `Zone.Identifier` files are clutter that Windows adds when downloaded files are dragged into Ubuntu. One way to create the file is `nano .gitignore`. Nano is a simple text editor that runs in the terminal: type the lines, press Ctrl+O and then Enter to save, and Ctrl+X to exit. (That is the Control key, not Command, even on a Mac.) Checkpoint: `cat .gitignore` prints the five lines.

- [ ] With the venv active, install the core libraries. `pip` downloads packages from the internet. The first command updates pip itself, and the second can take a few minutes:

  ```bash
  pip install --upgrade pip
  pip install jupyter numpy pandas matplotlib
  ```

  Checkpoint: the install ends with a "Successfully installed ..." line (or only "Requirement already satisfied" lines on a second run) and no line starting with `ERROR`.

- [ ] Check that everything imports. This one-line command runs a tiny Python program straight from the terminal:

  ```bash
  python -c "import numpy, pandas, matplotlib; print('workshop ready')"
  ```

  Checkpoint: it prints `workshop ready` and nothing else.

- [ ] Set up VS Code, the code editor:

  - **macOS:** download VS Code from [code.visualstudio.com](https://code.visualstudio.com), open the downloaded file and drag **Visual Studio Code** into Applications, then open it from there. Press Cmd+Shift+P, run **Shell Command: Install 'code' command in PATH** (it asks for the Mac password), and open a new Terminal window.
  - **Windows (WSL2):** install VS Code on the *Windows* side from [code.visualstudio.com](https://code.visualstudio.com), never inside Ubuntu, and leave the installer's **Add to PATH** option ticked. Open it, install the **WSL** extension (search "WSL" in the Extensions panel), then close and reopen the Ubuntu terminal.
  - **Linux:** on Ubuntu, run `sudo snap install --classic code`. On other distributions, or on an ARM computer (where that snap does not exist), download the `.deb` (Debian, Ubuntu) or `.rpm` (Fedora) package from [code.visualstudio.com](https://code.visualstudio.com). On an ARM computer, click the small **Arm64** link instead of the big button. Install the package with `sudo apt install -y ~/Downloads/code_*.deb` or `sudo dnf install -y ~/Downloads/code-*.rpm`. If apt asks whether to add Microsoft's repository, answer Yes, so that VS Code keeps itself updated. A last line starting `N: Download is performed unsandboxed` is harmless.

  Then run `cd ~/ml/ml-learn` and `source .venv/bin/activate` (a new terminal window always starts in the home folder with the venv off), and then `code .` (the dot means "this folder"). VS Code asks whether to trust the authors of the files in this folder. Choose **Yes, I trust the authors**: the folder is the fork made earlier in this project. On Windows, the bottom-left corner of the window now shows **WSL: Ubuntu** (or the name of the Ubuntu version in use, such as **WSL: Ubuntu-26.04**), which means VS Code is working inside Linux. Open the Extensions panel (Cmd+Shift+X on a Mac, Ctrl+Shift+X elsewhere) and install **Python** and **Jupyter**, both published by Microsoft. The Python extension does not include Jupyter. On Windows, if a button says **Install in WSL: Ubuntu**, click it. Checkpoint: open this lesson file in VS Code and press Cmd+Shift+V on a Mac (Ctrl+Shift+V elsewhere). The lesson appears neatly formatted, as on GitHub.

<details><summary>Hints</summary>

- **Mac:** if running `git` or `python3` brings up an offer to install "command line developer tools", click **Install**, wait, and try again.
- **Mac:** `zsh: command not found: brew` means the Homebrew **Next steps** were skipped. Run those three lines, then open a new Terminal window.
- **Windows:** if `wsl --install` fails with an error about virtualization, virtualization must be switched on in the PC's firmware (BIOS/UEFI) settings. Search the web for the PC model plus "enable virtualization". If it says "Cannot create a file when that file already exists", Ubuntu is already installed: open it from the Start menu and check its version as described in the note after step 4 of the first milestone.
- **Windows:** keep all curriculum files inside Ubuntu (under `~`), never under `/mnt/c/...`, because Linux tools are much slower on Windows files. To see the Ubuntu files in Windows Explorer, run `explorer.exe .` (with the dot).
- **Windows and Ubuntu:** if `sudo apt update` complains about a lock file, another update is running in the background. Wait a minute and try again. `E: Unable to locate package python3.13` means the Ubuntu version is older than 22.04 (`lsb_release -d` shows it). On Windows, add Ubuntu-26.04 as described in the first milestone; on a Linux PC, upgrade Ubuntu first.
- **Debian:** if `sudo` says the user is not in the sudoers file (or `sudo: command not found` appears), a root password was set while installing Debian, so the account cannot use sudo yet. Run `su - -c "apt install -y sudo && usermod -aG sudo $USER"`, type the root password, then restart the computer and run the install command again.
- `git clone` asks for a username or password, or fails with "Authentication failed" or "Repository not found": the address is wrong, because cloning a public fork never needs a password. Press Ctrl+C. Check that the fork exists at `https://github.com/YOUR-USERNAME/ml-learn` and that the real GitHub username replaced `YOUR-USERNAME`, then clone again. "destination path already exists": `~/ml/ml-learn` is already there, so `cd` into it instead of cloning again.
- `git remote -v` shows `anurag-2911`: the clone came from the original repository instead of the fork. Fork it as described above, then run `git remote set-url origin https://github.com/YOUR-USERNAME/ml-learn.git`.
- If the prompt does not show `(.venv)`, the venv is not active in *this* terminal window. Activation only lasts for the window where it was run. Run `cd ~/ml/ml-learn` and then `source .venv/bin/activate`.
- `error: externally-managed-environment`, `command not found: python` or `pip`, or (on Ubuntu and WSL) "Command 'python' not found" or "Command 'pip' not found" all have the same cause: the venv is not active. Activate it instead of installing what Ubuntu suggests (`python-is-python3` or `python3-pip`). Those packages make `python` and `pip` run Ubuntu's own Python whenever the venv is off, which hides this mistake. Never work around it with `sudo pip`.
- `python --version` inside the venv must print 3.13. If it prints another version, the venv was made with the wrong Python. Run `deactivate`, delete the venv with `rm -rf .venv`, and create it again with `python3.13 -m venv .venv`.
- **Intel Mac:** if `pip install` fails while building a package, add `--prefer-binary` so that pip picks a ready-made version: `pip install --prefer-binary jupyter numpy pandas matplotlib`.
- If `code .` says "command not found": on a Mac, repeat the **Shell Command** step, then open a new Terminal window. On Windows, close every Ubuntu window and open a new one. If it still fails, reinstall VS Code with **Add to PATH** ticked, and ignore Ubuntu's suggestion to `snap install code`, which would put a second VS Code inside Linux. On Linux, open a new terminal.
- Hidden files and folders on macOS and Linux start with a dot, which is why `ls` does not show `.venv` or `.gitignore`. Use `ls -a` to see them. (In the Finder on a Mac, Cmd+Shift+. shows and hides them.)

</details>

**Definition of done:** the import one-liner prints `workshop ready`, and `code .` opens the repo in VS Code with the Python and Jupyter extensions installed (and, on Windows, **WSL** showing in the bottom-left corner).

## Project 2 — Hello, data

**Goal:** Write a first real Python file, a script that prints a small multiplication table using a loop, and run it from the terminal. A **script** is a text file of Python commands that run from top to bottom.

**Milestones**

- [ ] Create the work folder for this lesson and move into it, with the venv active:

  ```bash
  cd ~/ml/ml-learn
  source .venv/bin/activate
  mkdir -p work/01-environment-setup
  cd work/01-environment-setup
  ```

  (`-p` means "create parent folders as needed".) Checkpoint: the prompt starts with `(.venv)` and `pwd` ends in `work/01-environment-setup`.
- [ ] In VS Code, create a new file named `hello.py` in that folder (in the Explorer panel, right-click `work/01-environment-setup` and choose **New File**). Start small, with a program that prints one line:

  ```python
  print("hello, data")
  ```

  Save the file with Cmd+S on a Mac or Ctrl+S elsewhere. VS Code does not save automatically, and a dot on the file's tab means there are unsaved changes. Run the script from the terminal (with the venv active) with `python hello.py`. Checkpoint: the terminal prints `hello, data`.
- [ ] Learn the loop. A **for loop** repeats a block of code once for each item. `range(1, 6)` gives the numbers 1 to 5 (the end number is not included). An **f-string** is a string with an `f` before the opening quote: Python fills in the value of anything written inside its curly braces. Try this in `hello.py` and run it:

  ```python
  for i in range(1, 6):
      print(f"3 x {i} = {3 * i}")
  ```

  The four spaces of indentation tell Python which lines are inside the loop, and they are required. Checkpoint: five lines, from `3 x 1 = 3` to `3 x 5 = 15`.
- [ ] Now write the full version: replace that code so the script prints the multiplication table for the numbers 1 to 5, every combination, which makes 25 lines like `4 x 3 = 12`. This needs a loop inside a loop. Checkpoint: `python hello.py` prints 25 correct lines, starting with `1 x 1 = 1` and ending with `5 x 5 = 25`.
- [ ] Make one small change for readability: print a blank line after each number's block (the 1-times block, a blank line, the 2-times block, and so on). Checkpoint: the output shows 5 tidy blocks.

<details><summary>Hints</summary>

- A loop inside a loop looks like this: an outer `for i in range(...):` and, indented inside it, another `for j in range(...):` with the `print` indented further still.
- `print()` with nothing inside prints a blank line. Where it goes (inside the outer loop, after the inner loop) decides where the blank lines appear.
- `IndentationError` means the spacing is inconsistent. Use exactly 4 spaces per level, and never mix tabs and spaces. VS Code's default settings take care of this.
- Output did not change after an edit? The file is probably not saved. Check its tab for a dot.

</details>

**Definition of done:** `python hello.py` prints the full 1-to-5 multiplication table, 25 lines in 5 blocks, with no errors.

## Project 3 — First notebook

**Goal:** Create a Jupyter notebook, plot y = x² with matplotlib inside VS Code, then use git to take a snapshot of everything and push it to GitHub. A **notebook** is an interactive document that mixes runnable code cells with their outputs. It is the standard tool for ML experiments.

**Milestones**

- [ ] In VS Code, create `first-plot.ipynb` inside `work/01-environment-setup/` (right-click the folder in the Explorer, choose **New File**, and type the name; the `.ipynb` ending makes it a notebook). Keep the repo root, `~/ml/ml-learn`, as the folder open in VS Code, so that VS Code can find the venv. When the first cell runs, VS Code asks which "kernel" (Python) to use. Choose **Python Environments**, then `.venv` (its path ends in `.venv/bin/python`). Checkpoint: a cell containing `1 + 1` runs with Shift+Enter and shows `2` below it.
- [ ] Make a cell that plots y = x². Here is an outline; the y values are left to fill in:

  ```python
  import matplotlib.pyplot as plt

  xs = list(range(-10, 11))       # the numbers -10 to 10
  ys = ...                        # to do: y = x*x for each x (a loop that builds a list)
  plt.plot(xs, ys)
  plt.title("y = x squared")
  plt.show()
  ```

  Checkpoint: the plot appears right below the cell and shows a clean U-shape, lowest at x = 0.
- [ ] Add one markdown cell above the plot (hover over the top edge of the plot cell and click **+ Markdown**). Give it a title line starting with `# ` and one sentence about what the notebook shows, then press Shift+Enter. Save the notebook (Cmd+S or Ctrl+S). Checkpoint: the text appears as a heading, not as code.
- [ ] Next, save the work with git. First, git needs a name and an email address to record in each commit. Set them once per computer (on Windows, inside Ubuntu). The fork is public, and so is the email address in every commit, so use a private address from GitHub instead of a real one. On github.com, open **Settings → Emails** and make sure **Keep my email addresses private** is ticked (this also covers commits that GitHub makes, such as those from **Sync fork**). Then copy the "noreply" address shown there (it looks like `12345678+YOUR-USERNAME@users.noreply.github.com`). Give git a name and that address:

  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "12345678+YOUR-USERNAME@users.noreply.github.com"
  ```

  Then go back to the repo root and look at what has changed:

  ```bash
  cd ~/ml/ml-learn
  git status
  ```

  Checkpoint: `git config user.email` prints the real noreply address (with the account's own number and username, not `12345678+YOUR-USERNAME`), and `git status` lists `.gitignore` and `work/` under "Untracked files" but not `.venv/` or `.DS_Store`. (`git status -u` also lists every file inside `work/`.)
- [ ] Stage and commit. **Staging** (`git add`) picks which changes go into the snapshot, and a **commit** is the snapshot itself, saved with a message:

  ```bash
  git add .
  git commit -m "Lesson 01: workshop setup, hello.py and first notebook"
  ```

  Checkpoint: `git status` now says "nothing to commit, working tree clean".
- [ ] Give GitHub a key to this computer. This is done once per computer (on Windows, inside Ubuntu). GitHub does not accept the account password for pushing. It uses an **SSH key** instead: a pair of files made of a private half, which never leaves the computer, and a public half, which goes to GitHub. Create a key, pressing Enter at each question to accept the defaults. (A passphrase is optional. If one is set, git asks for it when pushing. If `ssh-keygen` asks whether to overwrite an existing key, answer `n` and use the existing key.)

  ```bash
  ssh-keygen -t ed25519 -C "you@example.com"
  cat ~/.ssh/id_ed25519.pub
  ```

  The second command prints the public half: one line starting with `ssh-ed25519`. Select it with the mouse and copy it (Cmd+C on a Mac, Ctrl+Shift+C in Windows and Linux terminals). On github.com, click the profile picture → **Settings** → **SSH and GPG keys** → **New SSH key**, paste the line into **Key**, give it a title such as the computer's name, and click **Add SSH key**. Then test the connection:

  ```bash
  ssh -T git@github.com
  ```

  The first time, ssh shows a key fingerprint and asks `Are you sure you want to continue connecting (yes/no/[fingerprint])?`. Check that the fingerprint matches the ED25519 one listed on GitHub's [SSH key fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) page, and only then type `yes`. If it does not match, type `no` and stop. Checkpoint: the reply starts with `Hi` and the GitHub username, followed by `You've successfully authenticated`.
- [ ] Switch the clone from GitHub's HTTPS address to its SSH address, so that pushes use the key:

  ```bash
  git remote set-url origin git@github.com:YOUR-USERNAME/ml-learn.git
  ```

  Checkpoint: `git remote -v` shows `git@github.com:` followed by the GitHub username and `/ml-learn.git` on both lines, not `YOUR-USERNAME` or `anurag-2911`.
- [ ] Push the commit to GitHub with `git push`. A **push** uploads local commits to the fork on GitHub's servers. Checkpoint: the output ends with a line like `main -> main` and no error.
- [ ] Open the fork at `https://github.com/YOUR-USERNAME/ml-learn` in a browser (the fork, not the original) and click into `work/01-environment-setup/`. Checkpoint: `hello.py` and `first-plot.ipynb` are there, and GitHub shows the notebook with its U-shaped plot.

<details><summary>Hints</summary>

- To build `ys`, start with an empty list `ys = []`, loop over `xs`, and `ys.append(...)` each squared value. (In Python, x squared is `x * x` or `x ** 2`.)
- If VS Code cannot find a kernel, check three things: VS Code has the repo root open (run `code .` from `~/ml/ml-learn`), the Python and Jupyter extensions are installed (on Windows, in WSL), and jupyter was installed inside the venv in Project 1. Then click **Select Kernel** at the top right of the notebook.
- `ssh -T` says `Permission denied (publickey)`: GitHub does not have this computer's public key yet. Repeat the **New SSH key** step, pasting the line that `cat ~/.ssh/id_ed25519.pub` prints. If `ssh -T` just hangs, the network may block SSH. GitHub's guide to using SSH over the HTTPS port (listed under Resources) fixes that.
- `git push` asks for a username for `https://github.com`: the clone still uses the HTTPS address. Run the `git remote set-url` command above, then push again. `ERROR: Repository not found`: the address given to `git remote set-url` is wrong, usually because `YOUR-USERNAME` was not replaced. Run it again with the exact username.
- `git push` says "Permission to anurag-2911/ml-learn.git denied" or "ERROR: Permission ... denied": the clone points at the original repository instead of the fork. Run the `git remote set-url` command above with the GitHub username, then push again.
- `git push` fails with "GH007: Your push would publish a private email address": set the noreply address with `git config --global user.email ...` as shown above, run `git commit --amend --reset-author --no-edit`, and push again.
- `git push` is rejected with `(fetch first)`, or `git pull` stops with `fatal: Need to specify how to reconcile divergent branches`: the fork on GitHub and the computer each have commits that the other does not (usually after **Sync fork**, described in the next hint). Run `git pull --no-rebase --no-edit`, then `git push`.
- Later, when the original repository gets new or updated lessons, first commit and push all local work. Then, on the fork's GitHub page, click **Sync fork**, then **Update branch**, and run `git pull`. If GitHub says the branch has conflicts, do not click **Discard commits**: that deletes the lesson work saved in the fork. Ask for help instead.

</details>

**Definition of done:** the notebook shows a U-shaped parabola inside VS Code, and all of this lesson's files are visible in the fork on github.com.

## Stretch goals

- Make `hello.py` ask for the table size with `input()`, so that `python hello.py` asks "Up to what number?" and prints a table of that size.
- In the notebook, plot y = x² and y = x³ on the same chart and add a legend (search matplotlib's documentation for `plt.legend`).
- Learn 3 more terminal commands and try each one: `cp` (copy), `mv` (move or rename) and `rm` (delete). Use `rm` with care: there is no recycle bin.
- Run `git log` and read the commit history (press `q` to return to the prompt). Then make a small change, commit it with a clear message, and push again. Small, frequent commits make the history easy to follow.

## Getting unstuck

- **Read error messages from the bottom up.** The last line usually names the real problem. For example, `ModuleNotFoundError: No module named 'numpy'` means the package is missing *in the Python that is running*.
- **"Module not found" almost always means the venv is not active.** Look for `(.venv)` at the start of the prompt. If it is missing, run `source .venv/bin/activate` from the repo root.
- **"Command not found" means the terminal cannot find the program.** Either the program is not installed, or it was installed after this terminal window opened and a new window is needed.
- **A command in a later lesson fails on one system only?** A few later lessons still assume Windows with WSL. Some download files with `wget`, which Macs do not have (use `curl -L -o FILE URL` instead), and some copy from a Windows folder under `/mnt/c/...` (use a folder on the local computer instead). An AI assistant can suggest the equivalent command for other systems.
- **Print things.** When code misbehaves, add `print(...)` lines to show what values the variables really hold.
- **Ask an AI assistant for a hint, not a solution.** Paste the error and ask "what does this mean?", never "write this for me". Working through the problem is where the learning happens.
- **Type all project code by hand.** Copying commands such as `pip install ...` is fine, but pasting in project code or finished solutions defeats the purpose of the curriculum.

## Resources

- [The official Python tutorial](https://docs.python.org/3/tutorial/) — skim sections 1-3 for a feel of the language; lessons 02-04 cover it project by project.
- [Homebrew](https://brew.sh) and [Python releases for macOS](https://www.python.org/downloads/macos/) — the two ways to install Python on a Mac.
- [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) — Microsoft's official guide, for problems with `wsl --install`.
- VS Code setup guides for [macOS](https://code.visualstudio.com/docs/setup/mac), [Linux](https://code.visualstudio.com/docs/setup/linux) and [Windows with WSL](https://code.visualstudio.com/docs/remote/wsl), plus [Jupyter notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
- GitHub's guides to [forking a repository](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo), [syncing a fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/syncing-a-fork), [connecting with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and [using SSH over the HTTPS port](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port).

## Skills unlocked

- [ ] I can move around folders from the terminal with `cd`, `ls`, `pwd` and `mkdir`.
- [ ] I can install developer tools with a package manager (or, on an older Mac, an official installer).
- [ ] I can explain in one sentence what a virtual environment is and why projects use one.
- [ ] I can activate the venv and install packages into it with pip.
- [ ] I can write a Python file with a loop in it and run it from the terminal.
- [ ] I can create and run a Jupyter notebook inside VS Code, on the right kernel.
- [ ] I can make a basic matplotlib plot.
- [ ] I can fork and clone a repository, connect to GitHub with an SSH key, stage, commit and push work, and find it on github.com.

## Next up

With the workshop in place, the next lesson teaches the Python language through small games: [02 · Python Basics by Building Games](02-python-basics.md).
