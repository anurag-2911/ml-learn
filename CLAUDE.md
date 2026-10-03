# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is written for maintaining the curriculum upstream, but every learner's fork inherits it. First run `git remote get-url origin`. Unless it points to the upstream repository (`github.com/anurag-2911/ml-learn`, over SSH or HTTPS), you are helping a learner in their own fork: do not edit lesson files, `README.md` or this file there (Sync fork would conflict when upstream changes them), give hints rather than solutions (the curriculum's own rule for AI assistants), let the learner commit their own `work/`, `.gitignore` and `PROGRESS.md` ticks with `git add .` as lesson 01 teaches, and ignore the Maintaining upstream, Pending work and Checking edits sections.

## What this repository is

"ML Zero to Hero" (public at github.com/anurag-2911/ml-learn): a project-based curriculum of 42 Markdown lessons in 8 phase folders that takes a learner from zero programming to building GPTs, RAG systems and agents. There is no application code, build or test suite — the product is the lesson text. `README.md` is the curriculum map; `PROGRESS.md` is the learner's checklist. Learners fork the repo, clone their fork to `~/ml/ml-learn`, and write all their own code under `work/<NN>-<slug>/`, which is not part of the curriculum itself.

## Structure and cross-file invariants

- Lessons are numbered 01-42 across the whole curriculum (not per phase): `phase-N-<name>/<NN>-<slug>.md`.
- New and rewritten lessons use this skeleton, in this order: `# NN · Title`; a metadata line (`**Phase N — Name** · Estimated time: ... · Prerequisites: ...`); a blockquote intro; `## What you will build`; `## Concepts you will learn by doing`; `## Before you start`; one to four `## Project N — Name` sections (lesson 20 has one); `## Stretch goals`; `## If you get stuck`; `## Resources`; `## Skills unlocked`; `## Next up` (lesson 42 ends with `## The journey continues` instead). Lesson 24 adds `## Your secret decoder table`.
- Each Project section should have `**Goal:**`, `**Milestones**` written as GitHub task-list items (`- [ ]`) that each end in an observable `Checkpoint:`, a `<details><summary>Hints</summary>` block, and `**Definition of done:**`. Older lessons vary in small ways (some write `**Goal**:`, `**Goal** —` or `**Goal.**`; about a third of their milestones have no explicit `Checkpoint:`). Bring a lesson to this skeleton when you rewrite it, but do not normalize lessons you were not asked to touch.
- Adding or renaming a lesson means updating the README phase table, `PROGRESS.md` (see the next point), the previous lesson's `## Next up` link, any "Prerequisites" or "lesson NN" references, and, when adding, the "42 lessons" counts in `README.md`, lesson 01 and this file. Never reorder lessons: that renumbers them. `(K)` in the README marks lessons built on Andrej Karpathy's videos.
- Learners edit their own forks, and Sync fork merges upstream changes into them, so some upstream edits conflict with learner work:
  - `PROGRESS.md` is ticked, dated and annotated by every learner. Its titles are short labels, so a reworded lesson title does not need to be mirrored there. Leave existing lines alone, add lines only for new lessons, and never renumber existing lessons (that would also break learners' `work/NN-slug` folders). Keep at least one unchanged line between a new line and any line a learner may have ticked: a lesson-43 line directly under 42 conflicts for everyone who finished, while a new section after the blank line below 42 merges cleanly.
  - Never add a top-level `.gitignore` or anything under `work/` upstream. Every learner creates their own `.gitignore` in lesson 01, so an upstream one causes an add/add conflict on Sync fork, and GitHub's conflict prompt offers to discard the learner's commits. If a `.gitignore` is ever added on purpose, lesson 01's `git status` checkpoint must change with it.
- Lesson 01 (`phase-0-foundations/01-environment-setup.md`) defines the environment every later lesson assumes: the repo at `~/ml/ml-learn`; one Python 3.13 venv at the repo root (`.venv`, activated with `source .venv/bin/activate`, after which `python`, `python3` and `pip` all mean the venv's); git, curl and unzip on every platform; VS Code opened on the repo root with the Python and Jupyter extensions; and pushes over SSH to the learner's own fork.

## Writing conventions

- No emojis anywhere — house style; they read as unprofessional. Use plain-text markers such as `(K)` or `Milestone:`.
- Pedagogy: concepts are explained inside the projects, exactly when needed; learners type all code themselves; hints nudge but never give full solutions; skeleton code marks the learner's part with `...`; every checkpoint is something the learner can see (printed output, a file, a plot).
- House typography: `·` in lesson titles, em dashes, `→` for menu paths. Code blocks inside `- ` list items are indented two spaces (four inside a nested platform bullet). Inside `1.`-style numbered items they are indented three spaces, as in lessons 04, 16, 25 and 27 — CommonMark needs that, so do not "fix" it.
- Placeholders are written `YOUR-USERNAME` — never `<your-username>` (the shell reads `<` as a redirect) — and `~` is never put inside quotes in commands.
- No `#` comments inside shell code blocks, neither after a command nor on a line of their own; put the explanation in the sentence before or after the block. zsh, the macOS default shell, does not treat `#` as a comment in commands that are typed or pasted: `pip install numpy  # note` fails with `ERROR: Invalid requirement: '#'`, and `cd DIR  # note` fails with `cd: too many arguments`. Lesson 01 has Mac learners add `setopt interactivecomments` to `~/.zshrc`, but lessons must not rely on it.
- Quote pip extras (`pip install "pkg[extra]"`): unquoted brackets fail in zsh.
- Prefer `python` over `python3` in new text: inside the venv they are the same, but outside it `python` fails loudly on macOS and Ubuntu, while `python3` quietly runs the system Python.

## Platform policy

Every lesson must work for learners on macOS (Apple Silicon and Intel), Windows via WSL2 (Ubuntu), and native Linux (Ubuntu 22.04 or newer first; other distributions get a one-line pointer). Lesson 01 was rewritten to this standard on 2026-09-30; use it as the pattern:

- Where steps differ, give labeled sub-bullets in the order **macOS**, **Windows (WSL2)**, **Linux**. Everything unlabeled must be identical on all three: POSIX shell commands that behave the same in bash and zsh.
- Windows learners use WSL2 rather than native Windows Python, so paths such as `.venv/bin/python` hold everywhere.
- System packages: give `brew install ...` for Apple Silicon Macs on macOS 15 or later and `sudo apt install ...` for Ubuntu and WSL, plus a route for Macs without Homebrew: Intel Macs, which cannot install Homebrew any more (Homebrew 7.0, September 2026), and Macs on macOS 14 or older, which lesson 01 sends to the python.org installer. Prefer a pip-only alternative; where none exists, say so and point to Google Colab. This includes pip packages that load a Homebrew library at import time, such as xgboost and lightgbm (libomp).
- Never hard-code one person's machine (absolute home paths such as `/home/NAME` or `/Users/NAME`, personal names, or notes about what someone already has installed) or Windows-only paths (`/mnt/c/...`) without macOS and Linux equivalents; write `~` for the home folder.
- Python is pinned to 3.13: the venv is created with `python3.13 -m venv .venv`. As of September 2026, 3.14 cannot install gensim (lesson 30) or Box2D (lesson 39), 3.15 lacks many wheels, and 3.10/3.11 only get older numpy and scipy. Sources: Homebrew `python@3.13` (Apple Silicon, macOS 15+); the python.org 3.13 installer plus `Install Certificates.command` (Intel Macs and older macOS); the deadsnakes PPA (Ubuntu 22.04, 24.04 and 26.04 only — it publishes nothing for 20.04 or older, so lesson 01 sends those users to a newer Ubuntu; a fresh `wsl --install` now gives Ubuntu 26.04, whose own Python is 3.14); Fedora's own `python3.13` package (Fedora 43/44, whose system Python is 3.14); and Debian 13's system Python 3.13 plus `python3.13-venv`. Revisit the pin once gensim and Box2D ship 3.14 wheels — but Debian 13 has no Python 3.14 package, so a 3.14 pin needs a new route for Debian 13 learners.
- GitHub: learners fork anurag-2911/ml-learn, clone their fork over HTTPS in Project 1, then in Project 3 add an SSH key and run `git remote set-url origin git@github.com:YOUR-USERNAME/ml-learn.git`. SSH was chosen over the GitHub CLI because it needs no install and is identical on every platform (the `gh` packages in Ubuntu's own archive are reported broken by the gh maintainers). Commit email: the GitHub noreply address, with Keep my email addresses private ticked so that commits GitHub makes for the learner (Sync fork merges) use it too, because forks of a public repo are public. The lesson points to GitHub's published SSH host key fingerprints instead of quoting one, so a key rotation cannot make it wrong.

## Pending work

As of 2026-09-30 these lessons still assume WSL or otherwise break on some platform, and need the lesson-01 treatment. Three texts in lesson 01 depend on this list: its "If you get stuck" section has a stopgap note about `wget` and `/mnt/c` paths (remove it once those lessons are converted); the macOS terminal bullet justifies `setopt interactivecomments` with "some commands in later lessons end with a note after a `#`" (reword it, for example to commands copied from other websites, once the `#` item is done); and Concepts says "almost every command in this curriculum is the same on all three" (drop "almost" once the list is empty).

- 02: "WSL2 Ubuntu" in Before you start; `code .` is run inside `work/02-python-basics`, so VS Code cannot see the repo-root `.venv` (open the repo root instead).
- 03, 25, 28, 29, 30, 31, 40: downloads use `wget`, which macOS does not ship; switch to `curl -L -o FILE URL` (lesson 01 installs curl on Ubuntu and Debian, whose desktop images lack it).
- 08, 15, 17, 21, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40: shell blocks contain `#` comments (see Writing conventions); the pip lines, 37's `cd`, the `cp` lines in 31 and 32, and 29's `unzip` fail outright in zsh.
- 05, 18, 26, 27: copying photos from `/mnt/c/Users/...`.
- 07: the `plt.show()` advice names only WSL2, but the cause is Linux-wide: lesson 01's deadsnakes Python has no tkinter (`python3.13-tk` is a separate package), so on Ubuntu and WSL `plt.show()` in a script falls back to the non-interactive Agg backend and opens no window. Keep the `savefig` advice for Ubuntu and WSL, or add `sudo apt install python3.13-tk`; on macOS a script window opens normally.
- 16: Before you start installs XGBoost and checks `import xgboost`, and Project 3 is built on it. Its macOS wheels do not bundle libomp and look for it only in Homebrew's folders, so Apple Silicon Macs need `brew install libomp` and Macs without Homebrew cannot import it; give those learners scikit-learn's HistGradientBoostingClassifier/Regressor (the scikit-learn wheel bundles its own libomp) or Colab. The optional LightGBM row has the same limit.
- 20: the same `brew install libomp` note for XGBoost and LightGBM (Milestone 4 already offers HistGradientBoostingRegressor); moves `kaggle.json` from `/mnt/c/.../Downloads`.
- 21: screenshot shortcut Win+Shift+S (macOS: Cmd+Shift+4).
- 22: `sudo apt install graphviz` (Apple Silicon Macs with Homebrew: `brew install graphviz`; Macs without Homebrew skip the optional graph drawing, which the lesson already allows).
- 24, 31: "WSL2 without an NVIDIA GPU" and CPU-only wording; Apple Silicon uses the `mps` device (macOS 14+), and Intel Macs cannot install current PyTorch (point them to Colab).
- 25, 26, 27, 28, 29, 33, 38, 39: a plain `pip install torch ...` in a fresh Linux/WSL venv downloads the ~3 GB CUDA build. Put torch (and torchvision where used) on its own line with lesson 24's `--index-url https://download.pytorch.org/whl/cpu`, and install the other packages on those lines (matplotlib, gradio, jupyter and so on, which that index does not carry) with a separate plain `pip install`.
- 27: HEIC conversion with `apt install libheif-examples`; opens the demo "in your Windows browser"; its Hugging Face Space defaults to Python 3.10 unless `python_version` is set; `git lfs install` needs git-lfs, which no platform ships (`sudo apt install git-lfs`, `brew install git-lfs`, or the git-lfs.com download on Macs without Homebrew).
- 29: delete the `sudo apt-get install -y unzip` line (lesson 01 now installs unzip, and macOS ships it).
- 30: the gensim note is outdated — gensim 4.4.0 no longer pins an old NumPy, but it has no Python 3.14 wheels.
- 36: "your WSL2 machine" and Ollama "on WSL2"; `huggingface-cli login` no longer exists on any platform (huggingface_hub 1.0 removed it, and transformers 5 needs huggingface_hub 1.5 or newer), so use `hf auth login`; `ollama show <model> --modelfile` uses an angle-bracket placeholder.
- 39: pop-up rendering via WSLg; `gymnasium[box2d]` needs Python 3.13 or older on x86_64 or macOS.
- 41: Docker Desktop with the WSL2 backend, "Windows browser", and a `python:3.11-slim` Dockerfile that should match the 3.13 venv; its Gradio Space (Project 3) also needs `python_version` set in the Space README, as in 27; `docker logs <container-id>` and `docker logs <id>` use angle-bracket placeholders.
- 42: `peek` for GIF recording on Ubuntu; `pip install <whatever-you-need>` uses an angle-bracket placeholder.

## Maintaining upstream

The upstream checkout is the curriculum itself, and GitHub does not let an account fork its own repository, so lesson 01's fork, set-url and push steps do not apply there:

- Never commit `work/`, `.venv/`, a learner `.gitignore` or `PROGRESS.md` ticks to `main`: everything on `main` is copied into every new fork (handing out solutions and breaking lesson 01's `git status` checkpoint), and it makes existing forks' Sync fork conflict.
- Stage lesson edits by path (for example `git add phase-0-foundations/01-environment-setup.md`), never with `git add .`. Keep `.venv/` out of `git status` with `.git/info/exclude`, not with a committed `.gitignore`.
- Work through the lessons as a learner in a separate clone backed by a private repository (any branch pushed to the public repository is public too), and test lesson instructions in a scratch clone.
- Personal, machine-specific notes belong in `CLAUDE.local.md`, which Claude Code loads alongside this file; list it in `.git/info/exclude` so it is never committed.

## Checking edits

After editing lessons, run this from the repo root. It lists broken relative links, emoji characters, and `#` comments inside shell code blocks, and prints nothing when the repo is clean. Until the `#` item in Pending work is done it also prints 27 known lines from lessons 08-40; fix only lines in files you were asked to edit.

```bash
python3 - <<'EOF'
import pathlib, re
for p in sorted(pathlib.Path('.').rglob('*.md')):
    if {'.git', '.venv', 'work'} & set(p.parts): continue
    lang = None
    for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if line.lstrip().startswith('```'):
            lang = None if lang is not None else line.strip()[3:].strip()
            continue
        if lang in ('bash', 'sh', 'shell', 'zsh') and re.search(r'(^|\s)#', line):
            print(f'{p}:{n}: shell comment in a bash block')
        for t in re.findall(r'\]\(([^)\s#]+)', line):
            if not t.startswith(('http:', 'https:', 'mailto:')) and not (p.parent / t).exists():
                print(f'{p}:{n}: broken link {t}')
        if any(0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF or ord(c) == 0xFE0F for c in line):
            print(f'{p}:{n}: emoji')
EOF
```
