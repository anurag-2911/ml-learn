# 42 · Capstone: Your Hero Project (and What Comes Next)

**Phase 7 — MLOps & Capstone** · Estimated time: 4-6 weeks · Prerequisites: [41 · MLOps: Track, Serve, Containerize, Deploy](41-mlops-ship-your-models.md) and all earlier lessons (in Phase 6, at least one of lessons 38-40)

> This final lesson is the graduation project of the curriculum. [Lesson 01](../phase-0-foundations/01-environment-setup.md) started with a first Python script that printed a sentence. Since then, the lessons have built a gradient descent optimizer from a math formula, a k-NN classifier from scratch, a spam filter, a Kaggle pipeline, micrograd, a neural network in pure NumPy that reads handwritten digits, a CNN that sees, a GPT made layer by layer and then [trained](../phase-4-nlp-transformers/33-train-your-own-gpt.md), a RAG app that chats with personal documents, an AI agent, and a containerized model running behind a real API. Here the step-by-step lessons stop, and the work is a single substantial, self-chosen project: scoped, specced, built, deployed, and written up so well that it serves as proof of the whole journey. The lesson also covers scoping, one-page specs, weekly milestones, blog-style write-ups and sharing work in public, and it ends with a map of the specializations that come after the fundamentals.

## What this lesson builds

- **Project 1 — Choose and spec:** a one-page project spec (problem, users, data, success metric, four weekly milestones) for a hero project that combines skills from at least three phases.
- **Project 2 — Build:** the capstone itself, built in weekly milestones with a working end-to-end skeleton by the end of week 2.
- **Project 3 — Ship and tell:** a public GitHub repo with a blog-style README, a live demo link, and a post about it somewhere real. Together, the README and the repo form a public credential.
- **Project 4 — What next:** a personal 12-month specialization roadmap, plus one annotated research paper to start the next chapter.

## Concepts covered

- **Scoping**: choosing a project ambitious enough to be interesting but small enough to actually finish.
- **The one-page spec**: writing down the problem, users, data, and success metric *before* writing any code.
- **Weekly milestone planning**: breaking a month of work into four checkable deliverables.
- **Breadth before depth**: getting a crude end-to-end version working early, then improving the pieces.
- **The blog-style README** (problem, data, approach, results, demo, limitations, lessons): the write-up that turns code into a credential.
- **Sharing**: putting work in front of real people on GitHub, Hugging Face Spaces, LinkedIn/X and at local meetups.
- **The specialization map**: an honest overview of the field after the fundamentals, showing which directions exist and how to pick one.

## Before starting

Check the prerequisites. All of these should be true:

- [Lesson 41](41-mlops-ship-your-models.md) is finished, and at least one model has been deployed behind an API.
- There is a public demo from [lesson 27](../phase-3-deep-learning/27-transfer-learning-vision-project.md) on Hugging Face Spaces, and it is clear how to put up another one.
- A project can be started from an empty folder without a lesson saying what to type. (That is exactly what this lesson proves.)

Set up the workspace:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
mkdir -p work/42-capstone
cd work/42-capstone
```

There is no fixed pip install list this time, because the dependencies depend on the chosen project. Install each package when it is needed (inside the venv), and record every install in a `requirements.txt` from day one:

```bash
pip install WHATEVER-YOU-NEED
pip freeze > requirements.txt
```

`pip freeze` lists every package in the venv, including all those from earlier lessons. So before publishing, trim `requirements.txt` to the packages the project actually uses, as in lesson 41. On **Windows (WSL2)** and **Linux**, also delete the `+cpu` (or other `+` ending) from any torch or torchvision version left in it: pip finds such builds only on PyTorch's own index, so a Hugging Face Space or a stranger's computer could not install them.

One warning before beginning: the capstone will feel different from every lesson so far, because no one hands out milestones. Writing them is part of the work. That discomfort is the point. It is what working on real ML projects feels like.

## Project 1 — Choose and spec

**Goal:** Pick one project idea, pressure-test it, and write a one-page spec with a success metric and four weekly milestones, all before writing a single line of project code.

**The rule:** the capstone must (1) combine skills from at least **three phases** of this curriculum, and (2) end in a **deployed or demoable artifact** that a stranger can try. No exceptions. A notebook on a laptop is not a hero project.

**The idea list** (use one as it is, remix one, or bring a new one; the phases each idea draws on are in brackets):

1. **Personal study assistant**: RAG over personal notes from this curriculum, a fine-tuned open model for a chosen tone, and agent tools like "quiz me on lesson N" [phases 0, 4, 5].
2. **Local plant or bird identifier**: a transfer-learning CNN on photos of local species, deployed as a public demo where anyone can upload a photo [phases 0, 3, 7].
3. **End-to-end tabular product**: a real prediction service (rent estimator, crop-yield or air-quality predictor for a familiar city) with feature pipeline, API, monitoring dashboard, and scheduled retraining [phases 2, 0, 7].
4. **Niche-corpus GPT**: the from-scratch GPT trained on a favorite corpus (song lyrics, cricket commentary, old recipes, legal boilerplate) with a small web UI for generation [phases 3, 4, 7].
5. **RL game agent with replay viewer**: train an agent on a simple game and build a web page that replays its episodes so visitors can watch it learn [phases 3, 6, 7].
6. **"What should I watch tonight?"**: a recommender trained on MovieLens plus personal ratings, served behind an API with a simple front end [phases 2, 6, 7].
7. **Semantic search for a hobby**: embeddings-based search over a niche corpus (climbing routes, board-game rules, ghazals) with a hand-built evaluation set [phases 4, 5, 7].
8. **Doodle recognition game**: a browser game where people draw and the project's CNN guesses, with a leaderboard [phases 3, 0, 7].
9. **Audio classifier**: bird songs or music genres via spectrograms (pictures of sound) + CNN, with a record-and-classify demo [phases 0, 3, 7].
10. **Repo agent**: an AI agent that reviews its creator's own GitHub commits, suggests improvements, and is evaluated against a hand-curated test set of known-bad commits [phases 5, 2, 7].

**Milestones**

- [ ] Spend one day (no more) shortlisting 2-3 ideas. For each, answer in writing: What data would I use, and can I get it *today* without a signup wall? Who would actually try the demo? Which 3+ phases does it use? Checkpoint: any idea whose data source cannot be named concretely is eliminated.
- [ ] Run the "weekend test" on the favorite idea: could a laughably crude version be built in two days? If the honest answer is "not even close", shrink the idea (fewer classes, smaller corpus, one feature instead of five) until the answer is yes. Scoping means cutting until finishing is plausible.
- [ ] Write the one-page spec as `work/42-capstone/SPEC.md` with exactly these sections: **Problem** (2-3 sentences, no jargon), **Users** (who tries the demo and why they would care), **Data** (source, size, how to get it), **Success metric** (one number and a target, e.g. "top-3 accuracy ≥ 0.85 on a held-out set of 200 photos I labeled"; [lesson 14](../phase-2-classical-ml/14-model-evaluation.md) covered how to pick metrics), **Weekly milestones** (four, one line each, each ending in something checkable).
  Starter skeleton to copy:

  ```markdown
  # <Project name> — one-page spec
  ## Problem        <!-- 2-3 sentences, no jargon -->
  ## Users          <!-- who tries the demo, why they care -->
  ## Data           <!-- source, size, how I get it today -->
  ## Success metric <!-- one number + target, e.g. top-3 accuracy ≥ 0.85 -->
  ## Weekly milestones
  1. Week 1: ...
  2. Week 2: crude end-to-end pipeline works
  3. Week 3: ...
  4. Week 4: deployed demo + write-up
  ```

- [ ] Stress-test the spec: ask an AI assistant to play a skeptical reviewer and find the three biggest risks in it. Do not let it rewrite the spec. Argue with it, then revise the spec without its help. Checkpoint: the week-2 milestone says some version of "crude end-to-end pipeline works".
- [ ] Create the project's own public GitHub repo (separate from this learning repo), commit `SPEC.md` as its first commit, and add the spec's problem statement as the repo description. The learning repo's `.gitignore` does not apply to the new repo, so give it its own before adding any code: `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, `.env` (the API key from lesson 34), the data folder and the model files (for example `*.pt`). Checkpoint: `git status -u` lists none of them.

<details><summary>Hints</summary>

- The most common capstone failure is scope, not skill. When torn between two ideas, pick the one with the boring, available data. A finished small project beats an abandoned giant one every time.
- A good success metric is one that can be computed in week 1 with a dumb baseline (majority class, random recommender, tiny model). If it cannot be baselined in week 1, the metric is too vague.
- "Users" can be five friends and a subreddit. Naming them changes the design decisions: a demo for bird-watchers needs species names, not class indices.
- Data-availability check for ideas 2 and 9: search Hugging Face Datasets (huggingface.co) and Kaggle (kaggle.com) before assuming that the data must be collected from scratch.

</details>

**Definition of done:** A public repo exists whose first commit is a one-page `SPEC.md` with problem, users, data, success metric, and four weekly milestones, and the project passes the 3-phases + demoable rule.

## Project 2 — Build in weekly milestones

**Goal:** Build the project one weekly milestone at a time, with a working end-to-end skeleton by the end of week 2: crude first, deep later.

**Milestones**

- [ ] **Week 1: Data and baseline.** Get the real data onto disk, explore it with pandas exactly as in [lesson 06](../phase-0-foundations/06-pandas.md), split off a test set that stays untouched, and score the dumbest possible baseline on the success metric. Checkpoint: a number is written at the top of the README draft, e.g. "baseline: 0.31".
- [ ] **Week 2: End-to-end skeleton.** Wire the whole pipeline: data in → model (even a bad small one) → output → minimal demo (a bare Gradio page or FastAPI endpoint counts). Every piece can be crude; no piece may be missing. Checkpoint: the builder (or a friend) can open a URL or run one command and get a real prediction from real data.
- [ ] **Mid-project review.** Stop and check honestly: (1) skeleton runs end to end, (2) current metric vs. baseline vs. target written down, (3) riskiest remaining piece named, (4) scope still finishable in two weeks (if not, cut features now and say so later in the README; cutting scope mid-project is a professional skill, not a failure). Checkpoint: the biggest remaining risk can be stated in one sentence.
- [ ] **Week 3: Depth.** Attack the weakest link only: better model, better features, fine-tuning, hyperparameter search, whichever the metric says matters most. Track experiments as in [lesson 41](41-mlops-ship-your-models.md). Checkpoint: the metric improved over week 2, and the reason can be stated in one sentence per change.
- [ ] **Week 4: Deploy and harden.** Ship the demo properly: Hugging Face Spaces for a UI (a free account is needed, and the Space's `README.md` needs the line `python_version: "3.13"`, as in lesson 27), or the lesson-41 container for an API. That container only runs on localhost; to make it public, create a Space with the **Docker** SDK, push the `Dockerfile`, the service code, the model (through Git LFS, as in lesson 27) and `requirements.txt` to it, and add `app_port: 8000` to the settings block at the top of the Space's `README.md`, because Docker Spaces expect port 7860 unless told otherwise. Handle the three ugliest inputs that come to mind (empty input, wrong file type, absurd values), and ask two real people to break it. Checkpoint: the public link works from a phone that has never seen the code.

<details><summary>Hints</summary>

- Breadth before depth is the core strategy. A crude skeleton in week 2 shows where the real difficulty is; four weeks of polishing one component shows nothing until it is too late.
- Keep a `LOG.md` and spend five minutes a day on it: date, what was tried, what happened, next step. The week-2 notes pay off in week 4, and the log becomes the "lessons learned" section for free.
- Stuck choosing between improvements in week 3? Do the error analysis from [lesson 20](../phase-2-classical-ml/20-end-to-end-ml-project.md): look at 30 concrete mistakes the model makes and count the categories. The biggest category is the week-3 plan.
- If the deploy is a struggle, deploy the crude version first and iterate. A live mediocre demo beats a local brilliant one.

</details>

**Definition of done:** The success metric beats the baseline meaningfully, the mid-project review is checked off, and a stranger with the link can use the demo end to end.

## Project 3 — Ship and tell

**Goal:** Turn the working project into a public credential: a blog-style README, an honest results story, and a post where real people will see it.

**Milestones**

- [ ] Write the blog-style README with exactly these sections: **Problem** (why anyone should care, 1 paragraph), **Data** (what, where from, how much, what was messy), **Approach** (what was built and *why*, including the dead ends), **Results** (a small table: baseline vs. final on the metric, plus one honest plot made with the skills from [lesson 07](../phase-0-foundations/07-data-visualization.md)), **Demo** (the live link + a GIF or screenshot), **Limitations**, **Lessons learned**. Checkpoint: a friend who knows no ML can read it and explain back what was built and how well it works.
- [ ] Write the **three honest limitations**: real ones ("only trained on daytime photos, fails at night"), not humble-brags ("could be even more accurate"). Then add a short **"With 10x time I would..."** paragraph. It shows reviewers that the author sees the road ahead, which reads as expertise.
- [ ] Clean the repo as if someone's hiring decision depends on it (it might): `README.md`, `requirements.txt`, `.gitignore`, a `src/` folder, no dead files, no notebooks named `Untitled3.ipynb`, and a one-command way to run it locally.
- [ ] Post it somewhere real: LinkedIn or X with the demo link and one result number, a lightning talk at a local Python/ML meetup, or a relevant community. Keep it to one genuine paragraph: what was built, one number, one limitation, the link. Checkpoint: at least one stranger has clicked the demo.
- [ ] **Milestone: the credential.** Pin the repo on the GitHub profile and put the link in the profile bio/CV. The README and the repo together are the credential: they prove in public, with running code, that their author went from zero to shipping ML systems. No certificate says that more clearly.

<details><summary>Hints</summary>

- Write the Results section first: it forces honesty, and the rest of the README organizes itself around it.
- Record the demo GIF early; it is the single highest-value 30 minutes of this whole step. A short screen recording + an online GIF converter works fine, and every system has a screen recorder built in:
  - **macOS:** press Cmd+Shift+5, choose to record the entire screen or a selected portion, and click **Record**; to stop, click the stop button in the menu bar. The video is saved on the Desktop.
  - **Windows (WSL2):** in Snipping Tool, select **Record**, then **New** (or press Win+Shift+R), select the area to record and select **Start**; to stop, select **Stop**, then save the video. Windows 10's Snipping Tool cannot record; there, switch to the browser and press Win+Alt+R to start and stop an Xbox Game Bar recording, which is saved in Videos → Captures.
  - **Linux:** on Ubuntu, press Ctrl+Shift+Alt+R, choose **Selection** or **Screen** and click the big round red button; to stop, click the red indicator in the top bar. The video is saved in `~/Videos/Screencasts`.
- Fear of posting is normal and nearly universal. Post anyway. The realistic worst case is silence, and the best case is a new opportunity.

</details>

**Definition of done:** A public repo with the full blog-style README, a working demo link, three honest limitations, a 10x-time paragraph, and a post about it live somewhere real.

## Project 4 — What next: a map of the territory

**Goal:** All the fundamentals are now in place; what remains is direction. Chart the next 12 months deliberately instead of drifting.

**The specialization map** lists the main roads from here, and what "go deeper" means on each:

- **LLM engineering**: evals (systematic testing of AI systems, first practiced in [lesson 34](../phase-5-llms/34-llm-apis-prompting.md) and applied to agents in [lesson 37](../phase-5-llms/37-ai-agents-tool-use.md)), serving at scale, retrieval quality, agent reliability. Builds on phases 4-5. The fastest-moving road, and the one where GPT-from-scratch knowledge is an unusual strength.
- **Computer vision**: detection, segmentation, video, 3D. Builds on phase 3. Do Stanford's CS231n course for depth here.
- **Reinforcement learning**: robotics, game AI, and RLHF (training language models from human feedback, the technique behind modern chat assistants), where the phase 4 and [lesson 39](../phase-6-special-topics/39-reinforcement-learning.md) threads meet. The most mathematical road.
- **ML engineering / MLOps at scale**: data pipelines, distributed training, serving at millions of requests. Builds on phase 7 plus general software engineering. The most employable road.
- **Research**: reading and reproducing papers. Start with the **annotated-paper method**: print or open a paper, and do not turn a page until every equation can be restated in plain words and every architecture in tensor shapes. The skills from [lesson 31](../phase-4-nlp-transformers/31-attention-build-gpt.md) are exactly this. Then reproduce one small paper result; Hugging Face Papers (huggingface.co/papers) links many papers to their code repositories, so a reproduction can be checked against them.

**Courses worth doing now** (the background to get their full value is finally in place): fast.ai (top-down and projects-first, so it will feel familiar), CS231n (vision depth), CS224n (Stanford's NLP course; search for "Stanford CS224n", and expect to know a surprising amount of it already), and d2l.ai (an interactive deep-learning book with runnable code, ideal as a reference to fill gaps).

**Milestones**

- [ ] Write `work/42-capstone/NEXT.md`: pick *one* primary specialization (it can change later; drifting between all five is the only wrong answer) and write a 12-month plan of quarterly projects, because the habit that led to this point is now clear: **learn by building, forever**. Checkpoint: every quarter's entry names a concrete artifact, not a topic ("build X", never "study Y").
- [ ] Pick one paper connected to the capstone from Hugging Face Papers and annotate it with the method above. Checkpoint: its core idea can be explained to a friend in two minutes without opening the paper.
- [ ] Join one community and actually participate once: answer a beginner's question (10 months ago, the roles were reversed), enter a Kaggle competition, or show the capstone at a meetup.
- [ ] Check off lesson 42 and the final milestone in [PROGRESS.md](../PROGRESS.md), so that every required lesson is ticked (Phase 6 needs only one of its three). Read the commit history from lesson 01 onward. It is not a tutorial trail; it is a portfolio.

<details><summary>Hints</summary>

- Picking a specialization feels momentous, but it is not. Six months of building in any one direction teaches more than a year of comparing directions.
- On papers: everyone reads them slowly at first. Hours for one paper is normal, even for professionals. Speed comes from repetition, not talent.

</details>

**Definition of done:** `NEXT.md` exists with one chosen specialization and four quarterly build-projects, one paper is annotated, and one community has seen a real contribution.

## Stretch goals

- **Reproduce a paper** end to end and publish the implementation with a "what the paper didn't tell you" section. This is the classic entry ticket to research credibility.
- **Get real users:** find 10 people who use the capstone demo, collect their feedback, and ship one improvement they asked for. Nothing teaches like users.
- **Write the journey post:** "Zero to ML in 10 months: what I built", an artifact-by-artifact retrospective from first script to capstone. Posts like this help hundreds of people and tend to travel far.
- **Enter a Kaggle competition** in the chosen specialization and finish in the top half. The lesson 20 pipeline plus everything since gives a genuine shot at it.

## Getting unstuck

- **Stuck choosing (week 1):** the choice is a month of work, not a career. Flip a coin between the top two ideas and start; commitment beats optimization here.
- **The week-3 slump** is real: the skeleton works, the shine is gone, and the remaining work is grind. This is precisely where most public projects die, which is exactly why finishing one is a credential. Re-read the spec's Problem section, cut scope if needed, and ship the next smallest visible improvement.
- **Metric won't budge:** return to error analysis (30 concrete failures, categorized, biggest category first). Panic generalizes; error analysis localizes.
- Standing advice, one last time: read error messages bottom-up, print shapes and values before theorizing, ask an AI assistant for a *hint* (or a spec critique) rather than a solution, and type all code by hand. These four habits worked for 42 lessons, and they keep working for everything that comes next.

## Resources

- [fast.ai](https://course.fast.ai) — the projects-first deep learning course to take now that the fundamentals are in place; excellent for breadth and modern practice.
- [Dive into Deep Learning](https://d2l.ai) — free interactive book with runnable code; the reference for filling theory gaps as they appear.
- [CS231n](https://cs231n.stanford.edu) — Stanford's computer vision course; the definitive next step for the vision road.
- [Hugging Face Papers](https://huggingface.co/papers/trending) — research papers linked to their code; where to pick a first paper to annotate and reproduce. (It replaced Papers with Code, which closed in 2025.)
- [Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — rewatch any of it now and notice how much more of it makes sense; the model for how this curriculum teaches.

## Skills unlocked

- [ ] I can scope an ML project so it is ambitious enough to matter and small enough to finish.
- [ ] I can write a one-page spec with a measurable success metric before writing code.
- [ ] I can plan and hit weekly milestones, and cut scope deliberately when reality demands it.
- [ ] I can build an ML system end to end (data, model, evaluation, API, demo) without a tutorial.
- [ ] I can write a results-honest, blog-style README that a non-expert can follow.
- [ ] I can ship work publicly and talk about it, limitations included.
- [ ] I can read a research paper with the annotated-paper method and chart my own learning from here.
- [ ] I can prove with a link that I went from zero to hero.

## The journey continues

There is no lesson 43, because from here on the lessons are self-chosen projects. [PROGRESS.md](../PROGRESS.md) now reads as a map of the path so far, and `NEXT.md` as the map of the path ahead. The course began ten months ago, before a single line of Python had been written. It ends with a public repo, under the learner's own name, serving a self-built model, and with a GPT written from scratch in the commit history. Keep the habit that did all of this: pick something slightly too hard, build it, ship it, tell people, repeat. Learn by building, forever.
