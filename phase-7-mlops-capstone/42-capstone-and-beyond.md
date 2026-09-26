# 42 · Capstone: Your Hero Project (and What Comes Next)

**Phase 7 — MLOps & Capstone** · Estimated time: 4-6 weeks · Prerequisites: [41 · MLOps: Track, Serve, Containerize, Deploy](41-mlops-ship-your-models.md) — and, honestly, everything before it.

> Stop for a second and look back. In [lesson 01](../phase-0-foundations/01-environment-setup.md) you wrote your first Python script and it printed a sentence. Since then you have built a gradient descent optimizer from a math formula, a k-NN classifier from scratch, a spam filter, a Kaggle pipeline, micrograd, a neural network in pure NumPy that reads handwritten digits, a CNN that sees, a GPT you built layer by layer and [trained yourself](../phase-4-nlp-transformers/33-train-your-own-gpt.md), a RAG app that chats with your documents, an AI agent, and a containerized model running behind a real API. This final lesson is where you stop following lessons and build ONE substantial project that is entirely yours — scoped, specced, built, deployed, and written up so well that it becomes your proof of the whole journey. This is the graduation project. Make it count.

## What you will build

- **Project 1 — Choose and spec:** a one-page project spec (problem, users, data, success metric, four weekly milestones) for a hero project that combines skills from at least three phases.
- **Project 2 — Build:** the capstone itself, built in weekly milestones with a working end-to-end skeleton by the end of week 2.
- **Project 3 — Ship and tell:** a public GitHub repo with a blog-style README, a live demo link, and a post about it somewhere real. This README + repo IS your credential.
- **Project 4 — What next:** your personal 12-month specialization roadmap, plus one annotated research paper to start the next chapter.

## Concepts you will learn by doing

- **Scoping** — choosing a project ambitious enough to be interesting but small enough to actually finish.
- **The one-page spec** — writing down problem, users, data, and success metric BEFORE writing any code.
- **Weekly milestone planning** — breaking a month of work into four checkable deliverables.
- **Breadth before depth** — getting a crude end-to-end version working early, then improving pieces.
- **The blog-style README** — problem, data, approach, results, demo, limitations, lessons: the write-up that turns code into a credential.
- **Sharing** — putting work in front of real people: GitHub, Hugging Face Spaces, LinkedIn/X, local meetups.
- **The specialization map** — the honest lay of the land after the fundamentals: which directions exist and how to pick one.

## Before you start

Prerequisites check — you should be able to say yes to all of these:

- You finished [lesson 41](41-mlops-ship-your-models.md) and have deployed at least one model behind an API.
- You have a public demo from [lesson 27](../phase-3-deep-learning/27-transfer-learning-vision-project.md) (Hugging Face Spaces) and know how to put another one up.
- You can start a project from an empty folder without a lesson telling you what to type. (That is exactly what you are about to prove.)

Set up your workspace:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
mkdir -p work/42-capstone
cd work/42-capstone
```

There is no fixed pip install list this time — your dependencies depend on the project you choose. Install what you need as you need it (inside the venv), and record every install in a `requirements.txt` from day one:

```bash
pip install <whatever-you-need>
pip freeze > requirements.txt
```

One warning before you begin: the capstone will feel different from every lesson so far, because nobody hands you milestones — you write them. That discomfort is the point. It is what working on real ML projects feels like.

## Project 1 — Choose and spec

**Goal:** Pick one project idea, pressure-test it, and write a one-page spec with a success metric and four weekly milestones — before writing a single line of project code.

**The rule:** your capstone must (1) combine skills from at least **three phases** of this curriculum, and (2) end in a **deployed or demoable artifact** a stranger can try. No exceptions. A notebook on your laptop is not a hero project.

**The idea list** (steal one, remix one, or bring your own — the phases each idea draws on are in brackets):

1. **Personal study assistant** — RAG over your own notes from this curriculum, a fine-tuned open model for the tone you want, and agent tools like "quiz me on lesson N" [phases 0, 4, 5].
2. **Local plant or bird identifier** — transfer-learning CNN on photos of species from your region, deployed as a public demo where anyone can upload a photo [phases 0, 3, 7].
3. **End-to-end tabular product** — a real prediction service (rent estimator, crop-yield or air-quality predictor for your city) with feature pipeline, API, monitoring dashboard, and scheduled retraining [phases 2, 0, 7].
4. **Niche-corpus GPT** — your from-scratch GPT trained on a corpus you love (song lyrics, cricket commentary, old recipes, legal boilerplate) with a small web UI for generation [phases 3, 4, 7].
5. **RL game agent with replay viewer** — train an agent on a simple game and build a web page that replays its episodes so visitors can watch it learn [phases 3, 6, 7].
6. **"What should I watch tonight?"** — a recommender trained on MovieLens plus your own ratings, served behind an API with a simple front end [phases 2, 6, 7].
7. **Semantic search for a hobby** — embeddings-based search over a niche corpus (climbing routes, board-game rules, ghazals) with an evaluation set you build yourself [phases 4, 5, 7].
8. **Doodle recognition game** — a browser game where people draw and your CNN guesses, with a leaderboard [phases 3, 0, 7].
9. **Audio classifier** — bird songs or music genres via spectrograms (pictures of sound) + CNN, with a record-and-classify demo [phases 0, 3, 7].
10. **Repo agent** — an AI agent that reviews your own GitHub commits, suggests improvements, and is evaluated against a test set of known-bad commits you curate [phases 5, 2, 7].

**Milestones**

- [ ] Spend one day (no more!) shortlisting 2-3 ideas. For each, answer in writing: What data would I use, and can I get it TODAY without a signup wall? Who would actually try the demo? Which 3+ phases does it use? Checkpoint: any idea where you cannot name the data source concretely is eliminated.
- [ ] Run the "weekend test" on your favorite: could you build a laughably crude version in two days? If the honest answer is "not even close", shrink the idea (fewer classes, smaller corpus, one feature instead of five) until the answer is yes. Scoping means cutting until finishing is plausible.
- [ ] Write the one-page spec as `work/42-capstone/SPEC.md` with exactly these sections: **Problem** (2-3 sentences, no jargon), **Users** (who tries the demo and why they'd care), **Data** (source, size, how you get it), **Success metric** (one number and a target, e.g. "top-3 accuracy ≥ 0.85 on a held-out set of 200 photos I labeled" — you learned to pick metrics in [lesson 14](../phase-2-classical-ml/14-model-evaluation.md)), **Weekly milestones** (four, one line each, each ending in something checkable).
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

- [ ] Stress-test the spec: ask an AI assistant to play a skeptical reviewer and find the three biggest risks in your spec. Do not let it rewrite the spec — argue with it, then revise the spec yourself. Checkpoint: your week-2 milestone says some version of "crude end-to-end pipeline works".
- [ ] Create the project's own public GitHub repo (separate from this learning repo), commit `SPEC.md` as its first commit, and add the spec's problem statement as the repo description.

<details><summary>Hints</summary>

- The most common capstone failure is scope, not skill. When torn between two ideas, pick the one with the boring, available data. A finished small project beats an abandoned epic every time.
- A good success metric is one you can compute in week 1 with a dumb baseline (majority class, random recommender, tiny model). If you can't baseline it in week 1, the metric is too vague.
- "Users" can be five friends and a subreddit. Naming them changes your decisions — a demo for bird-watchers needs species names, not class indices.
- Data-availability check for ideas 2 and 9: search Hugging Face Datasets (huggingface.co) and Kaggle (kaggle.com) before assuming you must collect your own.

</details>

**Definition of done:** A public repo exists whose first commit is a one-page `SPEC.md` with problem, users, data, success metric, and four weekly milestones — and the project passes the 3-phases + demoable rule.

## Project 2 — Build in weekly milestones

**Goal:** Build the thing, one weekly milestone at a time, with a working end-to-end skeleton by the end of week 2 — crude first, deep later.

**Milestones**

- [ ] **Week 1 — Data and baseline.** Get the real data onto disk, explore it with pandas exactly as in [lesson 06](../phase-0-foundations/06-pandas.md), split off a test set you will not touch, and score the dumbest possible baseline on your success metric. Checkpoint: a number is written at the top of your README draft, e.g. "baseline: 0.31".
- [ ] **Week 2 — End-to-end skeleton.** Wire the whole pipeline: data in → model (even a bad small one) → output → minimal demo (a bare Gradio page or FastAPI endpoint counts). Every piece can be crude; no piece may be missing. Checkpoint: you (or a friend) can open a URL or run one command and get a real prediction from real data.
- [ ] **Mid-project review** — stop and check honestly: (1) skeleton runs end to end, (2) current metric vs. baseline vs. target written down, (3) riskiest remaining piece named, (4) scope still finishable in two weeks (if not, CUT features now, and say so later in the README — cutting scope mid-project is a professional skill, not a failure). Checkpoint: you can state in one sentence what the biggest risk left is.
- [ ] **Week 3 — Depth.** Attack the weakest link only: better model, better features, fine-tuning, hyperparameter search — whatever your metric says matters most. Track experiments the way you learned in [lesson 41](41-mlops-ship-your-models.md). Checkpoint: metric improved over week 2, and you can say why in one sentence per change.
- [ ] **Week 4 — Deploy and harden.** Ship the demo properly (Hugging Face Spaces for a UI — free account needed, as in lesson 27 — or your lesson-41 container recipe for an API), handle the three ugliest inputs you can think of (empty input, wrong file type, absurd values), and ask two real people to break it. Checkpoint: the public link works from a phone that has never seen your code.

<details><summary>Hints</summary>

- Breadth before depth is the whole game. A crude skeleton in week 2 tells you where the real difficulty is; four weeks of polishing one component tells you nothing until it's too late.
- Keep a `LOG.md` and spend five minutes a day on it: date, what you tried, what happened, next step. Week-4-you will thank week-2-you, and it becomes the "lessons learned" section for free.
- Stuck choosing between improvements in week 3? Do the error analysis from [lesson 20](../phase-2-classical-ml/20-end-to-end-ml-project.md): look at 30 concrete mistakes your model makes and count the categories. The biggest category is your week-3 plan.
- If the deploy fights you, deploy the crude version first and iterate. A live mediocre demo beats a local brilliant one.

</details>

**Definition of done:** The success metric beats your baseline meaningfully, the mid-project review is checked off, and a stranger with the link can use the demo end to end.

## Project 3 — Ship and tell

**Goal:** Turn the working project into a public credential: a blog-style README, an honest results story, and a post where real people will see it.

**Milestones**

- [ ] Write the blog-style README with exactly these sections: **Problem** (why anyone should care, 1 paragraph), **Data** (what, where from, how much, what was messy), **Approach** (what you built and WHY — including the dead ends), **Results** (a small table: baseline vs. final on your metric, plus one honest plot made with your [lesson 07](../phase-0-foundations/07-data-visualization.md) skills), **Demo** (the live link + a GIF or screenshot), **Limitations**, **Lessons learned**. Checkpoint: a friend who knows no ML can read it and explain back what you built and how well it works.
- [ ] Write the **three honest limitations** — real ones ("only trained on daytime photos, fails at night"), not humble-brags ("could be even more accurate"). Then add a short **"With 10x time I would..."** paragraph: it shows reviewers you see the road ahead, which reads as expertise.
- [ ] Clean the repo like someone's hiring depends on it (it might): `README.md`, `requirements.txt`, a `src/` folder, no dead files, no notebooks named `Untitled3.ipynb`, and a one-command way to run it locally.
- [ ] Post it somewhere real — LinkedIn or X with the demo link and one result number, a local Python/ML meetup lightning talk, or a relevant community. One genuine paragraph: what you built, one number, one limitation, the link. Checkpoint: at least one person you have never met has clicked your demo.
- [ ] **MILESTONE — the credential.** Pin the repo on your GitHub profile and put the link in your profile bio/CV. This README + repo IS your credential: it proves, in public, with running code, that you went from zero to shipping ML systems. No certificate says that louder.

<details><summary>Hints</summary>

- Write the Results section first — it forces honesty and the rest of the README organizes itself around it.
- Record the demo GIF early; it is the single highest-value 30 minutes of this whole step. On Ubuntu, `peek` or a screen recording + an online GIF converter works fine.
- Fear of posting is normal and nearly universal. Post anyway. The realistic worst case is silence, and the best case is your next opportunity.

</details>

**Definition of done:** A public repo with the full blog-style README, a working demo link, three honest limitations, a 10x-time paragraph — and a post about it live somewhere real.

## Project 4 — What next: your map of the territory

**Goal:** You now hold all the fundamentals; what remains is direction. Chart your next 12 months deliberately instead of drifting.

**The specialization map** — the main roads from here, and what "go deeper" means on each:

- **LLM engineering** — evals (systematic testing of AI systems — you started this in [lesson 37](../phase-5-llms/37-ai-agents-tool-use.md)), serving at scale, retrieval quality, agent reliability. Builds on phases 4-5. The fastest-moving road, and the one your GPT-from-scratch knowledge makes you unusually strong on.
- **Computer vision** — detection, segmentation, video, 3D. Builds on phase 3. Do Stanford's CS231n course for depth here.
- **Reinforcement learning** — robotics, game AI, and RLHF (training language models from human feedback — the technique behind modern chat assistants), where your phase 4 and [lesson 39](../phase-6-special-topics/39-reinforcement-learning.md) threads meet. The most mathematical road.
- **ML engineering / MLOps at scale** — data pipelines, distributed training, serving at millions of requests. Builds on phase 7 plus general software engineering. The most employable road.
- **Research** — reading and reproducing papers. Start with the **annotated-paper method**: print or open a paper, and refuse to turn a page until you can restate every equation in plain words and every architecture in tensor shapes — the skills from [lesson 31](../phase-4-nlp-transformers/31-attention-build-gpt.md) are exactly this. Then reproduce one small paper result; Papers with Code links papers to their implementations so you can check yourself.

**Courses worth doing NOW** (you finally have the background to get full value from them): fast.ai (top-down, projects-first — will feel comfortingly familiar), CS231n (vision depth), CS224n (Stanford's NLP course — search for "Stanford CS224n"; you will be amazed how much you already know), and d2l.ai (an interactive deep-learning book with runnable code, ideal as a reference to fill gaps).

**Milestones**

- [ ] Write `work/42-capstone/NEXT.md`: pick ONE primary specialization (you can change later; drifting between all five is the only wrong answer) and write a 12-month plan of quarterly projects — because you now know the habit that got you here: **learn by building, forever**. Checkpoint: every quarter's entry names a concrete artifact, not a topic ("build X", never "study Y").
- [ ] Pick one paper connected to your capstone from Papers with Code and annotate it with the method above. Checkpoint: you can explain its core idea to a friend in two minutes without opening the paper.
- [ ] Join one community and actually participate once: answer a beginner's question (you were them 10 months ago), enter a Kaggle competition, or show your capstone at a meetup.
- [ ] Update the main [README](../README.md) progress tracker: 42 of 42. Read your own commit history from lesson 01. That's not a tutorial trail — that's a portfolio.

<details><summary>Hints</summary>

- Picking a specialization feels momentous; it isn't. Six months of building in any one direction teaches you more than a year of comparing directions.
- On papers: everyone reads them slowly at first — hours for one paper is normal, even for professionals. Speed comes from repetition, not talent.

</details>

**Definition of done:** `NEXT.md` exists with one chosen specialization and four quarterly build-projects, one paper is annotated, and you have shown up in one community.

## Stretch goals

- **Reproduce a paper** end to end and publish your implementation with a "what the paper didn't tell you" section — the classic entry ticket to research credibility.
- **Get real users:** find 10 people who use your capstone demo, collect their feedback, and ship one improvement they asked for. Nothing teaches like users.
- **Write the journey post:** "Zero to ML in 10 months: what I built" — an artifact-by-artifact retrospective from first script to capstone. These posts help hundreds of people and tend to travel far.
- **Enter a Kaggle competition** in your specialization and finish in the top half — your lesson 20 pipeline plus everything since gives you a genuine shot.

## If you get stuck

- **Stuck choosing (week 1):** you are not choosing a career, you are choosing a month. Flip a coin between your top two and start; commitment beats optimization here.
- **The week-3 slump** is real: the skeleton works, the shine is gone, and the remaining work is grind. This is precisely where most public projects die, which is exactly why finishing one is a credential. Re-read your spec's Problem section, cut scope if needed, and ship the next smallest visible improvement.
- **Metric won't budge:** return to error analysis — 30 concrete failures, categorized, biggest category first. Panic generalizes; error analysis localizes.
- Standing advice, one last time: read error messages bottom-up, print shapes and values before theorizing, ask an AI assistant for a HINT (or a spec critique) rather than a solution, and type all code yourself. These four habits carried you through 42 lessons. They will carry you the rest of the way too.

## Resources

- [fast.ai](https://course.fast.ai) — the projects-first deep learning course to take NOW that you have fundamentals; excellent for breadth and modern practice.
- [Dive into Deep Learning](https://d2l.ai) — free interactive book with runnable code; the reference for filling theory gaps as they appear.
- [CS231n](https://cs231n.stanford.edu) — Stanford's computer vision course; the definitive next step for the vision road.
- [Papers with Code](https://paperswithcode.com) — papers linked to implementations; where to pick your first paper to annotate and reproduce.
- [Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — rewatch any of it now and notice how much more you see; the model for how this curriculum taught you.

## Skills unlocked

- [ ] I can scope an ML project so it is ambitious enough to matter and small enough to finish.
- [ ] I can write a one-page spec with a measurable success metric before writing code.
- [ ] I can plan and hit weekly milestones, and cut scope deliberately when reality demands it.
- [ ] I can build an ML system end to end — data, model, evaluation, API, demo — without a tutorial.
- [ ] I can write a results-honest, blog-style README that a non-expert can follow.
- [ ] I can ship work publicly and talk about it, limitations included.
- [ ] I can read a research paper with the annotated-paper method and chart my own learning from here.
- [ ] I went from zero to hero — and I can prove it with a link.

## The journey continues

There is no lesson 43, because from here the lessons are the projects you choose yourself — the [main README](../README.md) now reads as a map of where you have been, and your `NEXT.md` as the map of where you are going. Ten months ago you had never written a line of Python; today there is a public repo with your name on it serving a model you built, and a GPT in your commit history that you wrote from scratch. Keep the habit that did all of this: pick something slightly too hard, build it, ship it, tell people, repeat. Learn by building, forever. Congratulations, hero.
