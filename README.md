# ML Zero to Hero

**A project-based curriculum that takes you from zero programming knowledge to building GPTs, RAG systems, and AI agents — by building small projects**

Inspired by Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — the deep learning phases follow his videos directly, and the *entire* curriculum follows his philosophy: **you understand what you build from scratch.**

---

## How this works

- **42 lessons in 8 phases.** Each lesson is one markdown file built around 2–3 small projects with milestone checklists, verifiable checkpoints, hints, and a "skills unlocked" self-test.
- **You build everything.** Concepts are explained inside the projects, exactly when you need them. No lesson asks you to read a textbook chapter first.
- **From scratch first, framework second.** You'll write your own kNN, linear regression, decision tree, autograd engine, neural network, tokenizer, and GPT — *then* use scikit-learn and PyTorch, understanding exactly what they automate.
- **Your code lives in `work/`.** For each lesson, create `work/<NN>-<slug>/` (e.g. `work/12-linear-regression/`) and build there. The lesson files guide you; the `work/` folders are your portfolio.
- **Track yourself in [PROGRESS.md](PROGRESS.md).** Check off lessons as you complete them. Commit and push your work after every session — your git history becomes the story of your journey.

---

## The map

### Phase 0 — Foundations: Python, NumPy, Pandas *(≈ 5–6 weeks)*
Learn programming from absolute zero by building games and analyzing real data.

| # | Lesson | Time |
|---|--------|------|
| 01 | [Set Up Your ML Workshop](phase-0-foundations/01-environment-setup.md) | 2–4 days |
| 02 | [Python Basics by Building Games](phase-0-foundations/02-python-basics.md) | 1 week |
| 03 | [Data Structures: Lists, Dicts and Real Text](phase-0-foundations/03-python-data-structures.md) | 1 week |
| 04 | [Classes, Files, JSON and Errors](phase-0-foundations/04-python-oop-files-errors.md) | 1 week |
| 05 | [NumPy: Thinking in Arrays](phase-0-foundations/05-numpy.md) | 1 week |
| 06 | [Pandas: Interrogating Real Datasets](phase-0-foundations/06-pandas.md) | 1 week |
| 07 | [Data Visualization: Charts That Answer Questions](phase-0-foundations/07-data-visualization.md) | 4–5 days |

### Phase 1 — Math by Building *(≈ 3–4 weeks)*
Just the math ML actually uses — learned by coding it, visualizing it, and simulating it. No proofs, no drills.

| # | Lesson | Time |
|---|--------|------|
| 08 | [Linear Algebra by Writing Your Own](phase-1-math/08-linear-algebra-by-code.md) | 1–2 weeks |
| 09 | [Calculus You Can Run: Gradient Descent](phase-1-math/09-calculus-and-gradient-descent.md) | 1 week |
| 10 | [Probability and Statistics by Simulation](phase-1-math/10-probability-statistics.md) | 1–2 weeks |

### Phase 2 — Classical Machine Learning *(≈ 10–12 weeks)*
Build every classic algorithm from scratch, then wield scikit-learn like a pro. Ends with a real Kaggle competition.

| # | Lesson | Time |
|---|--------|------|
| 11 | [Your First ML Model: k-Nearest Neighbors from Scratch](phase-2-classical-ml/11-first-model-knn.md) | 1 week |
| 12 | [Linear Regression from Scratch](phase-2-classical-ml/12-linear-regression.md) | 1 week |
| 13 | [Logistic Regression from Scratch](phase-2-classical-ml/13-logistic-regression.md) | 1 week |
| 14 | [Model Evaluation: Build Your Own Metrics Library](phase-2-classical-ml/14-model-evaluation.md) | 1 week |
| 15 | [Decision Trees from Scratch](phase-2-classical-ml/15-decision-trees.md) | 1–2 weeks |
| 16 | [Ensembles: Random Forests and Gradient Boosting](phase-2-classical-ml/16-ensembles-random-forest-boosting.md) | 1–2 weeks |
| 17 | [Text Classification: Naive Bayes Spam Filter (+ SVM)](phase-2-classical-ml/17-naive-bayes-spam-svm.md) | 1 week |
| 18 | [Unsupervised Learning: k-Means and PCA from Scratch](phase-2-classical-ml/18-unsupervised-kmeans-pca.md) | 1–2 weeks |
| 19 | [Feature Engineering and Pipelines](phase-2-classical-ml/19-feature-engineering-pipelines.md) | 1 week |
| 20 | [Capstone: End-to-End ML Project (Kaggle)](phase-2-classical-ml/20-end-to-end-ml-project.md) | 2 weeks |

**Milestone:** After lesson 20, you are a working junior ML practitioner.

### Phase 3 — Deep Learning *(≈ 8–10 weeks)* 
Build backpropagation from scratch (micrograd), a neural net in raw NumPy, then master PyTorch and computer vision.

| # | Lesson | Time |
|---|--------|------|
| 21 | [Neural Networks: The Intuition (Perceptron and XOR)](phase-3-deep-learning/21-neural-networks-intuition.md) | 4–5 days |
| 22 | [Build micrograd: Backpropagation from Scratch](phase-3-deep-learning/22-micrograd-backpropagation.md) (K) | 1–2 weeks |
| 23 | [Neural Network in Pure NumPy: MNIST Digits](phase-3-deep-learning/23-mlp-numpy-mnist.md) | 1–2 weeks |
| 24 | [PyTorch Fundamentals](phase-3-deep-learning/24-pytorch-fundamentals.md) | 1 week |
| 25 | [The Dark Arts of Training Deep Networks](phase-3-deep-learning/25-training-deep-nets.md) (K) | 1–2 weeks |
| 26 | [Convolutional Neural Networks: Teaching Machines to See](phase-3-deep-learning/26-cnns-computer-vision.md) | 2 weeks |
| 27 | [Transfer Learning: A Custom Image Classifier + Demo](phase-3-deep-learning/27-transfer-learning-vision-project.md) | 1 week |

**Milestone:** after lesson 27 you have *shipped* a working AI product with a public demo.

### Phase 4 — NLP and Transformers *(≈ 8–9 weeks)*
From counting letter pairs to building and training your own GPT.

| # | Lesson | Time |
|---|--------|------|
| 28 | [Language Models 101: makemore](phase-4-nlp-transformers/28-language-models-makemore.md) (K) | 1–2 weeks |
| 29 | [RNNs and LSTMs: Networks with Memory](phase-4-nlp-transformers/29-rnn-lstm-char-generation.md) | 1–2 weeks |
| 30 | [Embeddings: Meaning as Geometry (word2vec from Scratch)](phase-4-nlp-transformers/30-embeddings-word2vec.md) | 1 week |
| 31 | [Attention and the Transformer: Let's Build GPT](phase-4-nlp-transformers/31-attention-build-gpt.md) (K) | 2 weeks |
| 32 | [Tokenization: Build the GPT Tokenizer](phase-4-nlp-transformers/32-tokenization-bpe.md) (K) | 1 week |
| 33 | [Train Your Own GPT (nanoGPT)](phase-4-nlp-transformers/33-train-your-own-gpt.md) (K) | 2 weeks |

**Milestone:** After lesson 33, you understand, end to end, how the engines of modern AI are built — because you built one.

### Phase 5 — Modern AI: LLMs in Practice *(≈ 5–7 weeks)*
Build *with* large models: APIs and prompting, RAG, fine-tuning with LoRA, and AI agents.

| # | Lesson | Time |
|---|--------|------|
| 34 | [Using LLMs: APIs, Prompting and Structured Output](phase-5-llms/34-llm-apis-prompting.md) | 1 week |
| 35 | [RAG: Build "Chat With My Documents"](phase-5-llms/35-rag-chat-with-your-docs.md) | 1–2 weeks |
| 36 | [Fine-Tuning Open Models with LoRA](phase-5-llms/36-fine-tuning-open-models.md) | 2 weeks |
| 37 | [AI Agents: Tool Use, Loops and Evaluation](phase-5-llms/37-ai-agents-tool-use.md) | 1–2 weeks |

### Phase 6 — Special Topics *(pick what excites you, ≈ 2–3 weeks each)*
Optional deep dives. Do at least one; do all three if you're having fun.

| # | Lesson | Time |
|---|--------|------|
| 38 | [Generative Models: Autoencoders, GANs and Diffusion](phase-6-special-topics/38-generative-models-images.md) | 2–3 weeks |
| 39 | [Reinforcement Learning: Agents That Learn from Reward](phase-6-special-topics/39-reinforcement-learning.md) | 2–3 weeks |
| 40 | [Recommender Systems: Matrix Factorization on MovieLens](phase-6-special-topics/40-recommender-systems.md) | 1–2 weeks |

### Phase 7 — MLOps and Capstone *(≈ 6–8 weeks)*
Ship models like an engineer, then build the hero project that proves the whole journey.

| # | Lesson | Time |
|---|--------|------|
| 41 | [MLOps: Track, Serve, Containerize, Deploy](phase-7-mlops-capstone/41-mlops-ship-your-models.md) | 2 weeks |
| 42 | [Capstone: Your Hero Project (and What Comes Next)](phase-7-mlops-capstone/42-capstone-and-beyond.md) | 4–6 weeks |

**Final milestone:** a shipped, original AI project with a public demo and a write-up — your credential.

---



---

*Start here → [Lesson 01: Set Up Your ML Workshop](phase-0-foundations/01-environment-setup.md)*
