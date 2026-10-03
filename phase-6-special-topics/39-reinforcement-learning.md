# 39 · Reinforcement Learning: Agents That Learn from Reward

**Phase 6 — Special Topics** · Estimated time: 2-3 weeks · Prerequisites: [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](../phase-3-deep-learning/25-training-deep-nets.md), [10 · Probability and Statistics by Simulation](../phase-1-math/10-probability-statistics.md)

> Everything you have trained so far learned from labeled examples: here is the input, here is the right answer. Reinforcement learning (RL) is different — nobody tells the agent the right answer. It just acts, gets a number back (reward), and has to figure out what to do from that alone. This is how AlphaGo beat the world champion, and its cousin RLHF is how ChatGPT-style models learn to be helpful. In this lesson you will build an agent that learns to cross a frozen lake, a neural network that learns to balance a pole, and — most importantly — you will trick your own agent into cheating, and see with your own eyes why "AI alignment" is a real engineering problem, not science fiction.

*This is an optional-track lesson: you can skip to [40 · Recommender Systems](40-recommender-systems.md) and come back later. But if the phrase "the AI optimized the wrong thing" has ever intrigued you, this is the lesson where you get to watch it happen.*

## What you will build

- **Project 1:** a tabular Q-learning agent that solves slippery FrozenLake, plus a plot of its success rate over training and a printed map of its learned policy as arrows.
- **Project 2:** a DQN (Deep Q-Network) in PyTorch that solves CartPole (average reward 475+), with a replay buffer and target network you write yourself — and a rendered video of your agent balancing the pole.
- **Project 3:** a reward-hacking lab: a deliberately mis-specified reward, an agent that ruthlessly exploits it, and a short write-up connecting what you saw to RLHF and AI safety.

## Concepts you will learn by doing

- **The RL loop** — the agent sees a *state*, picks an *action*, gets a *reward* and a *next state*, repeat.
- **Exploration vs exploitation** — try new things vs do what already works; epsilon-greedy is the classic compromise.
- **Q-values** — a number for each (state, action) pair meaning "how much total reward can I expect if I do this here?"
- **The Bellman update** — the one-line formula that makes Q-values converge toward the truth.
- **Discount factor (gamma)** — how much the agent cares about future reward vs immediate reward.
- **DQN** — replacing the Q-table with a neural network, made stable by a *replay buffer* and a *target network*.
- **Reward shaping and reward hacking** — agents optimize what you measure, not what you mean.
- **Policy gradients and RLHF** — the other big family of RL, and how it connects to training LLMs.

## Before you start

You need PyTorch working (lesson 24) and comfort with training loops and loss curves (lesson 25). From the repo root, inside your venv:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium[toy-text,classic-control]" matplotlib numpy
mkdir -p work/39-reinforcement-learning
cd work/39-reinforcement-learning
```

As in lesson 24, torch comes from PyTorch's own package index; if lesson 24 already installed it (even a GPU build), that line leaves it as it is. That index has no Gymnasium or matplotlib, so the other packages get a plain `pip install` line of their own. On an Intel Mac the torch line fails, because current PyTorch has no Intel Mac version: Projects 1 and 3 need only the second line and run fine on your Mac, but do Project 2, and everything else that uses PyTorch, in Google Colab, as lesson 01 suggested.

**Gymnasium** is the standard library of RL environments (worlds for agents to act in) — the maintained successor of OpenAI Gym. No datasets to download: in RL the agent *generates* its own data by acting. Sanity-check the install:

```bash
python3 -c "import gymnasium as gym; env = gym.make('FrozenLake-v1'); print(env.observation_space, env.action_space)"
```

You should see `Discrete(16) Discrete(4)` — 16 states, 4 actions. That's the whole world for Project 1.

## Project 1 — Q-learning from scratch on FrozenLake

**Goal:** teach an agent to cross a slippery frozen lake using nothing but a 16×4 table of numbers and one line of update code. No neural networks — you will see the *essence* of RL with zero deep-learning machinery in the way.

FrozenLake: a 4×4 grid. Start top-left, reach the goal bottom-right, don't fall in the holes. Reward is 1 for reaching the goal, 0 for everything else. The catch: the ice is *slippery* — when you step, you sometimes slide sideways instead. Actions are 0=left, 1=down, 2=right, 3=up.

**Milestones**

- [ ] Write `random_agent.py`: make the environment and run 1000 episodes (an *episode* is one full game, from reset to done) with random actions. The core loop looks like this — this loop IS reinforcement learning's skeleton, memorize its shape:

  ```python
  import gymnasium as gym
  env = gym.make("FrozenLake-v1", is_slippery=True)
  state, info = env.reset()
  done = False
  while not done:
      action = env.action_space.sample()          # random for now
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
  ```

  Count how often the agent reaches the goal. Checkpoint: success rate is dismal — around 1-2%. This is your baseline; everything the agent learns is measured against dumb luck.
- [ ] Create the Q-table: `Q = np.zeros((16, 4))` — one row per state, one column per action. `Q[s, a]` will come to mean "expected total future reward if I take action `a` in state `s`". Right now the agent knows nothing, so all zeros is honest.
- [ ] Write `train_q.py` with an **epsilon-greedy** policy: with probability epsilon pick a random action (*exploration* — try things to discover what works), otherwise pick `np.argmax(Q[state])` (*exploitation* — use what you know). Start epsilon at 1.0 and decay it toward ~0.05 over training. Without exploration the agent can never discover the goal; without exploitation it never uses what it learned.
- [ ] After every step, apply the **Bellman update** — the single most important line in this lesson:

  ```python
  Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
  ```

  Read it out loud: nudge my estimate of `Q[s,a]` toward "the reward I just got, plus gamma times the best I believe I can do from where I landed". `alpha` is a learning rate (try 0.1), just like gradient descent in lesson 09. `gamma` is the **discount factor** (try 0.99): reward next step is worth gamma times reward now, so the agent prefers reaching the goal *sooner*. Note the beautiful trick: the update uses the agent's *own current guess* about the next state — estimates improving estimates, called *bootstrapping*.
- [ ] Train for 20,000+ episodes. Every 500 episodes, evaluate: run 100 episodes with epsilon=0 (pure exploitation, no learning) and record the success rate. Checkpoint: the success curve climbs from ~0 and settles above 0.70 — around 75% is near-optimal, because on slippery ice even a perfect policy sometimes slides into a hole.
- [ ] Plot success rate vs training episodes with matplotlib (lesson 07) and save it as `frozenlake_learning.png`. Checkpoint: the plot shows a clear S-shaped or steady climb, then a plateau.
- [ ] Visualize the learned policy: for each of the 16 states print the arrow for `np.argmax(Q[state])` (`←↓→↑`), laid out as the 4×4 grid, with `H` for holes and `G` for the goal. Checkpoint: the arrows mostly steer *away from holes* — on slippery ice some arrows look weird (pointing along a hole's edge rather than at the goal) and that is genuinely optimal. Stare at it until you see why.

<details><summary>Hints</summary>

- Success rate stuck at 0? Nearly always an epsilon problem: it decayed too fast, so the agent stopped exploring before *ever* reaching the goal (and reward is only at the goal). Decay slower, e.g. `epsilon = max(0.05, epsilon * 0.9999)` per episode.
- Getting ~60% but not 70%? Train longer and decay `alpha` too, or drop it to 0.05 late in training. Slippery FrozenLake is noisy; the last few percent take patience.
- Evaluate with a *separate* loop that does not update Q and uses epsilon=0. Mixing training and evaluation numbers makes the curve lie to you.
- Debug trick: `print(np.round(Q, 2))` — states near the goal should light up first, then values flow backward toward the start over training. If the whole table is still zeros after 1000 episodes, the agent has never reached the goal even once.

</details>

**Definition of done:** eval success rate above 70% on slippery FrozenLake, a saved learning-curve plot, and a printed arrow-policy you can explain out loud.

## Project 2 — DQN on CartPole

**Goal:** the Q-table dies the moment states stop being countable — CartPole's state is 4 continuous numbers (cart position, cart velocity, pole angle, pole angular velocity), so there are infinitely many states. Replace the table with a small PyTorch network that *takes a state and outputs 4→2 Q-values*, and stabilize it with the two tricks that made the 2013 DQN paper famous.

CartPole: push a cart left or right to keep a pole balanced. Reward is +1 per timestep the pole stays up, episode ends at 500 steps or when it falls. **Solved = average reward ≥ 475 over 100 consecutive episodes.**

**Milestones**

- [ ] Explore the environment: `gym.make("CartPole-v1")`, print `env.observation_space` and `env.action_space`, run a random agent for 20 episodes. Checkpoint: random gets an average reward around 20-30 — the pole falls almost immediately.
- [ ] Build the Q-network in PyTorch (lesson 24): input 4, two hidden layers of 128 with ReLU, output 2 (one Q-value per action). Acting is `q_net(state_tensor).argmax()` — same idea as `np.argmax(Q[state])`, the table just became a function.
- [ ] Build the **replay buffer**: a `collections.deque(maxlen=10000)` of `(state, action, reward, next_state, done)` tuples; training samples random minibatches from it. Why: consecutive steps are nearly identical, and gradient descent (lesson 25) assumes shuffled, roughly independent samples — learning from raw experience order makes the network chase its own tail.
- [ ] Build the **target network**: a frozen copy of the Q-network, used only to compute the Bellman target `r + gamma * target_net(s_next).max()`; copy the weights over every ~500 steps. Why: without it, the network computes its target *with the same weights it is updating*, like measuring a table with a ruler that shrinks every time you cut — a moving target that famously makes training diverge.
- [ ] Write the training step: sample a batch of 64, compute predicted `q_net(s)[a]` and target `r + gamma * max(target_net(s')) * (1 - done)`, take an MSE (or Huber) loss between them, backprop, step. That `(1 - done)` matters: after the episode ends there is no future reward. This is the Bellman update from Project 1 wearing a neural network costume.
- [ ] Train with epsilon-greedy (decay 1.0 → 0.05 over ~10k steps), Adam with lr around 1e-3, gamma 0.99. Log episode reward. Checkpoint: rewards are noisy and ugly for a while (RL curves are far messier than the supervised loss curves of lesson 25 — this is normal), then climb past 100, then 300.
- [ ] Track the average over the last 100 episodes; stop when it reaches **475+**. Checkpoint: solved, typically within 300-1000 episodes. Save the weights with `torch.save`.
- [ ] Victory lap: load the weights and run one episode with `gym.make("CartPole-v1", render_mode="human")` and epsilon=0. A window should pop up on macOS, on Windows (WSL2 shows it on your Windows desktop through WSLg) and on a Linux desktop; if none appears, or you are working in Google Colab, use `render_mode="rgb_array"`, collect `env.render()` frames, and save them as PNGs or an animation with matplotlib. Checkpoint: the pole just... stays up, for the full 500 steps. Enjoy this. You taught it that.

<details><summary>Hints</summary>

- Reward climbs then collapses to ~10? Classic DQN instability. Usual suspects in order: target network updated too often (or never), learning rate too high (try 5e-4), epsilon decayed before the buffer had variety. Also don't start training until the buffer holds ~1000 transitions.
- Shape bugs are the #1 silent killer: `q_net(states)` is `(64, 2)` and you need the Q-value *of the action actually taken* — `torch.gather` along dim 1 — not the max. Print every tensor shape in the loss computation once.
- Wrap target computation in `with torch.no_grad():` — gradients must flow only through the prediction, not the target.
- If nothing works, simplify back to something you can verify: does your net at least overfit a single fixed batch to near-zero loss? (Lesson 25's first rule of debugging.)

</details>

**Definition of done:** average reward ≥ 475 over 100 consecutive episodes, saved weights, and you watched a rendered episode of your trained agent.

## Project 3 — Reward hacking lab

**Goal:** deliberately mis-specify a reward and watch your own agent exploit the loophole with complete indifference to what you *meant*. This is the most important project in the lesson: everything RL learns is downstream of the reward, and reward is written by fallible humans.

**Milestones**

- [ ] Wrap FrozenLake so the agent gets **+0.01 per step survived** ("stay alive" sounds sensible, right?) on top of +1 at the goal. Gymnasium's `gym.Wrapper` makes this a ~10-line class overriding `step()`. Also set `gym.make(..., max_episode_steps=200)` so episodes can drag.
- [ ] Retrain your Project 1 Q-learner on the wrapped environment, unchanged. Watch its behavior: print average episode *length* and goal-reach rate. Checkpoint: goal-reach rate *drops* (often toward 0) while episode length climbs toward the 200-step cap — the agent learns to shuffle around safe squares forever, farming survival reward. It found `200 × 0.01 = 2 > 1`: loitering literally pays better than winning. It is not broken. It is doing exactly what you asked.
- [ ] Now try honest **reward shaping** — extra reward signals meant to *help*, e.g. small reward for reducing distance to the goal. Tune it: too big and the agent finds a new exploit (pacing back and forth near the goal if you reward approach but don't penalize retreat), small and symmetric and it genuinely learns faster than sparse reward. Checkpoint: at least one shaping scheme that helps and one that backfires, with the numbers to show it.
- [ ] Optional CartPole version: reward the pole for being *exactly* vertical (`+1` only when `abs(angle) < 0.01`) and watch DQN learn jittery micro-twitching instead of calm balance.
- [ ] Write `reward-hacking-writeup.md` (in your work folder — this one is *for you*, in your own words, ~1 page): what you changed, what the agent did, and the general law: **agents optimize what you MEASURE, not what you MEAN.** Then connect it outward — this is the *alignment problem* in miniature. **RLHF** (reinforcement learning from human feedback), the technique used to fine-tune LLMs into helpful assistants, is RL where the reward comes from a model trained on human preference ratings. Your FrozenLake agent farming step-reward and an LLM learning to sound confident because raters reward confident-sounding answers (even when wrong — "sycophancy") are the *same failure*, at very different scales. Every "the AI gamed the metric" story you have ever heard is Project 3 wearing a bigger budget.

<details><summary>Hints</summary>

- If the wrapped agent still reaches the goal, your exploit isn't profitable enough — raise the per-step reward or the step cap until loitering beats winning. Finding the break-even point *is* the lesson.
- Compare three numbers per experiment: total reward collected, goal-reach rate, episode length. Reward goes UP while goal-rate goes DOWN — that scissors pattern is the signature of reward hacking.
- One paragraph of theory worth knowing: DQN learns *values* and derives actions from them. **Policy-gradient** methods instead learn the policy directly — a network outputting action probabilities, nudged to make actions that led to high reward more likely. That family (REINFORCE → PPO) is what RLHF actually uses. You now know enough to read [Spinning Up](https://spinningup.openai.com)'s intro and recognize every term.

</details>

**Definition of done:** a demonstrated reward hack with before/after numbers, one helpful and one backfiring shaping scheme, and a write-up connecting it to RLHF in your own words.

## Stretch goals

- Solve **8×8 FrozenLake** (64 states) with your Project 1 code — sparser reward, so exploration gets much harder. How do epsilon and episode count have to change?
- Implement **Double DQN**: select the best next action with the online net but *evaluate* it with the target net (one line changed) — it reduces DQN's systematic overestimation of Q-values. Compare learning curves.
- Beat **LunarLander-v3** (`pip install "gymnasium[box2d]"`) with your DQN — 8-dimensional state, 4 actions, and much more satisfying to watch land. Its physics engine, the Box2D package, installs only on Python 3.13 or older, on a Mac or on a PC with an Intel or AMD (x86_64) processor: on an ARM computer running Linux or WSL2, the install fails with `Failed to build box2d-py`, so try this one in Google Colab.
- Implement bare-bones **REINFORCE** (policy gradient) on CartPole in ~60 lines and compare it to your DQN: noisier per-episode, but no replay buffer or target network needed.

## If you get stuck

- **RL is uniquely frustrating to debug** because a silent bug just makes the agent mediocre instead of crashing. Fix hyperparameters last, bugs first: log epsilon, buffer size, average Q-value, and episode reward every N episodes — most bugs are visible in one of those four.
- **Randomness lies.** Any single RL run can fail by bad luck. Before concluding your code is broken, run it 3 times (or with `seed=0,1,2`); before concluding it works, same.
- Read error tracebacks bottom-up; the last line names the actual problem. Print tensor shapes and dtypes at every step of the loss computation.
- Reduce until it works: non-slippery FrozenLake (`is_slippery=False`) should hit 100% quickly — if it doesn't, the bug is in your Q-learning code, not the difficulty.
- Ask an AI assistant for a HINT ("here is my update rule and my symptom — what category of bug should I look for?"), not a solution. And type all code yourself: the Bellman update needs to pass through your fingers to stick.

## Resources

- [Gymnasium docs](https://gymnasium.farama.org) — API reference for every environment (state/action spaces, reward rules, wrappers); keep it open while coding.
- [OpenAI Spinning Up](https://spinningup.openai.com) — the best-written free intro to RL concepts; read "Key Concepts" after Project 1 and the policy-gradient intro after Project 3.
- David Silver's RL course — the classic 10-lecture university course (he led the AlphaGo team); search YouTube for "David Silver reinforcement learning course". Watch lectures 1-2 for depth on the theory behind your Bellman one-liner.

## Skills unlocked

- [ ] I can explain the RL loop (state → action → reward → next state) and code it against any Gymnasium environment.
- [ ] I can implement epsilon-greedy exploration and explain why pure exploitation fails.
- [ ] I can write the Bellman update from memory and say in plain words what each term does.
- [ ] I can explain what the discount factor changes about an agent's behavior.
- [ ] I can build a DQN with a replay buffer and target network, and explain why each exists.
- [ ] I can recognize reward hacking from its signature (metric up, true goal down) and design a fix.
- [ ] I can explain the policy-gradient idea and how RLHF uses RL to align LLMs — and why alignment is hard, from firsthand experience.

## Next up

Back on the main track: learn how Netflix and Spotify guess what you'll like by factoring a giant ratings matrix — [40 · Recommender Systems: Matrix Factorization on MovieLens](40-recommender-systems.md).
