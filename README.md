## Introduction to Reinforcement Learning 

_Programming exercises from Reinforcement Learning, an introduction by Sutton and Barto._

### Part I: Tabular Solution Methods

#### Chapter 2. Multi-Armed Bandits

- [x] Read
- [x] Tackling a non-stationary problem (Ex. 2.5., page 33)

#### Chapter 3. Finite Markov Decision Processes

- [x] Read

#### Chapter 4. Dynamic Programming

---

> Dynamic Programming (DP) methods compute the full expectations to evaluate states. The general DP update rule is:
$$V(S_t) \leftarrow \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})]$$

---

- [x] Read
- [x] Convergence of iterative policy evaluation on a small gridworld (Reproduced Figure 4.1, page 77)
- [x] Jack's car rental (Exercise 4.7, page 82)
- [x] Gambler's problem (Exercise 4.9, page 84)

#### Chapter 5. Monte Carlo Methods

---

> Monte Carlo (MC) methods learn value functions and optimal policies from experience by _sampling episodes_. The general update rule for MC methods is:
$$V(S_t) \leftarrow V(S_t) + \alpha [\textcolor{red}{G_t} - V(S_t)]$$
where $\textcolor{red}{G_t}$ is the episode return, which is used as a target. Note that these methods require finishing a full episode before the value estimates can be updated. Advantages of MC methods over DP: 
> - 1. Can interact directly with the environment 
> - 2. Can be used with simulation of sample models. This is useful as in many applications constructing an explicit transition model is hard, but simulating sample episodes is easy. 
> - 3. It is easy and efficient to focus MC methods on a subset of the state space. 

> A recurring challenge is maintaining sufficient exploration. How do we ensure that an agent continues to explore? Two main approaches:
> - **On-policy methods**: $\pi(a \mid s) > 0 \, \forall \, s, a$ ("soft") but shift gradually closer to a deterministic policy. Central idea: GPI
> - **Off-policy methods**: choose one policy to generate sample episodes (behavior policy $b$) and one target policy which we optimize $\pi$. This class of learning methods are based on the idea of _importance sampling_, which is a variance reduction technique. Put simply, we choose a distribution which encourages the important values (i.e. those that have more impact on the parameters). In RL, the importance sampling ratio is:
$$\rho_{t:T-1} = \prod_{k=1}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$
There are various kinds of importance sampling methods, such as _ordinary importance sampling_ or _weighted importance sampling_. 

---

- [x] Read
- [x] Monte Carlo control in Easy21; a simplification of Blackjack (Reproduced Figure 5.1, page 100)

#### Chapter 6. Temporal Difference Learning

---

> Temporal-Difference (TD) methods update the value estimates after **every** timestep. The general update rule is:
$$V(S_t) \leftarrow V(S_t) + \alpha [\textcolor{red}{R_{t+1} + \gamma V(S_{t+1})} - V(S_t)]$$
where $\textcolor{red}{R_{t+1} + \gamma V(S_{t+1})}$ is used as a target.

---

- [x] Read
- [x] Example 6.2: Random walk (page 125)
- [x] Figure 6.2: Batch training (page 127) --> $\textcolor{red}{TBR}$
- [x] Example 6.4: Sarsa in the windy gridworld (page 130)
- [x] Example 6.6: The cliff walking task (page 132) --> $\textcolor{red}{TBR}$
- [ ] Figure 6.5: Comparison of Q-learning and Double Q-learning (page 135)

<figure>
<center>
<img src='https://www.yanxurui.cc/posts/ai/2019-04-24-RL-Introduction/unified_view-144-best-min.jpg' width='600'/>
<figcaption>A unified view of the classes of RL algorithms discussed so far.</figcaption></center>
</figure>

#### Chapter 7. $n$-step Bootstrapping

---

> $n$-step methods give you a spectrum between Monte-Carlo methods on the one extreme and one-step methods on the other. $n$-step methods are essentially approximations of the full return, truncated after $n$ steps with a correction for the remaining missing terms: $V(S_{t+n-1})$. The state-value learning algorithm for $n$-step learning is:
$$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \left[\textcolor{red}{G_{t:t+n}} - V_{t+n-1}(S_t) \right]$$
where $\color{red}{G_{t:t+n}}$ is the target. What is a good value for $n$? That's a good question, and it seems there is no general answer. You typically look for a "sweet spot". Intuition. Small $n$: I credit the reward received to the last action; large $n$: I will update my estimates of the previous five actions in response for the reward I just received (See Fig. 7.4, page 147). The idea of $n$-step methods can be integrated with previously introduced topics like SARSA, Off-policy, Importance sampling, and so on.

---

- [x] Read
- [x] RL Course by David Silver - [Lecture 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc&ab_channel=GoogleDeepMind)
- [ ] Exercise 7.2: Sum of TD errors (page 143) 
- [ ] Figure 7.2: Performance of $n$-step TD-methods as a function of $\alpha$, for various values of $n$.

#### Chapter 8. Planning and Learning with Tabular Methods

---

> This chapter discusses model-free and model-based RL and introduces an architecture, _Dyna_, that integrates learning and planning into a single architecture. At the heart of both methods is the idea of **computing value functions** to understand how "good" a given action is in the current state. The Table below highlights the major differences:

| Model-free RL | Model-based RL |
| --- | --- |
| No model | Require a model or learn a model from experience  |
| **Learn** value function (and/or policy) **directly** from experience | **Plan** value function (and/or policy) from model |
| "Direct RL" | "Indirect RL" |
| e.g. Monte-Carlo & TD learning | e.g. Dynamic programming, heuristic search |
| Rely on learning by experience  | Rely on planning, which uses _simulated experience_ |

> Let's define some terminology first:
>  - What do we mean with a **model** in this context? Well, this is something that describes, for the environment we're in. the agents' understanding of that environment. The state to state transitions and state to reward transitions. If we have a good picture of the transition dynamics, we can use this to make a plan! A model is a representation of an MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} >$ parameterized by $\eta$ (where we assume for the moment that $\mathcal{S}, \mathcal{A}$ are known). 
>    - Example: in Chess, the model is the rules of the game 
>    - Concretely, a model $\mathcal{M}_v$ represents the state transitions:
$$S_{t+1} \sim \mathcal{P}_\eta (S_{t+1} \mid S_t, A_t) \\ R_{t+1} \sim \mathcal{R}_\eta (R_{t+1} \mid S_t, A_t)$$
where we typically assume conditional independence between the state and rewards, that is, 
$$\mathbb{p}[S_{t+1} , R_{t+1} \mid S_t, A_t] = \mathbb{p}[S_{t+1} \mid S_{t}, A_t] \mathbb{p}[R_{t+1} \mid S_{t}, A_t]$$
> - What is **planning**? Any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment
>    - state-space planning: searching through state-space (the one we focus on now)
>    - plan-space planning: search through the "space of plans" (? lol --> #TODO: check ref in book)
> Some advantages of model-based RL:
>   - We can **efficiently** learn a model through supervised learning
>   - We can **reason explicitly about model uncertainty**
>  Disadvantages:
>   - We need to learn a model first, then construct a value function from the model (so there are two sources of approximation error)

---

- [x] Read
- [ ] Figure 8.2: Average learning curves for Dyna-Q agents
- [ ] Figure 8.4: Average performance of Dyna agents, bocking task
- [ ] Example 8.4: Prioritized sweeping on mazes
- [ ] Figure 8.7: Comparison of efficiency of expected and sample updates

#### Fun Small Projects

- Implementation and analysis of Q-learning agents in the iterated prisoners dilemma (IPD) [[Github repo](https://github.com/daphnecor/prisoners-dilemma)]
     
     <a target="_blank" href="https://colab.research.google.com/drive/1dUiexAIpfyGwaiJ-M3OsAAONsXnX4hty?usp=sharing">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
     </a>
      
  
