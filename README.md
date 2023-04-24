## Introduction to Reinforcement Learning 

_Programming exercises from Reinforcement Learning, an introduction by Sutton and Barto._

### Part I: Tabular Solution Methods

#### Chapter 2. Multi-armed Bandits

- [x] Read
- [x] Tackling a non-stationary problem (Ex. 2.5., page 33)

#### Chapter 3. Finite Markov Decision Processes

- [x] Read

### Chapter 4. Dynamic Programming

---
> Important idea: General Policy Iteration (GPI)
---

- [x] Read
- [x] Convergence of iterative policy evaluation on a small gridworld (Reproduced Figure 4.1, page 77)
- [x] Jack's car rental (Exercise 4.7, page 82)
- [x] Gambler's problem (Exercise 4.9, page 84)

### Chapter 5. Monte Carlo Methods

---

> Monte Carlo methods learn value functions and optimal policies from experience by _sampling episodes_. Advantages of MC methods over DP: 
- 1. Can interact directly with the environment 
- 2. Can be used with simulation of sample models. This is useful as in many applications constructing an explicit transition model is hard, but simulating sample episodes is easy. 
- 3. It is easy and efficient to focus MC methods on a subset of the state space. 

A recurring challenge is maintaining sufficient exploration. How do we ensure that an agent continues to explore? Two main approaches:
- **On-policy methods**: $\pi(a \mid s) > 0 \forall s, a$ ("soft") but shift gradually closer to a deterministic policy. Central idea: GPI
- **Off-policy methods**: choose one policy to generate sample episodes (behavior policy $b$) and one target policy which we optimize $\pi$. This class of learning methods are based on the idea of _importance sampling_, which is a variance reduction technique. Put simply, we choose a distribution which encourages the important values (i.e. those that have more impact on the parameters). In RL, the importance sampling ratio is:
\begin{align}
\rho_{t:T-1} = \prod_{k=1}^T-1 \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}
\begin{align}
There are various kinds of importance sampling methods, such as _ordinary importance sampling_ or _weighted importance sampling_. 

---

- [x] Read
- [x] Monte Carlo control in Easy21; a simplification of Blackjack (Reproduced Figure 5.1, page 100)
- [ ] Racetrack (Exercise 5.12, page 111)

### Chapter 6

- [ ] Read
- [ ] 

### Fun small projects

- Implementation and analysis of Q-learning agents in the iterated prisoners dilemma (IPD) [[Github repo](https://github.com/daphnecor/prisoners-dilemma)]
     
     <a target="_blank" href="https://colab.research.google.com/drive/1dUiexAIpfyGwaiJ-M3OsAAONsXnX4hty?usp=sharing">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
     </a>
      
  
