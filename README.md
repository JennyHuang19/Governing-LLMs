# Governing AI

The AI systems being built now will decide how knowledge is distributed and how societies are structured in the next 5–10 years.  

*So who should decide how AI is governed?*
It would be dangerous for this process to fall into the hands of a privileged few. It is up to us (as ML PhD students!) to be informed about and to think critically about how current systems are set up. How can we construct principled systems and work towards a collective project that draws on many forms of wisdom: scientific, ethical, cultural, and institutional?

---

## The Status Quo: Behavioral Frameworks in AI Labs

The frameworks that large AI labs currently use to “guide model behavior” currently lack rigor and scientific grounding.

For instance:
- OpenAI relies on a [Model Spec](https://model-spec.openai.com/2025-02-12.html).
- Anthropic has an [AI Constitution](https://constitutional.ai/#definition).

Recent works have highlighted gaps between the ideal behaviors AI companies are declaring and the actual behaviors these models exhibit — for example,  
[SpecEval: Evaluating Model Adherence to Behavior Specifications (2025)](https://arxiv.org/pdf/2509.02464).

---

# How Does Post-training Work?

A pre-trained LLM is a **next-token prediction machine**.  
Much of the work that makes an LLM *useful* happens in post-training.  

You can think of:
- **Pre-training** → condensing large volumes of information into a smart data structure.  
- **Post-training** → algorithms to extract and refine useful behavior.

![Post-training pipeline](imgs/post_training_diagram.jpg)  
*Image credit: [1]*

### The Stages

0. **Define a rubric of ideal behaviors:** e.g., follow instructions, be honest and helpful, avoid harmful content.  
1. **Supervised Fine-tuning (SFT):** Human annotators provide “gold standard” responses, and models imitate these responses.  
2. **Reward Model Training:** The LLM generates two responses, and humans annotate which they prefer.  
3. **Reinforcement Learning (RL):** The LLM learns from trial and error.  
4. **Prompting (In-context learning):** Prompts guide the model on what to output during inference.

---

# RL Defined in an LLM Context

Language modeling fits naturally into a reinforcement learning framework:

- **Policy:** The rule (model weights) determining which token to output.  
- **State:** The input context or prompt.  
- **Action:** The next token chosen by the model.  
- **Reward:** The score assigned to an output (often from a reward model).  
- **Value:** The expected total reward from a state when following a policy.

![Simple RL diagram](imgs/simple_rl_diagram.png)  
*Image credit: [5]*

---

# Popular RL Paradigms for LLM Post-training

<!-- ![RL taxonomy](imgs/rl_taxonomy.png)  
*Image credit: [1]* -->

## PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization)

![PPO vs. GRPO](imgs/ppo_grpo.png)  
*Image credit: [2]*

**PPO** is the most widely used algorithm for RLHF and requires training three models:  
the generator, reward model, and critic.

1. The generator model outputs a response, which is scored by the reward model.  
2. The critic model (often a more capable model) also outputs a response, scored by the same reward model.  
3. The difference in reward between generator and critic responses becomes the training signal.

**GRPO** simplifies PPO by avoiding the value model, replacing it with an **average** over several generator outputs.

---

<!-- motivation for DPO (act as a transition) -->

## DPO (Direct Preference Optimization)

![Direct Preference Optimization](imgs/direct_preference_optimization.jpg)  
*Image credit: [6]*


In this section, we will look at the intuition behind **Direct Preference Optimization (DPO)**: how it connects to the RL objectives of PPO/GRPO, and why its gradient naturally increases the likelihood of *preferred* responses while decreasing that of *non-preferred* ones.


### i. From RLHF to DPO

In RLHF, the objective is to train a model to maximize expected reward under a KL constraint that keeps it close to a reference model (often the SFT model):

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(y | x)} \Big[ r(x, y) \Big] - \beta \mathbb{D}_{\text{KL}}\big(\pi_\theta(y | x) || \pi_{\text{ref}}(y | x)\big)
$$

where:
- $r(x, y)$ is the reward (learned from human preferences),
- $\pi_{\text{ref}}$ is the reference policy (often the SFT model),
- $\beta$ balances reward maximization vs. staying close to the reference.

### ii. The Optimal Policy

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \, e^{r(x,y)/\beta}
$$

This means the optimal policy is a **Boltzmann distribution** over rewards:
- Reweights the reference model’s probabilities.  
- Shifts more mass toward higher-reward outputs.  
- The temperature $\beta$ controls how strong this shift is.


### iii. Pairwise Preferences and the Bradley–Terry Model

Human feedback is often in the form of a pair of model completions $(y_w, y_l)$, where one is preferred over the other.

A probabilistic model, the **Bradley–-Terry model**, is used to model the pairwise-preference data:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)
$$

where $\sigma$ is the logistic sigmoid.

### iv. Substituting the Optimal Policy

$$
P(y_w \succ y_l \mid x)= \sigma\Big(\beta \log \frac{\pi^{*}(y_w|x)}{\pi^{*}(y_l|x)} - \beta \log \frac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)}\Big)
$$

This shows that **pairwise preferences** can be modeled entirely in terms of likelihood ratios: no explicit reward model needed.

---

### v. The DPO Objective

DPO trains a parameterized policy $\pi_\theta$ directly by minimizing the negative log-likelihood of observed preferences:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \Big[ \log \sigma \Big( \beta \big[ \log \tfrac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log \tfrac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)}\big]\Big)\Big]
$$

Thus, DPO is **supervised learning on preference pairs** — learning a policy that maximizes the observed human-preference dataset.

### vi. The Gradient of DPO

Define:

$$
z = \beta \left[\log \tfrac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log \tfrac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)}\right],\quad p = \sigma(z)
$$

Then:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}= \beta(p - 1)\Big[\nabla_\theta \log \pi_\theta(y_w|x)- \nabla_\theta \log \pi_\theta(y_l|x)\Big]
$$

During training, we step in the **negative gradient direction**:

$$
-\nabla_\theta \mathcal{L}_{\text{DPO}}\propto(1 - p)\Big[\nabla_\theta \log \pi_\theta(y_w|x)- \nabla_\theta \log \pi_\theta(y_l|x)\Big]
$$

---

# The Low Signal-to-Noise Problem

Some studies have shown that much of the RLHF signal can be explained by response length or surface features.  
Thus, RLHF may generalize well primarily because of its *scale* — not necessarily because the feedback signal is deeply informative.  
If we wish to align models on *nuanced behaviors* (e.g., reducing sycophancy), we need richer, more diagnostic training signals.

---

## RLVR (Reinforcement Learning with Verifiable Rewards)

Even **RLVR** (which uses verifiable, rule-based rewards) suffers from low signal-to-noise, illustrating how difficult it is to construct meaningful alignment objectives.


---

# Ways forward

Pluralistic Alignment: 
Inverse Constitutional Alignment: 

# Sources
1. [Proximal Policy Optimization (PPO) — Cameron Wolfe’s Blog](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo) 
2. [DeepSeek R1](https://arxiv.org/pdf/2501.12948#page=3.10)
3. [Stanford CS336 Lecture 15: Alignment — SFT/RLHF](https://web.stanford.edu/class/cs336/)  
4. [LLM Training & Reinforcement Learning - Explanation Video on YouTube](https://www.youtube.com/watch?v=aB7ddsbhhaU)  
5. [An Introduction to Reinforcement Learning for Beginners — AlmaBetter Blog](https://www.almabetter.com/bytes/articles/reinforcement-learning)  
6. [Direct Preference Optimization (DPO) — Cameron Wolfe’s Blog](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
7. [Stanford CS336 Lecture 16: Alignment — RL 1](https://web.stanford.edu/class/cs336/)  

 
