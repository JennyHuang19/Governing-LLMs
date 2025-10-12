# Governing AI

The AI systems being built now will decide how knowledge and agency are distributed in the next 5-10 years.
Because of this, it falls to us (researchers in ML) all to be knowledgeable enough to think critically about the way these systems are currently set up.

Who should decide how AI is governed?
It would be dangerous for this process to fall into the hands of a privileged few.
So, how can we construct principled systems, ones that ensure that, as the human-AI relationship progresses, it expands human flourishing rather than eroding it? 
How can we work towards a collective project that draws on many forms of wisdom: scientific, ethical, cultural, institutional.

The frameworks that large AI labs currently use to “guide model behavior” currently lack rigor and scientific grounding.

For instance:
- OpenAI relies on a [Model Spec](https://model-spec.openai.com/2025-02-12.html).
- Anthropic has an [AI Constitution](https://constitutional.ai/#definition).

Recent works have highlighted gaps between the ideal behaviors AI companies are declaring and the actual behaviors these models exhibit — for example, see this 2025 paper: [SpecEval: Evaluating Model Adherence to
Behavior Specifications](https://arxiv.org/pdf/2509.02464).

Let's first discuss the current big-picture pipeline.

---

# How Does Post-training Work?

A pre-trained LLM is a **next-token prediction machine**. Much of the process for getting an LLM to output responses useful for humans happens in post-training.

![Post-training pipeline](imgs/post_training_diagram.jpg)  
*(image credit: [Cameron Wolfe's blog on PPO](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo))*

1. **Supervised Fine-tuning (SFT):** Human annotators provide “gold standard” responses, and models imitate those responses. The model is fine-tuned using supervised learning to minimize the discrepancy between human- and model-generated responses.
2. **Reward-model Training:** The LLM generates two responses, and the human annotates which response they prefer (or ranks multiple responses).
3. **Reinforcement Learning (RL):** The LLM learns how to interact with its environment through trial and error, similar to how a human would learn from experience.

---

# RL Defined in an LLM Context

- **Policy:** A rule used by the generator model to decide which action (token) to take — i.e., the model’s weights.
- **State:** The context or prompt.
- **Action:** The next token that the generator model outputs.
- **Reward:** The score given to an output (typically from a reward model).
- **Value:** The total expected reward starting from a state and acting according to a particular policy.

![Simple RL diagram](imgs/simple_rl_diagram.png)  
*(image credit: [AlmaBetter blog on RL](https://www.almabetter.com/bytes/articles/reinforcement-learning))*

---

# Popular RL Paradigms for LLM Post-training

## PPO (Proximal Policy Optimization) and GPRO (Group Relative Policy Optimization)
![PPO vs. GRPO](imgs/ppo_grpo.png)  
*(image credit: [DeepSeek R1](https://arxiv.org/pdf/2501.12948#page=3.10))*

PPO is the most commonly-used algorithm for RLHF and the most complex of the three we’ll discuss. It requires training three models: the generator, the reward, and the critic.

1. The generator model outputs a response, which is scored by the reward model.  
2. The critic model (typically a more skilled model) also outputs a response, which is scored by the reward model.  
3. The difference in reward between the generator and critic responses is used as the training signal.

The GRPO algorithm avoids the use of the value model, replacing it with an **average** of several outputs from the generator model.

---

## DPO (Direct Preference Optimization)
![Direct Preference Optimization](imgs/direct_preference_optimization.jpg)  
*(image credit: [Cameron Wolfe’s blog on DPO](https://cameronrwolfe.substack.com/p/direct-preference-optimization))*

This algorithm removes both the critic and reward models, training the generator model directly on a **Bradley–Terry-based objective** derived from preference data.

Problem: Low signal-to-noise.
This paper[] shows that much of the RLHF signal can be explained by the length of the response. Thus, maybe RLHF generalizes well because there is a ton of data and also can pick up a "gut reaction" for which response is better.
Beyond that, if we want to align the model on more nuanced behaviors (e.g., confront the syncophancy problem) it will requires a more informative training signal.

---

## RLVR (Reinforcement Learning with Verifiable Rewards)
This paper[] shows that even RLVR has low signal-to-noise.

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

 
