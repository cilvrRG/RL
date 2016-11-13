# RL
##Reading Group on Reinforcement Learning topics
##NYU, Fall 2016

###Logistics 
  - Meetings run every **Wednesday at 9h30** (before the CILVR lab meeting), at the large conference room at 715/719 Broadway 12th floor. Breakfast will be provided.
  - Paper discussion + Paper review plan: Each week we will assign one or two papers to volunteers who will present it following week. During the reading/presentation, we will edit a review of the paper which will be posted here. [also subject to change].
  - Guest speakers. We will try to invite RL experts (e.g. G. Tesauro) with some frequency.
  - Other communication channels ( Facebook groups?, Slack? ) [TBD].
  
###Organization
The RG is initially organized by J.Bruna, K. Cho, S. Sukhbaatar, K. Ross, D. Sontag, with help from the rest of the CILVR group.

### Tentative Agenda

  - 9/21: [Tutorial on MDPs, Policy Gradient (part 1)](MDP_RL_Lecture1.pdf). [**Keith Ross**]
    - Markov Decision Process Paradigm
    - Discounted and average cost criteria
    - Model-free Reinforcement Learning Paradigm
    - Policy Gradient: parameterized policies; policy gradient theorem; Monte Carlo Policy Gradient (REINFORCE)
    - Using Policy Gradient and deep neural networks to learn the Atari game "pong".

  - 9/28: [Tutorial on MDPs, Policy Gradient (part 2)](MDP_RL_Lecture2.pdf). [**Keith**]
    - Dynamic Programming equations for MDPs
    - Policy iteration 
    - Value iteration
    - Monte Carlo methods for RL 
    - Q-learning for RL 
    
  - 10/5 and 10/12: Actor-Critic. [**Martin**]
    - Deterministic Policy Gradient
    - Off-Policy variants
    - Relevant Papers:
      - [Policy Gradient and Actor Critic](https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf)
      - [Deterministic Policy Gradients](http://jmlr.org/proceedings/papers/v32/silver14.pdf)
      - [Off-policy actor critic](https://webdocs.cs.ualberta.ca/~sutton/papers/Degris-OffPAC-ICML-2012.pdf)
       
  - 10/19: Tutorial on OpenAI Gym and Mazebase. Also, Twitter's new [twrl](https://github.com/twitter/torch-twrl)  [**Sainaa and Ilya**]
    - MazeBase: https://github.com/facebook/MazeBase
  - 10/26: [Apprenticeship Learning via Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) and  [Model-Free Imitation Learning with Policy Optimization](https://arxiv.org/abs/1605.08478) [**Arthur**]
  - 10/31: Trust region policy optimization (TRPO) [**Elman, Ilya**]




### Pool of Papers [please fill]

 - Guided Policy Search
 - Value Iteration Networks
 - TRPO [Elman, early November]
 - Review of recent hierarchical reinforcement learning papers [Sainaa]
 - Intrinsically Motivated Reinforcement Learning [Martin?]:
   - [Intrinsically Motivated Reinforcement Learning](https://web.eecs.umich.edu/~baveja/Papers/FinalNIPSIMRL.pdf)
   - [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](https://arxiv.org/pdf/1509.08731v1.pdf)
   - [Bayesian Surprise Attracts Human Attention](https://papers.nips.cc/paper/2822-bayesian-surprise-attracts-human-attention.pdf)
   - [Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)
   - [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868v2.pdf)
 - High dimensional action spaces:
   - [Reinforcement Learning with Factored States and Actions](http://www.jmlr.org/papers/volume5/sallans04a/sallans04a.pdf)
   - [Deep Reinforcement Learning in Large Discrete Action Spaces](https://arxiv.org/pdf/1512.07679.pdf)
   - [Learning Multiagent Communication with Backpropagation](https://arxiv.org/pdf/1605.07736.pdf)
 - Stability in RL (these 4 papers shouldn't take more than 1 or 2 lectures):
   - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
   - [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
   - [Double Q-learning](http://papers.nips.cc/paper/3964-double-q-learning.pdf). Also consider reading about [the optimizer's curse](https://faculty.fuqua.duke.edu/~jes9/bio/Optimizers_Curse.pdf) to make the reading simpler.
   - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
