# 大规模多智能体强化学习 -- 可变规模智能体环境下的挑战（一）
本系列首先从多智能体强化学习的基本概念入手，介绍多智能体系统基本概念与定义使用多智能体强化学习所对应的数学模型（Markov Game/Multiagent Markov Decision Process、Decentralized-Partially Observable Markov Decision Processes)。进一步，通过介绍近期大规模多智能体强化学习的现状，引出两点学术与工程并存的挑战：1.观测维度的变化、2.动作空间的变化，并于后续部分详细介绍针对上述两个挑战，学界中已有的处理方案及其实验效果。

此为大规模多智能体强化学习系列的第一部分，包含简介与大规模多智能体强化学习的现状。

## 大纲
* **简介**
    * 多智能体强化学习
        * 马尔可夫博弈（Markov Game)
            * 多智能体马尔可夫决策过程（Multiagent Markov Decision Process）
            * 去中心-部分观测马尔可夫决策过程（Decentralized-Partially Observable Markov Decision Processes）
* **大规模多智能体强化学习的现状**
    * 可变规模环境
        * 观测维度的变化
        * 动作空间的变化

## 简介
强化学习是人工智能领域中一个十分重要的研究领域，从Richard S.Sutton和Andrew G.Barto[1]提出后，吸引了学术界与工业界的广泛关注，并被应用在游戏领域[2-4]，机器人控制领域[5-7]，广告投放领域[8-10]等问题与领域中，取得了良好的成果。

多智能体系统是由一组具有同质/异质属性、结构与行为动态、并拥有自主决策能力的智能体构成的复杂系统，不同智能体间决策与行为相互作用，通过共同的协作或者对抗完成复杂的任务。多智能体系统广泛存在于现实生活与运用中[11-13]，如：在工业机器人作业中，多台机器手共同协作完成一个流水线任务[14]；在星际争霸II中，数十至数百个单位需要协作完成攻击或防御等多种任务[15-16]；在德州扑克中，玩家需要在无法知道对方底牌这种非完全信息博弈中，获得尽可能高的收益[4,17-18]。此外，还常见于分布式传感器系统[19]，分布式机器人系统[20]和交通控制系统[21]等。

针对研究问题与环境中智能体数目的不同，我们可以将强化学习领域划分成两大类：
* 第一类是单智能体强化学习[1]，即指考虑学习环境中只存在一个可控的智能体，通过不断与环境的交互，获得对于环境的认知与行为对应的价值信号，从而迭代优化，获得可执行的策略。
* 第二类是多智能体强化学习[22]，即学习环境中存在大于等于两个可控的智能体，智能体们在环境中不断交互与共同学习演化，并最终获得不同智能体的控制策略。

### 多智能体强化学习
在使用多智能体强化学习进行科研实验或解决实际问题时，我们会遇到两类挑战：
* 第一类挑战归属于强化学习这类优化方法本身的局限，单智能体强化学习优化过程中所需要考虑的利用-探索的权衡（exploration-exploitation trade-off）、稀疏奖赏（sparse reward）的获取、如何设置合理与正确的时序信度分配（temporal credit assignment）与面向学习效率提升的奖赏函数塑形（reward shaping）等等问题，均同样出现于具有相应特点的多智能体强化学习交互环境与学习优化过程之中。
* 第二类挑战归属于多智能体系统本身所具有的独特属性-交互环境中具有多个独立自主的智能体，其挑战可通过全局优化视角与特定智能体优化视角来分类：
    * 全局优化视角：以多智能体系统中智能体的整体表现为直接目标，在此前提下
        * 如何在合理与正确的时序信度分配基础上，合理与正确地将特定时间步骤下的贡献分配于环境中的每个智能体上。
        * 如何避免优化过程中由于学习样本中存在智能体的探索行为等因素导致智能体策略陷入shadowed equilibrium/relative over-generalization，从而获得“更优”的策略。
        * 等等。
    * 特定智能体优化视角：
        * 如何建模其他智能体的行为/策略，从而帮助特定智能体的策略进行优化或做出最佳响应。
        * 如何考虑与“我”交互的其他智能体的特点，进一步设计学习方案与对应的网络结构（当采用神经网络作为函数近似器时）：
            * 该时刻下“我”的最优行为与奖赏只与部分智能体相关，而非全体。
            * “我”的动作只与部分智能体交互，而非全体。
            * 在部分可观察的环境下，当前“我”只观测到部分智能体等等。
            * 等等。
        * 等等。

上述不同的挑战指明了不同科研领域的前进方向，在深入介绍本系列关注的“大规模多智能体强化学习 -- 可变规模智能体环境下的挑战”前，我们先从新手村出发，介绍几类具有代表性的多智能体环境对应的数学模型。

#### 马尔可夫博弈（Markov Game)

马尔可夫博弈（Markov Game）[27] 又被称为随机博弈（stochastic game）[28]，最初为博弈论领域建模与解释多智能体系统的数学模型，自Michael L. Littman采用马尔可夫博弈建模双人零和博弈并采用强化学习（独立的Q学习，independent Q-learning）求解后[29]，该建模方式逐渐成为多智能体强化学习领域约定俗成的用来描述环境与优化目标的数学模型。

马尔可夫博弈（Markov Game）由以下几个部分组成：
* $n$：环境中的智能体数量，通常大于等于2。
* $S$：状态空间（state space），通常包含所有智能体的信息与优化目标相关的环境信息。
* $A_{i}$：智能体$i$的动作空间（action space），该动作空间可能与当前智能体$i$所处的状态相关，也可能与智能体$i$的类别相关，一般情况下我们会假设所有智能体具有相同的动作空间。
* $P$: 状态转移函数（state transition function），通常采用$P$来描述在状态$S_t$下，智能体们采用相应动作后，转移到下一个状态$S_{t+1}$的概率, 即，$P:S \times A_1 \times A_2 \times \dots A_n \times S \rightarrow [0,1]$ 
* $R_i$: 智能体$i$的奖赏函数，通常用来描述在状态转移发生时，智能体$i$获得的奖赏，即，$P:S \times A_1 \times A_2 \times \dots A_n \times S \rightarrow \mathbb{R} $
* $\gamma$：折扣因子（discount factor），$0 \le \gamma < 1$

通过定义，我们可以发现马尔可夫博弈是一种非常“通用”的数学建模方式，不论所面临的优化问题是智能体们合作解决问题还是存在一定的冲突与合作，乃至于完全冲突，我们均可通过设置不同的奖赏函数$R_{i}$来进行刻画。然而由于其太过于“通用”，而学界所提出的多智能体强化学习方法常常只关注于相对“特定”的问题，比如完全合作的场景或完全竞争的场景等等，这使得采用马尔可夫博弈来定义优化问题颇有种“杀猪用牛刀”的感觉。所以学界进一步提出了相应的马尔可夫博弈变体，来缩小其描述的问题空间，从而更加精准地定位与解决相应的问题。

##### 针对不同环境特点的马尔可夫博弈变体
* 多智能体马尔可夫决策过程（Multiagent Markov Decision Process）[30]
对于完全合作的问题，可以理解成所有智能体具有完全相同的目标，这也意味着我们可以让所有智能体共享同一个奖赏信号，即所有智能体共享同一个奖赏函数，而不是每个智能体具有特定的奖赏函数。这样的马尔可夫博弈变体，我们称之为多智能体马尔可夫决策过程。

* 去中心-部分观测马尔可夫决策过程（Decentralized-Partially Observable Markov Decision Processes）[31]
通常，由于问题的客观原因与方法的通用性考虑，我们会考虑智能体无法获得环境中的所有信息（即state），所以在建模时，我们会在多智能体马尔可夫决策过程的基础上考虑每个智能体所能观测的信息，形式化的描述为：
    * $Z$: 观测空间（observations space），为智能体观测到的信息，通常为该时刻下state的子集。
    * $O$：观测函数（observation function），在每次状态转移时，智能体获得的信息，$O：S \times A \rightarrow Z$


## 大规模多智能体的现状
近年来，强化学习的研究逐渐从单智能体深度强化学习领域转向了多智能体深度强化学习领域[23-26]，其中如何将多智能体深度强化学习运用在大规模智能体环境上，由于其实际工业价值而备受关注。然而由于大规模智能体环境中智能体间存在复杂的交互关系、相比小规模智能体系统具有更高的策略学习动态性、庞大的动作空间与状态空间等挑战，当前学界聚焦于探索设计针对大规模多智能体深度强化学习的框架与算法。

当前大规模多智能体研究主要关注于完全合作的场景或竞争场景中特定一方的学习（将另一方视为环境的一部分，所以也可视为完全合作的场景），所以我们沿用上述去中心-部分观测马尔可夫决策过程（Decentralized-Partially Observable Markov Decision Processes）来描述我们关注的问题并沿用其符号。

由于缺乏高质量的大规模多智能体强化学习实验环境（当前的仿真环境随着数量的增多而性能显著下降），同时在实践上需要考虑存储样本对于内存的需求随着数量增多而显著增多（来源于两点，一点是obs可能随着数量的增多而增多，另外一点是需要存储n个agent的数据）与采用GPU训练时显存的增多，所以当前大规模多智能体研究大部分暂未直接挑战大规模多智能体环境下的训练，而是从小规模与中规模入手，一方面利用不同数量的仿真环境来研究算法/框架的扩展性[32]，另外一方面在“小”规模的环境中构造特定的大规模环境存在的特点并研究如何利用相应特点[33]，等等。

### 可变规模环境

当前间接研究大规模多智能体强化学习的一个切入点是利用可变规模环境来进行探索，其主要考虑的是大规模多智能体系统中常见的两个性质[32]：

```
Property 1 Partial Observability: 
In MASs, agents make decisions based on their local observations, 
in which way large-scale problems can be reduced to relatively indepen-
dent but correlated small-size ones.
```

```
Property 2 Sparse Interactivity: 
From the perspective of the global view, 
each agent only interacts with some of the agents in MASs at the same time, 
and the interactions do not happen all the time.
```

而上述两个特点主要体现在状态/观测空间与动作空间上，比如由于Sparse Interactivity特点的存在，所以当前只存在部分“有效”动作，而非整个动作空间均可以执行。即使存在上述的特点，当前常见的多智能体强化学习通常采用粗暴与直接的做法：默认固定obs/state的长度与动作空间的大小，当智能体缺乏对应观测信息的时候，采用填补0之类的做法，当采用“无效”动作时，让其停留于原地等等。这样的做法虽然粗暴与直接，但是从工程角度便于实现，在定义神经网络结构时，采用常见的MLP（ + RNN/GRU/LSTM，如果需要考虑时序依赖）便可以使用。

这自然抛出一个疑惑，既然存在这些特点，那为什么我们不利用这些特点呢？利用这些特点的增益较为直观，不过如果我们利用这些特点，那么会引入什么挑战？最直接的挑战便是我们需要能够处理：1. 观测维度的变化， 2. 动作空间的变化的算法与框架，怎么合理处理与利用上述挑战，我们将在下一期讨论与分享。


## 参考文献
[1] Sutton R S, Barto A G. Reinforcement learning: An introduction[M]. Cambridge: MIT press, 1998. 
[2] Mnih, Volodymyr, et al. “Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.
[3] Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature 529.7587 (2016): 484.
[4] Moravčík, Matej, et al. "Deepstack: Expert-level artificial intelligence in heads-up no-limit poker." Science 356.6337 (2017): 508-513.
[5] Gu, Shixiang, et al. “Continuous Deep Q-Learning with Model-Based Acceleration.” ICML’16 Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, 2016, pp. 2829–2838.
[6] Ebert, Frederik, et al. “Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control.” ArXiv Preprint ArXiv:1812.00568, 2018.
[7] Hester, Todd, and Peter Stone. “TEXPLORE: Real-Time Sample-Efficient Reinforcement Learning for Robots.” Machine Learning, vol. 90, no. 3, 2013, pp. 385–429.
[8] Wang, Weixun, et al. “Learning Adaptive Display Exposure for Real-Time Advertising.” Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019, pp. 2595–2603.
[9] Chen, Dagui, et al. “Learning to Advertise for Organic Traffic Maximization in E-Commerce Product Feeds.” Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019, pp. 2527–2535.
[10] Peng, Zhaoqing, et al. “Learning to Infer User Hidden States for Online Sequential Advertising.” Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 2020, pp. 2677–2684.
[11] Wooldridge M, Jennings N R. Intelligent agents: Theory and practice. The knowledge engineering review, 1995, 10(02): 115-152.
[12] Barto A G, Sutton R S, Anderson C W. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, 1983 (5): 834-846.
[13] Duan, Jiajun, et al. “Deep-Reinforcement-Learning-Based Autonomous Voltage Control for Power Grid Operations.” IEEE Transactions on Power Systems, vol. 35, no. 1, 2020, pp. 814–817.
[14] Laengle T, Wörn H. Human–Robot Cooperation Using Multi-Agent-Systems. Journal of intelligent and Robotic Systems, 2001, 32(2): 143-160.
[15 ] Vinyals, Oriol, et al. “Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning.” Nature, vol. 575, no. 7782, 2019, pp. 350–354.
[16] Zambaldi, Vinícius Flores, et al. “Deep Reinforcement Learning with Relational Inductive Biases.” International Conference on Learning Representations, 2018.
[17] Brown, Noam, and Tuomas Sandholm. “Superhuman AI for Multiplayer Poker.” Science, vol. 365, no. 6456, 2019, pp. 885–890.
[18] Brown, Noam, and Tuomas Sandholm. “Superhuman AI for Heads-up No-Limit Poker: Libratus Beats Top Professionals.” Science, vol. 359, no. 6374, 2018, pp. 418–424.
[19] Criado N, Argente E, Garrido A, et al. “Norm enforceability in electronic institutions?” Coordination, organizations, institutions, and norms in agent systems VI. Springer, Berlin, Heidelberg, 2011: 250-267.
[20] Morales J, Lopez-Sanchez M, Rodriguez-Aguilar J A, et al. “Automated synthesis of normative systems.” Proceedings of the 2013 international conference on Autonomous agents and multi-agent systems. International Foundation for Autonomous Agents and Multiagent Systems, 2013: 483-490.
[21] Morales J, López-Sánchez M, Esteva M. Using experience to generate new regulations. IJCAI. 2011: 307-312.
[22] Wooldridge, M.. “An Introduction to MultiAgent Systems, Second Edition.” (2009).
[23] Hernandez-Leal, Pablo, Bilal Kartal, and Matthew E. Taylor. "A survey and critique of multiagent deep reinforcement learning." Autonomous Agents and Multi-Agent Systems 33.6 (2019): 750-797.
[24] Palmer, Gregory, et al. "Lenient multi-agent deep reinforcement learning." arXiv preprint arXiv:1707.04402(2017).
[25] Gupta, Jayesh K., Maxim Egorov, and Mykel Kochenderfer. "Cooperative multi-agent control using deep reinforcement learning." International Conference on Autonomous Agents and Multiagent Systems. Springer, Cham, 2017.
[26] Jaques, Natasha, et al. "Social influence as intrinsic motivation for multi-agent deep reinforcement learning." International Conference on Machine Learning. PMLR, 2019.
[27] Van Der Wal, J. Stochastic dy- namic programming. In Mathematical Centre Tracts 139. Morgan Kaufmann, Amsterdam. 1981. 
[28] Owen, Guillermo. Game Theory: Sec- ond edition. Academic Press, Orlando, Florida. 1982.
[29] Littman M L. Markov games as a framework for multi-agent reinforcement learning[M]//Machine learning proceedings 1994. Morgan Kaufmann, 1994: 157-163.
[30] Hu, Junling, and Michael P. Wellman. "Multiagent reinforcement learning: theoretical framework and an algorithm." ICML. Vol. 98. 1998.
[31] Bernstein, Daniel S., et al. "The complexity of decentralized control of Markov decision processes." Mathematics of operations research 27.4 (2002): 819-840.
[32] Wang, Weixun, et al. "From few to more: Large-scale dynamic multiagent curriculum learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 05. 2020.
[33] Liu, Yong, et al. "Multi-agent game abstraction via graph attention neural network." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 05. 2020.
