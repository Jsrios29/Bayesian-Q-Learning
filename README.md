# BQL
Bayesian Q-learning to model pedestrian street-crossing behavior

Description: These python files train an angent to cross a street with oncoming traffic. The agent learns both through traditional and Bayesian-Q learning. The agent is tested and the results of the behavior are compared to real street crossing behavior.

Files:

1. Bayesian_Q_Learning_JuanRios.pdf - this is the main file that details the algorithm and the results.
2. Main.py - This runs the simulation and analysis of the agent's performance in the environment
3. Epsilon.py - This class manages the agent's greedy-epsilon learning strategy, only useful for the traditional Q-learning agent, since the Bayesian agent implicitly balances this exploration/exploitation concept.
4. Environment.py - manages the simulation of the environment, example: the oncoming traffic, the world, the rewards.
5. BayesianAgent.py -although this class also covers the traditional agent, it also represents that main Bayesian agent that interacts with the world and learns from its actions.
