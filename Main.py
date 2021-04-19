# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:10:22 2020

Main code for CS841 Project

Here, the intereaction between environment, agent
results, and data vizualization take place

@author: Juan Rios
"""
# %% Load the dependencies and Hyperparameters
from Environment import Environment
from BayesianAgent import BayesianAgent
from Epsilon import Epsilon

# FUTURE WORK: 
# 1. add bayesian learning


# TO DO for project
# train a Q learning anget and Bayesian agent, with certian information and plot performance w/ baseline random
# Train a Q learning agent and Bayesian agent, with uncertainty and plot performance w/ baseline random
# train a number of Q learning agents, and plot the time gaps, compare to experiment in paper
# train a number of bayesian learning agents and plot time gaps, compare to experiment in paper



# HYPERPARAMETERS:
# This code section provides the adjustable hyper parameters for ruinning this simulation

# World params
fullObs = False; # Describes whether an agent observed the true state, or a probabilistic approximation
numCols = 13 # describes the world size, in terms of the length of the road the car can transverse

# Agent params
learnRate = 0.01 # The learning rate of the agents
e = Epsilon(1, 0.01, 0.001) # epsilon greedy strategy
gamma = 0.99 # The discount rate of future rewards
agentXtime = 5 # time it takes the agent to cross the street

# training and testing
numTraining = 300; # number of training simulations
numTesting = 2000 # number of testing simulations
timeIntTrain = 1 # how many training trials to perfom before minitesting
timeIntTest = 50 # the num of test trials to wait at test time before collecting data
numTests = 200 # the number of trials at each mini test


# %% Initialize the agents, the world, and begin training

# init the list to hold the training scores
trainScores = []
# create the world
World = Environment(numCols)

# create the agents
bAgent = BayesianAgent(agentXtime, "Bayesian", learnRate, e, gamma, numCols)
qAgent = BayesianAgent(agentXtime, "Q-learning", learnRate, e, gamma, numCols)
rAgent = BayesianAgent(agentXtime, "Random", learnRate, e, gamma, numCols)

# trian and test the agent
trainScore  = World.train(numTraining, fullObs, bAgent, timeIntTrain, numTests)
trainScores.append(trainScore)

trainScore = World.train(numTraining, fullObs, qAgent, timeIntTrain, numTests)
trainScores.append(trainScore)

trainScore  = World.train(numTraining, fullObs, rAgent, timeIntTrain, numTests)
trainScores.append(trainScore)





# %% Plot the agents train performance

labels = ["Bayesian learning", "Q learning", "no learning"]
World.plotTrainScores(numTraining, trainScores, labels)



# %% Test the agents

World.test(numTesting, fullObs, bAgent, timeIntTest)
World.test(numTesting, fullObs, qAgent, timeIntTest)
World.test(numTesting, fullObs, rAgent, timeIntTest)


# %% Plot the agents test performance

agentList = []
agentList.append(bAgent)
agentList.append(qAgent)
agentList.append(rAgent)
labels = ["Bayesian", "Q", "Random"]
World.plotScores(numTesting, agentList, labels)


# %% Testing
import numpy as np
import math

# a multitude of agents are trained, and their behavior collected

mode = "Bayesian"
fullObs = False
World = Environment(numCols)
Obs = np.zeros((2,0))# a list to hold the timegap, action park
t = 0
numTests = 1

while t < 311:
    
    # pick a random crosstime
    agentXtime = round(np.random.uniform(2,5))
    # make the learning agent
    agent = BayesianAgent(agentXtime, mode, learnRate, e, gamma, numCols)
    trainScore  = World.train(numTraining, fullObs, agent, timeIntTrain, numTests)
    
    # generate world
    state = World.genWorld(agent)
    done = False
    
    while not done:
        
        # agent chooses action based on the observed time-gap
        agent.selectAction(state, math.inf, fullObs)
        if agent.getAction() == agent.getPossibleActions()[1]:
            # each time step, the crossing time to arrive at the goal is diminshed by 1
            agent.cross()
            done = True
        # world exceucutes action
        nextState = World.step(state)
         
        Obs = np.append(Obs, [[agent.getAction() - 1],[state]], axis = 1)
        
        
        t = t + 1
        
        state = nextState
        
        if state == 0:
            done = True
                
 # %% Fitting the data to logistic function
from sklearn import linear_model
from scipy.special import expit   
import matplotlib.pyplot as plt    
        

X = Obs[1,:].reshape(-1, 1)
y = Obs[0,:]
# Fit the classifier
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(0, 12, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)
plt.title("Crossing Behavior for " + mode)
plt.ylabel("action probability")
plt.xlabel("time gap (s)")
plt.show()



# %% Finding the midpoint

probs = clf.predict_proba(X_test.reshape(-1,1))


