# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:59:43 2020

This class deals with the environment of the grid world. Producing the states and keeping track of
OBject position. It also builds the grid-world itself

@author: Juan Rios
"""
import numpy as np
import matplotlib.pyplot as plt
import math

class Environment:
    
    
    # Initializing the Environment, along with variables describing it
    # Initializing agent position and agent.goal
    # Env - 2D matrix represeting the envorinment, numRows x numCols size
    # state -  a vector where [0] contains the true position of car x* and 
    #          [1] contains the true speed of the car.
    # maxV - the maximum number of indeces the car can move in a given time step
    # minV - the min number of indeces the can can displace in 1 time step

    def __init__(self, numCols):
        
        numRows = 3
        self.Env = np.zeros((numRows,numCols))               
      
    # This method advances the state by one
    #  
    # INOUTS:
    #
    # state - the current state of the world
    #
    # OUTPUTS:
    # nextState - the next state of the world
                
    def step(self,state):
        
        nextState = state - 1           
        return nextState
    
    # This methods generates a new world, in other words a new state
    #
    # INPUTS:
    #
    # agent -  an agent to reset the Crossing
    #
    # OUTPUTS:
    #
    # state - the true, randomly generated state (time gap)
    def genWorld(self, agent):
        
        agent.resetCrossing()
        state = 1000
        bound = self.Env.shape[1]
        while (state > bound - 1) or (state < 1) :
            state = np.random.normal(0,bound/2)
            state = round(state)
            
        return state
        
    
    # This Method trains the agent by simulating the world exceution. Producing states that the agent uses
    # to observe and learn from/
    #
    # INPUTS:
    # numTrials - the number of trial to train on. Each trial consists of 
    #             the car appearing in a random location, and arriving at
    #             in Env[1,0]
    # fullObs - a boolean describing whether the agent sees the true state or
    #           a noisy approximation
    # agent -   the agent to receive training
    # timeInt - how many training trials to allow before miniTest() call
    # numTests - how many tests to perform in miniTest() call
    #
    # LOCAL VARS:
    #
    # done - a boolean than indicates whether a trial is complete or ongoing
    # reward - the reward the agetn observes
    # t -  the current time step, starting at 0
    # state - the current state of the world. Described as the current time gap
    # nextState - the next state of the world, the time gap after the car has moved
    # trial - each single trial where the total is numTrials. The trial ends when either the car
    #         or the agent have arrived at their destination
    # OUTPUTS:
    # trainScores, the scores of the agent as it trains
    def train(self, numTrials, fullObs, agent, timeInt, numTests):
        
        trainScores = [] 
        # n keeps track of how many times the agetn learns
        n = 0
        for trial in range(numTrials):
    
            # init state, arguement 0 is passed to step(), a new world is generated
            t = 0 # timestep in each trial
            state = self.genWorld(agent)
            done = False
            reward = 0
           # print("Trial " + str(trial) + " of " + str(numTrials)) 
            while not done:
                
                # agent chooses action based on the observed time-gap
                agent.selectAction(state, trial, fullObs)
                
                
                if agent.getAction() == agent.getPossibleActions()[1]:
                    # each time step, the crossing time to arrive at the goal is diminshed by 1
                    agent.cross()
                
                # world exceucutes action
                nextState = self.step(state)

                # agent observes next state, and reward
                reward,done = self.getReward(agent, nextState, done)   
                # once reward is sampled increase n
                n = n + 1
                    
                # agent learns from this 
                agent.learn(reward, state, nextState, fullObs, n)
       
                # updates the current state
                state = nextState
                t = t + 1
                
            # update agent's score
            if (((trial + 1) % timeInt) == 0) and (trial != 0) :
                score = self.miniTest(numTests, fullObs, agent)
                trainScores.append(score)
           
                
        return trainScores     
                
    # This method observes the state and the next state, and determines the reward based
    # on what happebed to the agent
    #
    # INPUTS
    #
    # agent - the learning agent
    # nextState - the next stae observed
    # done   boolean describing whether the the trial has finished
    #        might be unneeded
    # Note, changing reward -1,1 will break the test performance tracking
    def getReward(self, agent, nextState, done):
        
        action  = agent.getAction()
        
        if action == 2:
            if agent.crossing > nextState:
                #print("Agent is going to get hit while crossing")
                reward = -1
            elif agent.crossing <= nextState:
               # print("agent is going to cross succesfully")
                reward = 1
            
            done = True
        else:
            if nextState == 0:
               #print("The agent has not crossed, the car arrived")
                done = True
    
            else:
               #print("The agent has not crossed, and the car has not arrived")
                done = False               
        
            reward = 0.01
            
        return reward, done
    
    # This method puts an agent to the test, no learning is done
    # no random action selection is done, no epsilon greedy strategy
    #
    # INPUTS:
    # 
    # numTrials - the total number of trials to perform the test
    # fullObs - whether the environment is fully observable
    # agent - the agent taking the test
    # timeInt - the time interval to collect agent data
    def test(self, numTrials, fullObs, agent, timeInt):
        
        numSuccess = 0 # the number of times in timeInt the agent reaches the goal
        numFail = 0
        
        for trial in range(numTrials):
    
            # init state, arguement 0 is passed to step(), a new world is generated
            t = 0 # timestep in each trial
            state = self.genWorld(agent)
            done = False
            
            #print("----------Beginning test " + str(trial) + " out of " + str(numTrials) + "----------")
            while not done:
                # agent osberves state
                # agent chooses action based on the observed time-gap
                agent.selectAction(state, math.inf, fullObs)
                
                if agent.getAction() == agent.getPossibleActions()[1]:
                    # each time step, the crossing time to arrive at the goal is diminshed by 1
                    agent.cross()
                
                # world exceucutes action
                nextState = self.step(state)
    
                # agent observes next state, and reward
                # Check if the car arrived at the crossing at the current time step              
                reward,done = self.getReward(agent, nextState, done) 
               
                # updates the current state
                state = nextState
                t = t + 1
                
                # track number of fail/success
                if reward == 1:
                    numSuccess = numSuccess + 1
                elif reward == -1:
                    numFail = numFail +1
                    
                # update agent's score
            if (((trial + 1) % timeInt) == 0):
                score = numSuccess / (numSuccess + numFail + 0.001)
                agent.updateScores(score, trial, timeInt)
                numSuccess = 0 
                numFail = 0
                
    # plots the scores in agent's score array
    def plotScores(self, numTrials, agentList, Labels):
        
        i = 0
        while i < len(agentList):
            
            data = agentList[i].getScores()
            plt.plot(data[1,:] , data[0,:])
            i = i + 1
            
        plt.ylim(0.0,1.0)
        plt.xlabel( "time interval")
        plt.ylabel("Score")
        plt.title("Testing of agent for " + str(numTrials) + " trials")
        plt.legend(Labels)
        plt.show()
    
    # This method performs a test on the agent, designed to only do a handful of trials
    # and is meant to be performed during training, to see training progress
    #
    # INPUTS:
    # 
    # numTrials - the total number of trials to perform the test
    # fullObs - whether the environment is fully observable
    # agent - the agent taking the test
    # 
    # OUTPUTS:
    # score - the score of this mini test
    def miniTest(self, numTrials, fullObs, agent):
        
        numSuccess = 0 # the number of times in timeInt the agent reaches the goal
        numFail = 0
        
        for trial in range(numTrials):
    
            # init state, arguement 0 is passed to step(), a new world is generated
            t = 0 # timestep in each trial
            state = self.genWorld(agent)
            done = False
            
            while not done:
                # agent osberves state
                # agent chooses action based on the observed time-gap
                agent.selectAction(state, math.inf, fullObs)
                
                if agent.getAction() == agent.getPossibleActions()[1]:
                    # each time step, the crossing time to arrive at the goal is diminshed by 1
                    agent.cross()
                
                # world exceucutes action
                nextState = self.step(state)
    
                # agent observes next state, and reward
                # Check if the car arrived at the crossing at the current time step              
                reward,done = self.getReward(agent, nextState, done) 
               
                # updates the current state
                state = nextState
                t = t + 1
                
            
              # track number of fail/success
            if reward == 1:
                    numSuccess = numSuccess + 1
            elif reward == -1:
                    numFail = numFail +1
                    
            
        score = numFail / numTrials
        return (1-score)
        
                 
    # plots the given training scores
    def plotTrainScores(self, numTrials, scores, labels):
        
        x = np.arange(0, len(scores[0]))
        for score in scores:
                      
            plt.plot(x , score)
                    
        plt.ylim(0.0,1.0)
        plt.xlabel( "test")
        plt.ylabel("Score")
        plt.title("Training of agent for " + str(numTrials) + " trials")
        plt.legend(labels)
        plt.show()       
        