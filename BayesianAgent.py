# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:18:01 2020

This class represents a Bayesian agent, which uses Bayesian Q learning
to decide upon an action presented by a state

Qvalue sampling: The agents knowledge of the available rewards are represented
                 explicitely as probability distributions.
Bayesian Q-Learning: The agent has a probability distribution to represent it's uncertainty about
                     The Q-value of performing an action a at state s.
mu_s,a: the mean of a distribution of the total discounted reward when executing a at s. This is the Q-value of s,a
tau_s,a: the precision of the mean. also calculated as 1/varience_s,a

The Q-table in bayesian Q learning is contains the posteriors, and as rewards are seen, the posterios are updated.

use myopic-VPI sampling to choose an action, update using moment updating? with exponential forgetting

to learn more: write down the equations on paper. read up on normal-gamma distributions, then read the github paper
@author: Juan Rios
"""
import numpy as np
from scipy.special import gamma
from scipy.special import beta
import scipy.special as ss
from scipy.stats import t
import math


class BayesianAgent:
    
    # initializing the learning agent
    # agent.symbol - the representation of agent's position in environment
    # agent.Qvals - a matrix containing state-action que values. the possible states
    #               are all the possible time gaps(time to arrival) of the car, indexed as
    #               rows in Qvals. The actions are the columns on Qvals.
    # possActions - a list containing the possibel actions. 1 - means to stay. 2- means to cross
    # action - the current action planned by the agent and the current state.
    # crossTime - the timesteps the agent spends crossing the road. AKA the number of steps the agent
    #             spends at Env[1,0]. This value ranges from 2 - 5 seconds
    # possCrossTimes - the minimum and maximum time steps the agent takes to cross, min = 2 max = 5
    # crossing - determines wether the agent is crossing or not. if crossing = -1, the agent has not crossed yet/
    #            if crossing = 1, the agent is crossing and will be done in 1 time step. if crossing = 2, the agent 
    #            is crossing and will be done in 2 time steps and so on. if crossing = 0, the agent has arrived at the goal state
    # mode - the mode of learning for the agent. "Bayesian" or "Q-learning"
    # alpha - the learning rate of this agent, takes the value of learRate param
    # EGS - the epsilon greedy strategy of this agent, takes value of e
    # gamma - the discount rate of future rewards
    # scores - whenever an agent tests, every n trials the scores are calculated for that n number
    #          of trials, and appended to the scores array
    # numCols
    def __init__(self, crossTime, mode, learnRate, e, gamma, numCols):
         
         self.Symbol = 1
         self.Qvals = 0
         self.possActions = [1,2]
         self.action = 0       
         self.crossTime = crossTime        
         self.crossing = -1
         self.mode = mode         
         self.alpha = learnRate
         self.EGS = e
         self.gamma = gamma
         self.scores = np.zeros((2,0))
         self.initQvals(numCols) 
         
    # This method initializes the Q-value table for the agent
    # The Q-value is primarily a 2D array, where the row contains a possible state
    # and each column a possible action.
    # If the learbner is Bayesian, the 2D array is agumented to hold a 3rd dimension.
    # The 3rd dimension holds the 4-index tuple rho, where rho contains the parameters
    # describing the distribution of the Q value. rho = mu_0,lambda, alpha, beta
    # INPUTS:
    # numState: The number of possible states, which defines the first dimensional
    #           size of the Q-table
    # 
    def initQvals(self, numStates):
        
        if (self.mode == "Q-learning") or (self.mode == "Random"):    
            self.Qvals = np.zeros((numStates,len(self.possActions)))
        elif self.mode == "Bayesian":
            self.Qvals = np.zeros((numStates,len(self.possActions),4))
            # default the initial rho values as rho = (0,2,1.1,1)
            # 1.1 for alpha becayse at some point alpha - 1 = and want to avoid this 0
            self.Qvals[:,:,:] = (0,2,1.1,1) 
        else:
            print("There was an error in method initQvals")
            
    # This method computes the 'c' parameter when choosing the action
    # using Myopic VPI in Bayesian learning
    #
    # INPUTS:
    # actn - is a tuple rho, containing 4 parameters
    #        rho = mu_0, lmbda, alpha, beta
    # LOCAL VAR:
    # numerator, denominator, middle, exponent describe regions of the total formula, made for clarity
    # OUTPUTS:
    # c - the target value for c as calculated        
    def computeC(self, actn):
        
        mu = actn[0]
        lmbda = actn[1]
        alpha = actn[2]
        beta = actn[3]
                    
        numerator = alpha*math.sqrt(beta)
        denominator = (alpha - 0.5)*alpha*math.sqrt(2*lmbda)*ss.beta(alpha, 0.5)
        middle = (pow(mu,2)/(2*alpha)) + 1
        exponent = 0.5 - alpha
        
        
        c = pow(middle,exponent)*numerator/denominator
        return c
    # This method samples a mu from the normalGamma sitribution given the 
    # parameters stored in actn.
    #
    # INPUTS:
    # actn - is a tuple rho, containing mu_0, lmbda, alpha, beta
    # LOCAL VARS:
    # OUTPUTS:
    # mu - a sampled varible from normal
    # tau - a sampled variable from gamma
    def sampleNormalGamma(self, actn):
        mu_0 = actn[0]
        lmbda = actn[1]
        alpha = actn[2]
        beta = actn[3]
                
        tau = np.random.gamma(alpha, beta)
        mu = np.random.normal(mu_0, math.sqrt(1/(lmbda*tau)))
        
        return mu,tau        
    
   
    # This method the agent chooses an action based on the observed state.
    # The action can be chosen using Qvalue table, Bayesian Q value, or randomly.
    # IF Bayesian, the action is chosen using Myopic VPI
    #
    # INPUTS:
    # state: the current state observed by the agent, which the agent will take an action
    #
    # state -the state observed by the agent, this is the state the agent decides to act upon
    def selectAction(self, state, trialNum, fullObs):
        
        # if the agent has already chosen to cross, just return
        if self.crossing != -1:
            return
        
        # if not fully observable, choose to add noise to state
        if fullObs == False:
            
            state = self.distortState(state)
        
        # Selecting a random action. IF the agent is not crossing
        if self.mode == "Random":
         
            self.action =  round(np.random.uniform(self.possActions[0],self.possActions[1]))
              
        
        elif self.mode == "Q-learning":
            
            # get the epsRate based on the current trial number
            epsRate = self.EGS.getEpsilonRate(trialNum)
            
             # Explore
            if epsRate > np.random.uniform(0,1):
                self.action =  round(np.random.uniform(self.possActions[0],self.possActions[1]))
                       
            # Exploit
            else:
                # basically, if both values are 0, pick a random action
                if np.count_nonzero(self.Qvals[state,:]) == 0:
                    self.action = round(np.random.uniform(self.possActions[0], self.possActions[1]))
                else:
                    self.action = np.argmax(self.Qvals[state,:]) + 1
            
        elif self.mode == "Bayesian":
            bestVal = -math.inf
            # 0. compute the difference in expected value of each action
            diff = -abs(self.Qvals[state,0,0] - self.Qvals[state,1,0])
            # 1. Loop through each possible action at state 
            actnIdx = 0
            for actn in self.Qvals[state,:,:]:
                
                # 2. Compute c
                c = self.computeC(actn)
               
                # 3. we sample a mu and tau from the normalgamma dsitribution
                mu,tau = self.sampleNormalGamma(actn)

                # 4. Determine whether actn is the best actn, ie, has biggest mu_0
                # and evaluate the comulative distribution
                
                if actn[0] == max(self.Qvals[state,:,0]) and actnIdx == 0:
                    # if the actn is the first action, but also the best
                    Pr = t.cdf((self.Qvals[state,1,0] - self.Qvals[state,0,0])*math.sqrt(actn[1]*actn[2]/actn[3]), 2*actn[2])
                elif actnIdx == 0:
                    # else if actn is the first but not the best action
                    Pr = t.cdf((self.Qvals[state,0,0] - self.Qvals[state,1,0])*math.sqrt(actn[1]*actn[2]/actn[3]), 2*actn[2])
                    
                elif actn[0] == max(self.Qvals[state,:,0]) and actnIdx == 1:
                   # else if the actn is the second and it is the best
                    Pr = t.cdf((self.Qvals[state,0,0] - self.Qvals[state,1,0])*math.sqrt(actn[1]*actn[2]/actn[3]), 2*actn[2])
                    
                elif actnIdx == 1:
                # if current action is action 2, but it is not the best action
                   Pr = t.cdf((self.Qvals[state,1,0] - self.Qvals[state,0,0])*math.sqrt(actn[1]*actn[2]/actn[3]), 2*actn[2])
                else:
                    print("Could not compute a Pr value, none of the if/if else/ else statements evaluated true")
                # 5. calculate the VPI for the current action
                VPI = c + diff*Pr
                # 6. compute the value to maximize
                currValue = mu + VPI
                
                # 7. determine which action has the best currValue
                if currValue >= bestVal:
                    self.action = actnIdx + 1 # +1 shifts actions 0,1 to 1,2
                    bestVal = currValue
                    
                actnIdx = actnIdx + 1;
                
        if self.action == self.possActions[1]:
            self.crossing = self.crossTime + 1
            
            
            
  
    # This method should be called when the agent decides to cross, or is in the process of
    # crossing. The current crossing time
    def cross(self):    
        self.crossing = self.crossing - 1
        
    # This method returns the current action the agent has chosen    
    def getAction(self):
        return self.action
    # This method returns the agent's possible actions
    def getPossibleActions(self):
        return self.possActions
    # This method returns the agent's crossing value
    def isCrossing(self):
        return self.crossing
    # This method resets the agent to a state of not having crossed. Call this method before every trial
    def resetCrossing(self):
        self.crossing = -1
    # updates the scores array
    def updateScores(self, score, trial, timeInt):
        self.scores = np.append(self.scores, [[score], [trial/(timeInt-1)]], axis = 1)
    # return scores array    
    def getScores(self):
        return self.scores
    # add noise to the state
    def distortState(self, state):
        
        # chhose a random sigma
        sigma = np.random.uniform(0,100)
        
        if sigma > 87:
            distort = 2
        elif sigma > 70:
            distort = 1
        elif sigma > 30:
            distort = 0
        elif sigma > 13:
            distort = -1
        else:
            distort = -2
        
        state = state + distort
        
        if state < 1:
            state = 1
        elif state >= self.Qvals.shape[0]:
            state = self.Qvals.shape[0] - 1
            
        return state
            
        
    # This method is what allows the agent to learn, by updating the q table
    # random agents do not learn, thus will not be included in this method
    #
    # INPUTS: 
    # state - the state the agent observes and decides what action to take
    # nextStae - the state the car is in at the end of the time step
    # reward -  the reward observed by the agent. after executing its action
    # n - how many times reward is observed
    def learn(self, reward, state, nextState, fullObs, n):
        
        if fullObs == False:
            
            state = self.distortState(state)
            nextState = state - 1
        
        
        if (self.mode == "Q-learning"):
            
            # Bellman Equation    
            self.Qvals[state, self.action - 1] = self.Qvals[state, self.action - 1] + \
            self.alpha*(reward + self.gamma*np.max(self.Qvals[nextState, :]) - \
            self.Qvals[state, self.action - 1])
                
        elif (self.mode == "Bayesian"):
            
            # get nextState parameters        
            u0 = self.Qvals[nextState, self.action - 1, 0]
            lmbda = self.Qvals[nextState, self.action - 1, 1]
            alpha = self.Qvals[nextState, self.action - 1, 2]
            beta = self.Qvals[nextState, self.action - 1, 3]
            
            # Use momentum Updating
            M1 = (reward + self.gamma*u0)/n
            ER2 = ((lmbda)/(lmbda + 1)) * (beta / (alpha - 1)) * math.pow(u0, 2)
            M2 = (math.pow(reward, 2) + 2*reward*self.gamma*u0 + ER2*math.pow(self.gamma,2))/n
            
            # get current parameters
            u0 = self.Qvals[state, self.action - 1, 0]
            lmbda = self.Qvals[state, self.action - 1, 1]
            alpha = self.Qvals[state, self.action - 1, 2]
            beta = self.Qvals[state, self.action - 1, 3]
            
            
            # calculate new params
            lmbda_p = n + lmbda
            u0_p = (lmbda*u0 + n*M1)/lmbda_p
            alpha_p = alpha + 0.5*n
            beta_p = beta + 0.5*n*(M2 - math.pow(M1, 2)) + 0.5*n*lmbda*math.pow(M1 - u0, 2)/lmbda_p
                   
            # update Q-vals
            self.Qvals[state, self.action - 1, 0] = u0_p
            self.Qvals[state, self.action - 1, 1] = lmbda_p
            self.Qvals[state, self.action - 1, 2] = alpha_p
            self.Qvals[state, self.action - 1, 3] = beta_p
            
            
            
           
        
    
   
        
        