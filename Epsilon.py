
"""
Created on Sat Apr 18 16:37:27 2020

This class manages the epsilon-greedy strategy, which is the system
for a Q-learning agent to choose between exploration and exploitation

Summary: the agent will have a high exploration rate
# at the beginning of training, and once it has trained, 
# the exploitation rate will in turn be high.

@author: Juan Rios
"""

import math

class Epsilon():
    # Initialize the epsilon
    # INPUT: 
    #
    # start, the starting number of episodes
    # end - the ending number of episodes
    # decayRate - the decay rate of epsilon
    def __init__(self, start, end, decayRate):
        self.start = start
        self.end = end
        self.decayRate = decayRate
    
    # This method returns the epsilon rate
    # INPUTS:
    #
    # currentStep - the time step at which this method is called
    def getEpsilonRate(self, currentStep):
        
        ER = self.end + (self.start - self.end) * \
            math.exp(-1. * currentStep * self.decayRate)
        return ER