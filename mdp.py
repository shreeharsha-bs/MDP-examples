# Markov decision process setup
import numpy as np

random_state = 42
np.random.seed(random_state)

class MDP():
    '''
    Setup an MDP environment with the following requirements:
    S - set of finite states
    A - set of finite actions
    P - Transition probability matrix for each state action pair
    R - Reward for each state action pair
    gamma - discount factor
    '''
    def __init__(self, S, A, P, gamma):
        self.P = P
        self.S = S
        self.A = A
        self.gamma = gamma
        self.VK = [0 for i in range(len(self.S))]

    def reward(self, s, a):
        self.R = s+a**1.125
        return self.R

    def value_func(self, Aseq):
        k = 0
        for K in range(50):
            for s in self.S:
                if(k >= len(Aseq)):
                    return self.VK
                self.VK[s] = self.reward(s,Aseq[k]) + self.gamma*np.sum(P[s]*self.VK[s])
                k += 1

        return self.VK

class agent():
    '''
    This class is an agent who will try out different policies he has access to
    '''
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.V = 0

    def action_seq(self, l):
        return np.random.choice(self.A, size = l, replace = True)

    def get_Vf(self, mdp):
        self.V = mdp.value_func(self.action_seq(50))
        return self.V

if __name__ == '__main__':

    S = [0,1,2,3,4,5,6,7] # Set of states
    A = [1,2] # Two actions

    P = np.random.rand(len(S),len(S))
    P = P/P.sum(axis=1)[:,None] # Random Stochastic Transition matrix
    gamma = 0.1

    MDP1 = MDP(S, A, P, gamma)
    ag = agent(S, A)
