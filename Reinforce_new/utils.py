import math
import numpy as np
import torch
import random
from torch.distributions import Categorical
def reb_generate(origin_state, origin_goal, status, reb_record, turn):

    origin_state_eps = origin_state + 1e-12
    forward = -origin_goal.mm(torch.log(origin_state_eps.t()))

    origin_state_back = 1 - origin_state + 1e-12
    back = -(1-origin_goal).mm(torch.log(origin_state_back.t()))

    for i in range(len(status)):
        if status[i] != 4:
             reb_record[turn][i] = forward[i,i] + back[i,i]

    return 0

def env_generate(Symptom, Test, Disease, status, env_record, turn):
    for i in range(len(status)):
        if status[i] == 1:
            state = Symptom[i][Symptom[i] != 0].unsqueeze(0)
            env_record[turn][i] = state.mm(torch.log(state).t())
        elif status[i] == 2:
            state = Test[i][Test[i] != 0].unsqueeze(0)
            env_record[turn][i] = state.mm(torch.log(state).t())        
        elif status[i] == 3:
            state = Disease[i][Disease[i] != 0].unsqueeze(0)
            env_record[turn][i] = state.mm(torch.log(state).t())      
        else:
            env_record[turn][i] = 0
    return 0

def random_generate(Symptom, Test, Disease, status, mode, goal_disease, prob_record, turn,  sigma = 0.0056):
    Sample_matrix = []
    #prob_record = []
    if mode == 'train':
        for i in range(len(status)):
            if status[i] == 1:
            #assert prob.sum() == 1
                j = Categorical(Symptom[i])
                action = j.sample()
                prob_record[turn][i] =  -j.log_prob(action)
                Sample_matrix.append(action)
                

            elif status[i] == 2:
                sample = []
                prob = 0
                for j in range(len(Test[i])):
                    #choice = 0.5
                    choice = random.random()
                    # sum(log)
                    if choice < Test[i][j]:
                        sample.append(j)
                        prob = prob - torch.log(Test[i][j])
                    else:
                        prob = prob - torch.log(1 - Test[i][j])

                Sample_matrix.append(sample)   
                prob_record[turn][i] =  prob


            elif status[i] == 3:
                choice = random.random()
                expand = random.random()
                if expand < sigma:
                    Sample_matrix.append(goal_disease[i])
                    prob_record[turn][i] = torch.log(Disease[i][goal_disease[i]])
                else:
                #assert prob.sum() == 1
                    j = Categorical(Disease[i])
                    action = j.sample()
                    Sample_matrix.append(action)
                    prob_record[turn][i] =  -j.log_prob(action)
            else:
                Sample_matrix.append(0)
                prob_record[turn][i] = 0
    else:
        for i in range(len(status)):
            if status[i] == 1:
                j = torch.argmax(Symptom[i])
                Sample_matrix.append(j)
                prob_record[turn][i] =  torch.log(Symptom[i][j])
                
            elif status[i] == 2:
                sample = []
                prob = 0
                for j in range(len(Test[i])):
                    choice = 0.5
                    if choice < Test[i][j]:
                        sample.append(j)
                        prob = prob + torch.log(Test[i][j])
                    else:
                        prob = prob + torch.log(1 - Test[i][j])
                Sample_matrix.append(sample)  
                prob_record[turn][i] =  prob
      

            elif status[i] == 3:
                j = torch.argmax(Disease[i])
                Sample_matrix.append(j)
                prob_record[turn][i] =  torch.log(Disease[i][j])
               

            else:
                Sample_matrix.append(0)
                prob_record[turn][i] =  0
    return Sample_matrix