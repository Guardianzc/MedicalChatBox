# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:47:05 2018

@author: DELL
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os,sys
import pickle
import copy
import pandas as pd
sys.path.append(os.getcwd().replace('/resource/tagger2',''))
from preprocess.label.preprocess_label import GoalDumper
from sklearn.svm import SVC
from sklearn import svm
import torch
import argparse
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
import random
def disease_symptom_clip(disease_symptom, denominator):
    """
    Keep the top min(symptom_num, max_turn//denominator) for each disease, and the related symptoms are sorted
    descendent according to their frequencies.

    Args:
        disease_symptom: a dict, key is the names of diseases, and the corresponding value is a dict too which
            contains the index of this disease and the related symptoms.
        denominator: int, the number of symptoms for each diseases is  max_turn // denominator.
        parameter: the super-parameter.

    Returns:
         and dict, whose keys are the names of diseases, and the values are dicts too with two keys: {'index', symptom}
    """
    max_turn = 22
    temp_disease_symptom = copy.deepcopy(disease_symptom)
    for key, value in disease_symptom.items():
        symptom_list = sorted(value['symptom'].items(),key = lambda x:x[1],reverse = True)
        symptom_list = [v[0] for v in symptom_list]
        symptom_list = symptom_list[0:min(len(symptom_list), int(max_turn / float(denominator)))]
        temp_disease_symptom[key]['symptom'] = symptom_list
    #print('\n',disease_symptom)
    #print('\n',temp_disease_symptom)
    return temp_disease_symptom

def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            #loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))
    return model
    
def valid(x, y, model, args):
    count = 0
    for i in range(x.shape[0]):
        x_ten = torch.Tensor([x[i, :]]).to(args.device)
        y_prob = model(x_ten).squeeze()
        if torch.argmax(y_prob) == np.argmax(y[i, :]):
            count += 1
    return count / x.shape[0]

slots_set_file = '/root/Downloads/MeicalChatbot-HRL-master/src/data/Fudan-Medical-Dialogue2/synthetic_dataset///slot_set.p'
goal_set_file = '/root/Downloads/MeicalChatbot-HRL-master/src/data/Fudan-Medical-Dialogue2/synthetic_dataset///goal_set.p'
disease_symptom_file = '/root/Downloads/MeicalChatbot-HRL-master/src/data/Fudan-Medical-Dialogue2/synthetic_dataset///disease_symptom.p'
goal_set = pickle.load(open(goal_set_file,'rb'))
slot_set=pickle.load(open(slots_set_file,'rb'))
slot_set.pop('disease')
disease_symptom= pickle.load(open(disease_symptom_file, 'rb'))
disease_symptom1=disease_symptom_clip(disease_symptom,2)
#slot_set.pop('发热39度3')
#slot_set.pop('发热37.7至38.4度')

disease_y=[]
total_set=copy.deepcopy(goal_set['train'])
total_set.extend(goal_set['test'])
slots_exp=np.zeros((len(total_set),len(slot_set)))
slots_all=np.zeros((len(total_set),len(slot_set)))

#slots_exp=pd.DataFrame(slots_exp,columns=slot_set.keys())
#slots_all=pd.DataFrame(slots_all,columns=slot_set.keys())
for i,dialogue in enumerate(total_set):
    tag=dialogue['disease_tag']
    tag_group=disease_symptom1[tag]['symptom']
    disease_y.append(tag)
    goal=dialogue['goal']
    explicit=goal['explicit_inform_slots']
    implicit=goal['implicit_inform_slots']
    for slot,value in implicit.items():
        try:
            slot_id=slot_set[slot]
            if value==True:
                slots_all[i,slot_id]='1'
            if value==False:
                slots_all[i,slot_id]='-1'
        except:
            pass
    for exp_slot,value in explicit.items():
        try:
            slot_id=slot_set[exp_slot]
            if value==True:
                slots_exp[i,slot_id]='1'
                slots_all[i,slot_id]='1'
            if value==False:
                slots_exp[i,slot_id]='-1'
                slots_all[i,slot_id]='-1'
        except:
            pass
    

disease_list = list(set(disease_y))
length_disease = len(disease_list)
disease_total = np.zeros((len(total_set),length_disease))
i = 0
for disease_index in disease_y:
    index = disease_list.index(disease_index)
    disease_total[i, index] = 1
    i += 1

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def SVM_torch(x, y, mode = 'exp'):
    length = x.shape[0]
    length_list = list(range(length))
    random.shuffle(length_list)
    print(mode)
    max_score = 0
    model = nn.Linear(len(slot_set), length_disease)
    model.to(args.device)
    for i in range(10):
        train_list = x[length_list[:27000:], :]
        valid_list = x[length_list[27000::], :]
        train_goal = y[length_list[:27000:], :]
        valid_goal = y[length_list[27000::], :]

        model = train(train_list, train_goal, model, args)     
        score = valid(valid_list, valid_goal, model, args)
        if score > max_score:
            max_score = score
        print(score)
    return max_score





score1 = SVM_torch(slots_exp, disease_total, 'exp')
score2 = SVM_torch(slots_all, disease_total, 'all')
print(score1, score2)