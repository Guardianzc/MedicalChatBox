# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:47:05 2018

@author: DELL
"""

import numpy as np
import os,sys
import pickle
import copy
import pandas as pd
sys.path.append(os.getcwd().replace('/resource/tagger2',''))
from preprocess.label.preprocess_label import GoalDumper
from sklearn.svm import SVC
from sklearn import svm
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate

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

def svm_model(dataset,min_count,target,svm_c):
    index_len=600
    slots_x=np.zeros((index_len,sum(sum(abs(dataset),1)>min_count)))  
    count=0       
    for i,value in enumerate(sum(dataset,1)):
        if value<=min_count:
            pass
        else:
            slots_x[:,count]=dataset[:,i]
            count+=1
            
    slots_input=pd.DataFrame(index=range(index_len))
    for col in  range(slots_x.shape[1]):
        column=slots_x[:,col]
        column_mod=[]
        for j in column:
            if j==1:
                column_mod.append('yes')
            elif j==-1:
                column_mod.append('no')
            else:
                column_mod.append('UNK')
        slots_input[str(col)]=column_mod
    
    slots_input=pd.get_dummies(slots_input)
    target=np.array(target)
    #x_train, x_val, y_train, y_val = train_test_split(slots_input, disease_y, test_size=0.3, random_state=10)
    clf = svm.SVC(kernel='linear', C=svm_c)
    scores=[]
    for i in range(10):
        scores_exp = cross_validate(clf, slots_input, target, cv=5,scoring='accuracy',return_train_score=False)
        scores_tot=sum(scores_exp['test_score'])/5
        scores.append(scores_tot)
    return np.mean(scores)

goal_file = "./goal_batch2.json"
goal_dump_file = "./goal_set.p"
slots_dump_file = "./slot_set.p"
goal=GoalDumper(goal_file)
goal.dump(goal_dump_file)
goal_set=goal.goalset
goal.dump_slot(slots_dump_file)
#goal_set,slot_set=goal.set_return()
slot_set=pickle.load(open(slots_dump_file,'rb'))
slot_set.pop('disease')
disease_symptom=goal.disease_symptom
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





        
        
        
score_tot_exp=svm_model(dataset=slots_exp,min_count=0,target=disease_y,svm_c=10)    
score_tot_all=svm_model(dataset=slots_all,min_count=0,target=disease_y,svm_c=10)    
'''
slots_x_all=np.zeros((len(total_set),sum(sum(abs(slots_all),1)>5)))  
count=0       
for i,value in enumerate(sum(slots_all,1)):
    if value<=5:
        pass
    else:
        slots_x_all[:,count]=slots_all[:,i]
        count+=1
        
slots_input_all=pd.DataFrame(index=range(len(total_set)))
for col in  range(slots_x_all.shape[1]):
    column=slots_x_all[:,col]
    column_mod=[]
    for j in column:
        if j==1:
            column_mod.append('yes')
        elif j==-1:
            column_mod.append('no')
        else:
            column_mod.append('UNK')
    slots_input_all[str(col)]=column_mod
    
slots_input_all=pd.get_dummies(slots_input_all)
disease_y=np.array(disease_y)
#x_train, x_val, y_train, y_val = train_test_split(slots_input, disease_y, test_size=0.3, random_state=10)
clf = svm.SVC(kernel='linear', C=5)

scores_all = cross_validate(clf, slots_input_all, disease_y, cv=5,scoring='accuracy',return_train_score=False)
scores_tot_all=sum(scores_all['test_score'])/5
'''


   
    
    
    