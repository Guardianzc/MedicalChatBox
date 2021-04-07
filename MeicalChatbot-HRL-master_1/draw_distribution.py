# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:44:49 2019

@author: DELL
"""

import pickle

file0='./src/data/real_world'
#file0='./src/data/simulated/label13'
goal_set=pickle.load(open(file0+'/goal_set.p','rb'))
slot_set = pickle.load(open(file0+'/slot_set.p','rb'))
disease_symptom = pickle.load(open(file0+'/disease_symptom.p','rb'))
slot_set.pop('disease')
goals=[]
for goal in goal_set.values():
    goals += goal
    
diseases = list(set([x['disease_tag'] for x in goals]))
disease_symptom2 = {}
disease_symptom3 = {}
for i in diseases:
    disease_symptom2[i] = {}
    disease_symptom3[i] = {}
    
symptom_disease = {}
symptom_disease2 = {}
for s in slot_set.keys():
    symptom_disease[s] = {}
    symptom_disease2[s] = {}
    for d in diseases:
        symptom_disease[s][d]=0
        symptom_disease2[s][d] = 0
        
    

for a in goals:
    disease = a['disease_tag']
    explicit = a['goal']['explicit_inform_slots']
    implicit = a['goal']['implicit_inform_slots']
    dict_combined = dict( explicit, **implicit)
    for s,value in dict_combined.items():
        symptom_disease[s][disease]+=1
        if s not in disease_symptom2[disease].keys():
            disease_symptom2[disease][s]=1
        else:
            disease_symptom2[disease][s]+=1
        if value==True:
            symptom_disease2[s][disease]+=1
            if s not in disease_symptom3[disease].keys():
                disease_symptom3[disease][s]=1
            else:
                disease_symptom3[disease][s]+=1
            
with open('./symptom_disease2.csv','w') as f:
    f.writelines(','+','.join(diseases))
    f.writelines('\n')
    for d,s_dict in symptom_disease2.items():
        s_value = list(s_dict.values())
        s_value = [str(x) for x in s_value]
        f.writelines(d+','+','.join(s_value))
        f.writelines('\n')
