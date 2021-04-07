# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:26:03 2019

@author: DELL
"""

import pickle
import os
import copy

#file0='./src/data/real_world'
file0='./src/data/simulated/label13'
disease_symptom = pickle.load(open(file0+'/disease_symptom.p','rb'))
slot_set = pickle.load(open(file0+'/slot_set.p','rb'))
slot_set.pop('disease')
disease2id = {}
for disease,value in disease_symptom.items():
    index = value['index']
    disease2id[disease] = index
sorts = sorted(disease2id.items(),key = lambda x:x[1],reverse = False)
diseases = [x[0] for x in sorts]
id2disease = {value:key for key,value in disease2id.items()}
id2slot = {value:key for key,value in slot_set.items()}

dirs = os.listdir('./visit/')
result_dir = os.path.join('./visit',dirs[-1])

result = pickle.load(open(result_dir,'rb'))

symptom_disease = {}
for s in slot_set.values():
    symptom_disease[id2slot[s]] = {}
    for d in diseases:
        symptom_disease[id2slot[s]][d]=0

temp = copy.deepcopy(result['disease'])
for d,values in temp.items():
    for slot,count in values.items():
        if slot>=(len(slot_set)):
            pass
        else:
            symptom_disease[id2slot[slot]][id2disease[d]]+=count

with open('./symptom_disease_final_s.csv','w') as f:
    f.writelines(','+','.join(diseases))
    f.writelines('\n')
    for d,s_dict in symptom_disease.items():
        s_value = list(s_dict.values())
        s_value = [str(x) for x in s_value]
        f.writelines(d+','+','.join(s_value))
        f.writelines('\n')
      
#with open('./resource/MedicalChatbotMultiAgent1/visit')
