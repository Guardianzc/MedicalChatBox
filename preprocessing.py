import json
import pickle
import copy
import random
import pandas as pd
import openpyxl
with open('./data_20210412/医疗对话数据样本_20210412.json','r',encoding='utf8') as fr:
    data = json.load(fr)
wb = pd.read_excel(r'./data_20210412/第一阶段数据划分.xlsx', engine='openpyxl')
wb_dict = dict()
for i in wb._values:
    wb_dict[str(i[0])] = i[1]

class preprocessing(object):
    def __init__(self):
        pass
    def scrapy_preprocess(self,data):
        self.disease_set = []
        self.symptom_set = []
        self.disease_symptom = dict()
        for num, record in data.items():
            index = 1
            if record['diagnosis'] not in self.disease_symptom.keys():
                self.disease_symptom[record['diagnosis']] = dict()
                self.disease_symptom[record['diagnosis']]['index'] = index
                self.disease_symptom[record['diagnosis']]['Symptom'] = dict()
                index += 1
            self.disease_set.append(record['diagnosis'])
            for key,values in record['explicit_info'].items():
                if key in ['Symptom','Medical_Examination']:
                    for value in values:
                        self.symptom_set.append(value)
                        self.disease_symptom[record['diagnosis']]['Symptom'][value] = self.disease_symptom[record['diagnosis']]['Symptom'].get(value, 0) + 1
            for key,values in record['implicit_info'].items():
                for name, status in values.items():
                    if status != '2':
                        self.symptom_set.append(name)
                        self.disease_symptom[record['diagnosis']]['Symptom'][name] = self.disease_symptom[record['diagnosis']]['Symptom'].get(name, 0) + 1
        #self.symptom_set.append('disease')
        self.disease_set = list(set(self.disease_set))
        self.symptom_set = list(set(self.symptom_set))
        pass


    def disease_dumper(self,output_slot):
        
        disease_set={}
        for i in range(len(self.disease_set)):
            disease_set[self.disease_set[i]]=i
        pickle.dump(file=open(output_slot,'wb'),obj=disease_set)
        return disease_set
    
    def slot_dumper(self,output_slot):
        
        slot_set={}
        for i in range(len(self.symptom_set)):
            slot_set[self.symptom_set[i]]=i
        pickle.dump(file=open(output_slot,'wb'),obj=slot_set)
        return slot_set
    
    def disease_symptom_dumper(self,output_slot):
        pickle.dump(file=open(output_slot,'wb'),obj=self.disease_symptom)
        return self.disease_symptom

    def create_goaltest(self,data, wb_dict, output_slot):
        '''
        split the data into train samples and test samples
        '''
        goal_set_store = dict()
        goal_set_store['train'] = list()
        goal_set_store['dev'] = list()
        goal_set_store['test'] = list()
        #assert train_rate+test_rate==1
        for pid, episode in data.items():
            goal_set = dict()
            goal_set['consult_id'] = pid
            goal_set['disease_tag'] = episode['diagnosis']
            goal_set['goal'] = dict()
            goal_set['goal']['request_slots'] = {'disease':'UNK'}
            goal_set['goal']['explicit_inform_slots'] = dict()
            goal_set['goal']['implicit_inform_slots'] = dict()
            for key,values in episode['explicit_info'].items():
                if key in ['Symptom','Medical_Examination']:
                    for value in values:
                        goal_set['goal']['explicit_inform_slots'][value] = True
            for key,values in episode['implicit_info'].items():
                for name, status in values.items():
                    if status == '0':
                        goal_set['goal']['implicit_inform_slots'][name] = False
                    if status == '1':
                        goal_set['goal']['implicit_inform_slots'][name] = True
            goal_set_store[wb_dict[pid]].append(goal_set)

        random.shuffle(goal_set_store['train'])
        random.shuffle(goal_set_store['dev'])
        random.shuffle(goal_set_store['test'])
        pickle.dump(file=open(output_slot,'wb'),obj=goal_set_store)
        return goal_set_store



if __name__ == '__main__':
    preprocess = preprocessing()
    preprocess.scrapy_preprocess(data)
    goal = preprocess.create_goaltest(data, wb_dict, output_slot='./resource/goal_set.p')
    disease_set=preprocess.disease_dumper(output_slot='./resource/disease_set.p')
    slot_set=preprocess.slot_dumper(output_slot='./resource/slot_set.p')
    disease_symptom=preprocess.disease_symptom_dumper(output_slot='./resource/disease_symptom.p')
    pass