import json
import pickle
import copy
import random
with open('医疗对话数据样本_20210404.json','r',encoding='utf8') as fr:
    data = json.load(fr)


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
                implicit_symptom_set = values.split(',')
                for implicit_symptom in implicit_symptom_set:
                    if implicit_symptom != '':
                        value = implicit_symptom[:-2:]
                        self.symptom_set.append(value)
                        self.disease_symptom[record['diagnosis']]['Symptom'][value] = self.disease_symptom[record['diagnosis']]['Symptom'].get(value, 0) + 1
        self.symptom_set.append('disease')
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

    def create_goaltest(self,data, output_slot,train_rate=0.8,test_rate=0.2):
        '''
        split the data into train samples and test samples
        '''
        goal_set_sep = dict()
        goal_set_store = list()
        assert train_rate+test_rate==1
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
                implicit_symptom_set = values.split(',')
                for implicit_symptom in implicit_symptom_set:
                    if implicit_symptom != '':
                        value = implicit_symptom[:-2:]
                        goal_set['goal']['implicit_inform_slots'][value] = True
            goal_set_store.append(goal_set)

        random.shuffle(goal_set_store)
        train_len = int(len(goal_set_store) * train_rate)
        goal_set_sep['train'] = goal_set_store[:train_len:]
        goal_set_sep['test'] = goal_set_store[train_len::]
        pickle.dump(file=open(output_slot,'wb'),obj=goal_set_sep)
        return goal_set_sep



if __name__ == '__main__':
    preprocess = preprocessing()
    preprocess.scrapy_preprocess(data)
    goal = preprocess.create_goaltest(data, output_slot='./resource/goal_set.p')
    disease_set=preprocess.disease_dumper(output_slot='./resource/disease_set.p')
    slot_set=preprocess.slot_dumper(output_slot='./resource/slot_set.p')
    disease_symptom=preprocess.disease_symptom_dumper(output_slot='./resource/disease_symptom.p')
    