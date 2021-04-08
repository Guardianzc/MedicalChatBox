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
        for num, record in data.items():
            self.disease_set.append(record['diagnosis'])
            for key,values in record['explicit_info'].items():
                if key in ['Symptom','Medical_Examination']:
                    for value in values:
                        self.symptom_set.append(value)
            for key,values in record['implicit_info'].items():
                implicit_symptom_set = values.split(',')
                for implicit_symptom in implicit_symptom_set:
                    if implicit_symptom != '':
                        self.symptom_set.append(implicit_symptom[:-2:])
        pass
        self.disease_set = list(set(self.disease_set))
        self.symptom_set = list(set(self.symptom_set))
        pass

    def create_goaltest(self, data, output_slot,train_rate=0.8,test_rate=0.2):
        assert train_rate+test_rate==1
        goal_test_store = list()
        for num, record in data.items():
            goal_test = dict()
            goal_test['consult_id'] = num
            goal_test['disease_tag'] = record['diagnosis']
            goal_test['goal'] = dict()
            goal_test['goal']['request_slots'] = {'disease':'UNK'}
            goal_test['goal']['implicit_inform_slots'] = dict()
            goal_test['goal']['explicit_inform_slots'] = dict()
            for key,values in record['explicit_info'].items():
                if key in ['Symptom','Medical_Examination']:
                    for value in values:
                        goal_test['goal']['explicit_inform_slots'][value] = True
            for key,values in record['implicit_info'].items():
                implicit_symptom_set = values.split(',')
                for implicit_symptom in implicit_symptom_set:
                    if implicit_symptom != '':
                        goal_test['goal']['implicit_inform_slots'][implicit_symptom] = True
            goal_test_store.append(goal_test)
        random.shuffle(goal_test_store)
        train_len=int(len(data)*train_rate)
        goal = dict()
        goal['train'] = goal_test_store[:train_len:]
        goal['test'] = goal_test_store[train_len::]
        pickle.dump(file=open(output_slot,'wb'),obj=goal)
                
            

    def disease_dumper(self,output_slot):
        
        disease_set={}
        for i in range(len(self.diseases)):
            disease_set[self.diseases[i]]=i
        pickle.dump(file=open(output_slot,'wb'),obj=disease_set)
        return disease_set
    
    def slot_dumper(self,output_slot):
        
        slot_set={}
        for i in range(len(self.symptom_all)):
            slot_set[self.symptom_all[i]]=i
        pickle.dump(file=open(output_slot,'wb'),obj=slot_set)
        return slot_set

    def train_test(self,n,train_rate=0.8,test_rate=0.2):
        '''
        split the data into train samples and test samples
        '''
        assert train_rate+test_rate==1
        data=self.synthetic_data(n,synthetic_for_all=True)
        train_len=len(data)*train_rate
        train=random.sample(data,int(train_len))
        test=copy.deepcopy(data)
        for i in train:
            test.remove(i)
        self.data_new={}
        self.data_new['train']=train
        self.data_new['test']=test
        self.data_new['validate']=[]
        return self.data_new


if __name__ == '__main__':
    preprocess = preprocessing()
    preprocess.scrapy_preprocess(data)
    preprocess.create_goaltest(data, output_slot='./resource/disease_set.p')
    disease_set=preprocess.disease_dumper(output_slot='./resource/disease_set.p')
    slot_set=preprocess.slot_dumper(output_slot='./resource/slot_set.p')
    
    