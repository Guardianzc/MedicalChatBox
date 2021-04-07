#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 18:49
# @Author  : Ting
import csv
import copy
import json
from collections import Counter
class normal_filter(object):
    def __init__(self):
        self.filter={
           'consult_id':None,
           "disease_tag":None,
           "goal":{"request_slots":{"disease": "UNK"},"implicit_inform_slots":{},"explicit_inform_slots":{}}
                }
        self.goal_filter=[]
        
    def implicit_extraction(self,conversation):
        for ii,diag in conversation.items():
            filters=copy.deepcopy(self.filter)
            implicit_inform_slots={}
            for sent in diag.values():
                slot=sent['ner']
                type1=sent['type']
                for i in range(len(slot)):
                    if slot[i]=='' or slot[i]==' ':
                        pass
                    else:
                        if slot[i] not in implicit_inform_slots.keys():
                            if type1[i]=='1':
                                implicit_inform_slots[slot[i]]=[True]
                            elif type1[i]=='2':
                                implicit_inform_slots[slot[i]]=[False]
                            else:
                                implicit_inform_slots[slot[i]]=['UNK']
                        else:
                            if type1[i]=='1':
                                implicit_inform_slots[slot[i]].append(True)
                            elif type1[i]=='2':
                                implicit_inform_slots[slot[i]].append(False)
                            else:
                                implicit_inform_slots[slot[i]].append('UNK')
                                
            for slots,values in implicit_inform_slots.items():
                slot_dict={}
                slot_dict[True]=values.count(True)
                slot_dict[False]=values.count(False)
                slot_dict['UNK']=values.count('UNK')
                if slot_dict[True]>slot_dict[False] and slot_dict[True]>slot_dict['UNK']:
                    implicit_inform_slots[slots]=True
                elif slot_dict[False]>slot_dict['UNK'] and slot_dict[False]>slot_dict[True]:
                    implicit_inform_slots[slots]=False
                else:
                    implicit_inform_slots[slots]='UNK'
                    
            filters['goal']['implicit_inform_slots']=implicit_inform_slots
            filters['consult_id']=str(ii)
            #print(str(ii))
            filters['disease_tag']=self.disease[str(ii)]
            self.goal_filter.append(filters)
            
                
    def explicit_extraction(self,report):
         for i,sent in enumerate(self.goal_filter):
             idx=sent['consult_id']
             #print(idx)
             slot=report[idx]['ner']
             explicit_inform_slots={}
             for k in slot:
                    if k=='' or k==' ':
                        pass
                    else:
                        k_splited=k.split('，')
                        for kk in k_splited:
                            explicit_inform_slots[kk]=True
             self.goal_filter[i]['goal']["explicit_inform_slots"]=explicit_inform_slots
         return self.goal_filter
                        
    def disease_tag(self,file):
        f=open(file,encoding='gb18030')
        data={}
        '''
        for line in f:
            try:
                line_seg=line.split(',')
                data[line_seg[3]]=line_seg[6]
            except:
                print(line)
        '''
        reader=csv.reader(f)
        data1 = [r for r in reader]
        for sent in data1:
            idx=sent[3]
            disease=sent[6]
            if disease in ['上呼吸道感染','小儿支气管炎','小儿消化不良','小儿腹泻']:
                data[idx]=disease
            else:
                pass
        #data.pop('咨询ID')
        self.disease=data
             









def read_csv(file):
    f = open(file, 'r', encoding='utf8')
    reader = csv.reader(f)
    data = [r for r in reader]
    #data=f.readlines()
    f.close()
    return data


# def read_conversation(file):
#     data = read_csv(file)
#     if data[0][0][0] == '\ufeff':
#         data[0][0] = data[0][0][1:]
#     conversations = dict()
#     conv = []
#     key = 'null'
#     for d in data:
#         if not d:
#             conv.append(d)
#             continue
#         else:
#             if d[0].strip() and not ''.join(d[1:]):
#                 if key != 'null':
#                     conversations[key] = conv
#                 key = int(d[0])
#                 conv = []
#             else:
#                 conv.append(d)
#     while len(conv)%4 != 0:
#         conv.append(['' for i in range(202)])
#     conversations[key] = conv
#     for key in conversations:
#         conv = conversations[key]
#         conversations[key] = dict()
#         for i in range(0,int(len(conv)/4)):
#             sent = dict()
#             sent['content'] = conv[i*4]     # 标注原文
#             sent['tag'] = conv[i*4+1]       # BIO标记
#             sent['type'] = conv[i*4+2]      # 症状是否出现 1肯定/2否定/3不确定
#             sent['ner'] = conv[i*4+3]       # 归一化标记
#             conversations[key][i] = sent
#     return conversations


def read_self_report(file):
    data = read_csv(file)
    if data[0][0][0] == '\ufeff':
        data[0][0] = data[0][0][1:]
    sr = dict()
    for i in range(0,len(data),3):
        key = str(data[i][0])
        sr[key] = dict()
        sr[key]['content'] = data[i]
        sr[key]['tag'] = data[i + 1]
        sr[key]['ner'] = data[i + 2]
    return sr


def write(file, data):
    f = open(file, 'w', newline='', encoding='utf8')
    f.write(data)
    f.close()


def write_csv(file, data):
    f = open(file, 'w', newline='', encoding='utf8')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()


def write_conversation(file, conv):
    result = []
    for k in conv:
        result.append([str(k)])
        for n in conv[k]:
            result.append(conv[k][n]['content'])
            result.append(conv[k][n]['tag'])
            result.append(conv[k][n]['ner'])
            result.append(conv[k][n]['type'])
            result.append(conv[k][n]['no.'])
    write_csv(file, result)


def append(file, data):
    f = open(file, 'a', newline='', encoding='utf8')
    f.write(data)
    f.close()


def read_conversation_with_check(file):
    data = read_csv(file)
    if data[0][0][0] == '\ufeff':
        data[0][0] = data[0][0][1:]
    conversations = dict()
    conv = []
    key = 'null'
    for d in data:
        if not d:
            conv.append(d)
            continue
        else:
            if d[0].strip() and not ''.join(d[1:]):
                if key != 'null':
                    conversations[key] = conv
                key = d[0].strip()
                conv = []
            else:
                conv.append(d)
    while len(conv) % 5 != 0:
        conv.append(['' for _ in range(202)])
    conversations[key] = conv
    for key in conversations:
        conv = conversations[key]
        conversations[key] = dict()
        for i in range(0,int(len(conv)/5)):
            sent = dict()
            sent['content'] = conv[i*5]     # 标注原文
            sent['tag'] = conv[i*5+1]       # BIO标记
            if '4' in conv[i*5+3]:
                print('关键词' + key + '下，第' + str(i+1) + '句， 第 3 行未标注')
            sent['type'] = conv[i*5+3]      # 症状是否出现 1肯定/2否定/3不确定
            sent['ner'] = conv[i*5+2]       # 归一化标记
            if '*' in conv[i*5+4]:
                print('关键词' + key + '下，第' + str(i + 1) + '句， 第 4 行未标注')
            sent['no.'] = conv[i*5+4]       # 判断依据的行号
            conversations[key][i] = sent
    return conversations




ii='2'
file1='/Users/qianlong/Downloads/conversation.csv'
con1=read_conversation_with_check(file1)
file2='/Users/qianlong/Downloads/self_report.csv'
report=read_self_report(file2)
file3='/Users/qianlong/Downloads/raw_data.csv'
outputfile='/Users/qianlong/Downloads/goal.json'

goal=normal_filter()
goal.disease_tag(file3)
goal.implicit_extraction(con1)
goals=goal.explicit_extraction(report)

f1=open(outputfile,'w')
for sent in goals:
    f1.writelines(json.dumps(sent))
    f1.write('\n')
f1.close()


# file11='./bio_data/conversation/batch2/tagger'+ii+'.csv'
# con11=read_conversation_with_check(file11)
# goal=normal_filter()
# goal.disease_tag(file3)
# goal.implicit_extraction(con1)
# goals=goal.explicit_extraction(report)
#
# f1=open(outputfile,'a')
# for sent in goals:
#     f1.writelines(json.dumps(sent))
#     f1.write('\n')
# f1.close()


disease=[]
for sent in goals:
    disease.append(sent['disease_tag'])
c = Counter(disease)
print(c)
