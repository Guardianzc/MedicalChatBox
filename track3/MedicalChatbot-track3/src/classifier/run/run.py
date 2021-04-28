# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
sys.path.append(os.getcwd().replace("src/classifier/run",""))
import numpy as np
import random
import torch
from src.classifier.symptom_as_feature.symptom_classifier import SymptomClassifier


parser = argparse.ArgumentParser()

parser.add_argument("--goal_set", dest="goal_set", type=str, default="./src/dialogue_system/data/dataset/label/goal_set.p", help='path and filename of user goal')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./src/dialogue_system/data/dataset/label/slot_set.p', help='path and filename of the slots set')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str, default="./src/dialogue_system/data/dataset/label/disease_symptom.p", help="path and filename of the disease_symptom file")
parser.add_argument("--train_feature", dest="train_feature", type=str, default="ex&im", help="only use explicit symptom for classification? ex:yes, ex&im:no")
parser.add_argument("--test_feature", dest="test_feature", type=str, default="ex&im", help="only use explicit symptom for testing? ex:yes, ex&im:no")
parser.add_argument("--model", dest="model", type=str, default="SVM", help="SVM or MLP")
parser.add_argument("--fold", dest="fold", type=bool, default=False, help="whether to use the split data or 5-fold all the data")
args = parser.parse_args()
parameter = vars(args)

def run():
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    hidden_size = parameter.get("hidden_size")

    print("##"*30+"\nSymptom as features\n"+"##"*30)
    pfold = parameter.get("fold")
    classifier = SymptomClassifier(goal_set=goal_set,symptom_set=slot_set,disease_symptom=disease_symptom,hidden_size=hidden_size,parameter=parameter,fold = pfold, fold_num=5)
    if parameter.get('model') =='MLP':
        classifier.MLP_train()
    else:
        classifier.SVM_train()

if __name__ == "__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(500)
    run()