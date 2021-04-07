# -*- coding:utf-8 -*-
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
from agent import Agent
os.chdir(os.path.dirname(sys.argv[0]))
parser = argparse.ArgumentParser()
file0='..//resource//'
parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
parser.add_argument("--disease_set", dest="disease_set", type=str, default=file0+'/disease_set.p',help='path and filename of the disease set')
parser.add_argument("--test_set", dest="test_set", type=str, default=file0+'/test_set.p',help='path and filename of the action set')

parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")
parser.add_argument("--disease_test", dest="disease_test", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")

parser.add_argument("--train_mode", dest="train_mode", type=bool, default=False, help="Runing this code in training mode? [True, False]")
parser.add_argument("--load_old_model", dest="load", type=bool, default=False)
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=5000, help="The number of simulate epoch.")
parser.add_argument("--model_savepath", dest="model_savepath", type=str, default='./model_save/', help="The path for save model.")

parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="The batchsize.")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=8, help="The maxturn.")
parser.add_argument("--wrong_prediction_reward", dest="n", type=int, default=-0.7075)
parser.add_argument("--Abnormality_reward_factor", dest="lambda", type=int, default=0.1915)
parser.add_argument("--Medical_test_cost", dest="c", type=int, default=-0.0084)
parser.add_argument("--Correct_prediction_reward", dest="m", type=int, default=0.8743)
parser.add_argument("--Entropy_regularizer", dest="beta", type=int, default=0.0117)
parser.add_argument("--rebuilding_loss", dest="k", type=int, default=10)
parser.add_argument("--discount_factor", dest="gamma", type=int, default=0.99)
parser.add_argument("--expand_factor", dest="sigma", type=int, default=0.0156)

parser.add_argument("--cuda_idx", dest="cuda_idx", type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
args = parser.parse_args()
parameter = vars(args)

def run(parameter):
    time.sleep(2)
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    disease_set = pickle.load(file=open(parameter["disease_set"], "rb"))
    exam_set = pickle.load(file=open(parameter["test_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    disease_test = pickle.load(file=open(parameter["disease_test"], "rb"))
    train_mode = parameter.get("train_mode")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    agent = Agent(slot_set, exam_set, disease_set, parameter)


    if train_mode:
        best_success_rate_test = agent.train(simulate_epoch_number)
        print('SC = ', best_success_rate_test)
        
    else:
        agent.load(parameter['model_savepath'] + '/newest/')
        #agent.load(parameter['model_savepath'] )
        success_rate_test, avg_turns_test, avg_object_test = agent.simulation_epoch(mode = 'test', epoch = 0, simulate_epoch_number = 1)
        print(success_rate_test, avg_turns_test, avg_object_test)

if __name__ == '__main__':
    run(parameter)