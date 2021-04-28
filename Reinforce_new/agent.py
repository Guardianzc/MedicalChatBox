from network import DQNEncoder, DQNSymptomDecoder, DQNTestDecoder, DQNDiseaseDecoder, DQNAuxiliaryDecoder, init_weights
import torch
import random
import pickle
import math
import copy
from torch import optim
from tensorboardX import SummaryWriter
from dataloader import get_loader
import sys
import multiprocessing as mp
import math
import numpy as np
import utils


def progress_bar(bar_len, SR, avg_turn, avg_obj, avg_reward, avg_env, SRT, ATT, avg_reward_t,  best_SR,  currentNumber, wholeNumber):
    # 20, success_rate, avg_turns, avg_object, success_rate_test, avg_turns_test, best_success_rate_test, best_avg_turns_test, i, simulate_epoch_number
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mSR\033[0m = %3f \033[36;1mAvg_turn\033[0m= %3f \033[36;1mAvg_obj\033[0m= %3f \033[36;1mAvg_reward\033[0m= %3f \033[36;1mAvg_env\033[0m= %3f \033[33;1mSR_t\033[0m= %3f \033[33;1mAT_t\033[0m= %3f \033[33;1mreward_t\033[0m= %3f  \033[33;1mBest_SR\033[0m= %3f  \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', SR, avg_turn, avg_obj, avg_reward, avg_env, SRT, ATT, avg_reward_t,  best_SR))
    sys.stdout.flush()


class Agent(object):
    def __init__(self, slot_set, test_set, disease_set, parameter):
        self.num_slot = len(slot_set)
        self.num_test = len(test_set)
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        self.test_set = test_set
        self.disease_set = disease_set
        self.goal_test_path = '..//resource//goal_set.p'
        self.batch_size = parameter['batch_size']
        self.max_turn = parameter['max_turn']
        self.n = parameter['n']
        self.l = parameter['lambda']
        self.c = parameter['c']
        self.m = parameter['m']
        self.beta = parameter['beta']
        self.k = parameter['k']
        self.gamma = parameter['gamma']
        self.sigma = parameter['sigma']
        self.load_model = parameter['load']
        
        self.parameter = parameter
        self.num_cores = int(mp.cpu_count())
        self.build_model()
        print("本地计算机有: " + str(self.num_cores) + " 核心")

    def build_model(self, mode = 'normal'):
        self.Encoder = DQNEncoder(input_size= self.num_slot + self.num_test)
        self.SymptomDecoder = DQNSymptomDecoder(output_size=self.num_slot+2)
        self.TestDecoder = DQNTestDecoder(output_size=self.num_test)
        self.DiseaseDecoder = DQNDiseaseDecoder(output_size=self.num_disease)
        self.AuxiliaryDecoder = DQNAuxiliaryDecoder(output_size = self.num_test + self.num_slot)
        self.optimizer_para = list(self.Encoder.parameters()) + list(self.SymptomDecoder.parameters()) + list(self.TestDecoder.parameters()) + list(self.AuxiliaryDecoder.parameters()) 
        self.optimizer = optim.Adam(self.optimizer_para, self.parameter['lr'])
        self.device = torch.device('cuda:' + str(self.parameter['cuda_idx']) if torch.cuda.is_available() else 'cpu')
        self.Encoder.to(self.device)
        self.SymptomDecoder.to(self.device)
        self.TestDecoder.to(self.device)
        self.DiseaseDecoder.to(self.device)
        self.AuxiliaryDecoder.to(self.device)
        init_weights(self.Encoder, mode)
        init_weights(self.SymptomDecoder, mode)
        init_weights(self.TestDecoder, mode)
        init_weights(self.DiseaseDecoder, mode)
        init_weights(self.AuxiliaryDecoder, mode)
        if self.load_model:
            self.load(self.parameter['model_savepath'] + '/newest/')
        

    def update_state(self, simulate, goal, goal_disease, state, status, reward, mode, flag, turn):
        change = torch.zeros_like(state).to(self.device)
        for i in range(len(status)):
            
            if status[i] == 1:
                if turn >= self.max_turn: #or state[i][simulate[i]] != 0:
                    status[i] = 4
                    reward[turn][i] = self.n 
                                
                #elif state[i][simulate[i]] != 0:
                #    status[i] = 4
                #    reward[turn][i] = self.n                  
                
                else:
                    if simulate[i] == self.num_slot:
                        status[i] = 2
                    elif simulate[i] == self.num_slot + 1:
                        status[i] = 3
                    else:
                        if state[i][simulate[i]] == 0:
                            change[i][simulate[i]] = goal[i][simulate[i]] * 2 - 1
                            reward[turn][i] = self.l * len(torch.where(change[i] > 0)[0])
                        #state[i][simulate[i]] = goal[i][simulate[i]] * 2 - 1
            elif status[i] == 2:
                for j in simulate[i]:
                    if state[i][j + self.num_slot] != 0:
                    #state[i][j + self.num_slot] = goal[i][j + self.num_slot] * 2 - 1
                        change[i][j + self.num_slot] = goal[i][j + self.num_slot] * 2 - 1
                reward[turn][i] = self.c * len(simulate[i]) + self.l * len(torch.where(change[i] > 0)[0])
                status[i] = 3
            
            elif status[i] == 3:
                if simulate[i]  == goal_disease[i]:
                    reward[turn][i] = self.m #+ self.l * len(torch.where(state[i] > 0)[0])
                    flag[i] = 1
                else:
                    reward[turn][i] = self.n #+ self.l * len(torch.where(state[i] > 0)[0])
                    flag[i] = 0
                status[i] = 4
            #state[i] = state[i] + change
        return change

    def simulate(self, origin_state, goal, goal_disease, mode):
        # goal : vector
        stop_test = self.num_slot
        stop_disease = self.num_slot + 1
        
        turn = 0
        state = origin_state
        length = origin_state.shape[0]
        #####change
        reb_record = torch.zeros((self.max_turn+2, length))
        objective_list = torch.zeros_like(reb_record)
        env_record = torch.zeros_like(reb_record)
        reward_total = 0
        reward_cost = 0
        flag = torch.zeros(length)
        reward_record = torch.zeros_like(reb_record)
        prob_record = torch.zeros_like(reb_record)
        status =  torch.ones(length)
        ######## Symptom stage ########

        while (turn < self.max_turn + 2):
            
            # copy
            Encoder_feature = self.Encoder(state)
            Symptom = self.SymptomDecoder(Encoder_feature)
            Test = self.TestDecoder(Encoder_feature)
            Disease = self.DiseaseDecoder(Encoder_feature)
            Auxiliary = self.AuxiliaryDecoder(Encoder_feature)
            
            
            simulate = utils.random_generate(Symptom, Test, Disease, status, mode, goal_disease,  prob_record, turn)

            utils.reb_generate(Auxiliary, goal, status, reb_record, turn)
            utils.env_generate(Symptom, Test, Disease, status, env_record, turn)

            change = self.update_state(simulate, goal, goal_disease, state, status, reward_record, mode, flag ,turn)
            # change is necessary for not occuring "inplace error"
            state = state + change
            turn += 1

        reward_gamma_list = torch.zeros_like(reward_record)

        for i in reversed(range(reward_record.shape[0]-1)):
            reward_record[i] = reward_record[i+1] * self.gamma + reward_record[i] 


        #prob_record = prob_record + 1e-12

        return_matrix =  reward_record +  self.beta * env_record - self.k * reb_record
        '''
        mask_matrix = return_matrix != 0
        mean = return_matrix[return_matrix != 0].mean().item()
        std = return_matrix[return_matrix != 0].std().item()
        normalize_matrix = ((return_matrix - mean) / std).mul(mask_matrix)
        '''
        objective_list = prob_record.mul(return_matrix)
        return objective_list, flag, status,  reward_record, env_record.sum()
    
    def train_network(self, objective_value, eps = 1e-10):

        self.Encoder.zero_grad()
        self.SymptomDecoder.zero_grad()
        self.TestDecoder.zero_grad()
        self.DiseaseDecoder.zero_grad()
        self.AuxiliaryDecoder.zero_grad()
        #backward_value = -(objective_value - objective_episode_norm.mean()) / (objective_episode_norm.std())
        backward_value = -objective_value
        with torch.autograd.set_detect_anomaly(True):
            backward_value.backward()    
        self.optimizer.step()	
        pass
    
    def simulation_epoch(self, mode, epoch, simulate_epoch_number):
        if mode == 'train':
            dataset = get_loader(self.slot_set, self.test_set, self.disease_set, self.goal_test_path, batch_size=self.batch_size, mode = 'train')
        else:
            dataset = get_loader(self.slot_set, self.test_set, self.disease_set, self.goal_test_path, batch_size=self.batch_size, mode = 'test')
        success_count = 0
        total_object = 0

        total_rewards = 0
        total_env = 0
        length = 0
        total_simulate = 0

        # 多核运算
        
        #pool = mp.Pool(self.num_cores)
        for i, (origin_state, goal, goal_disease) in enumerate(dataset):
            temp_object = 0

            origin_state = origin_state.to(self.device)
            goal = goal.to(self.device)
            goal_disease = goal_disease.to(self.device)
            
            objective_list, flag, turn, reward_total, env_total = self.simulate(origin_state, goal, goal_disease, mode)
            #  objective_list, flag, status,  reward_record.sum(), env_record.sum()
            num_simulate = (reward_total != 0).sum().item()

            total_simulate = total_simulate + num_simulate
            length += origin_state.shape[0]
            total_object = total_object + objective_list.sum().item()
            total_rewards = total_rewards + reward_total.sum().item()
            total_env = total_env +  env_total.sum().item()
            
            temp_object = objective_list.sum()
            
            success_count += flag.sum().item()

            if mode == 'train':
                self.train_network(temp_object)
                progress_bar(10, self.success_rate, self.avg_turns, self.avg_object, self.avg_reward, self.avg_envs, self.success_rate_test, self.avg_turns_test, self.avg_reward_test, self.best_success_rate_test, i + epoch * len(dataset), simulate_epoch_number* len(dataset))
                self.save(self.parameter['model_savepath'] + '/newest/')
            
            #print(i)
        return success_count/length, float(total_simulate)/length, total_object/total_simulate, total_rewards/total_simulate, total_env/total_simulate
    
    def train(self, simulate_epoch_number):
        writer = SummaryWriter('logs')
        self.best_success_rate_test = 0
        self.best_avg_turns_test = 0
        self.success_rate = self.avg_turns = self.avg_object = self.avg_reward = self.avg_envs = 0
        self.success_rate_test = self.avg_turns_test = self.avg_object_test = self.avg_reward_test = self.avg_envs_test = 0
        for epoch in range(simulate_epoch_number):

            self.success_rate, self.avg_turns, self.avg_object, self.avg_reward, self.avg_envs = self.simulation_epoch(mode = 'train', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            self.success_rate_test, self.avg_turns_test, self.avg_object_test, self.avg_reward_test, self.avg_envs_test = self.simulation_epoch(mode = 'test', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            if self.best_success_rate_test < self.success_rate_test:
                self.best_success_rate_test = self.success_rate_test
                self.best_avg_turns_test = self.avg_turns_test
                self.save(self.parameter['model_savepath'])
            
            # write
            writer.add_scalar('success_rate', self.success_rate, global_step=epoch)
            writer.add_scalar('avg_turns', self.avg_turns, global_step=epoch)
            writer.add_scalar('avg_object', self.avg_object, global_step=epoch)
            writer.add_scalar('avg_reward', self.avg_reward, global_step=epoch)
            writer.add_scalar('avg_envs', self.avg_envs, global_step=epoch)
            writer.add_scalar('success_rate_test', self.success_rate_test, global_step=epoch)
            writer.add_scalar('avg_turns_test', self.avg_turns_test, global_step=epoch)
            writer.add_scalar('avg_reward_test', self.avg_reward_test, global_step=epoch)
            writer.add_scalar('avg_envs_test', self.avg_envs_test, global_step=epoch)
            
        return best_success_rate_test
    
    def save(self, path):
        Encoder_path = path + 'Encoder.pkl'
        SymptomDecoder_path = path + 'Symptom_Decoder.pkl'
        TestDecoder_path = path + 'Test_Decoder.pkl'
        DiseaseDecoder_path = path + 'Disease_Decoder.pkl'
        AuxiliaryDecoder_path = path + 'Auxiliary_Decoder.pkl'

        torch.save(self.Encoder.state_dict(), Encoder_path)
        torch.save(self.SymptomDecoder.state_dict(), SymptomDecoder_path)
        torch.save(self.TestDecoder.state_dict(), TestDecoder_path)
        torch.save(self.DiseaseDecoder.state_dict(), DiseaseDecoder_path)
        torch.save(self.AuxiliaryDecoder.state_dict(), AuxiliaryDecoder_path)

    def load(self, path):
        Encoder_path = path + 'Encoder.pkl'
        SymptomDecoder_path = path + 'Symptom_Decoder.pkl'
        TestDecoder_path = path + 'Test_Decoder.pkl'
        DiseaseDecoder_path = path + 'Disease_Decoder.pkl'
        AuxiliaryDecoder_path = path + 'Auxiliary_Decoder.pkl'

        self.Encoder.load_state_dict(torch.load(Encoder_path))
        self.SymptomDecoder.load_state_dict(torch.load(SymptomDecoder_path))
        self.TestDecoder.load_state_dict(torch.load(TestDecoder_path))
        self.DiseaseDecoder.load_state_dict(torch.load(DiseaseDecoder_path))
        self.AuxiliaryDecoder.load_state_dict(torch.load(AuxiliaryDecoder_path))



        

