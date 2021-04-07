# -*- coding: utf8 -*-
"""
用于画learning curve的图，这里只是不同agent之间进行对比分析，不包含simulator的greedy程度。
"""

from __future__ import print_function
import argparse, json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import sys, os
sys.path.append(os.getcwd().replace('/src/utils',''))

sns.set(style="darkgrid")
sns.set(font_scale=1.4)

width = 8
height = 5.8
plt.figure(figsize=(width, height))

linewidth = 1.1


class DrawCurve(object):
    def __init__(self, params):
        self.params = params

    def read_performance_records(self, path):
        """ load the performance score (.json) file """
        performance = pickle.load(file=open(path, 'rb'))

        success_rate = []
        average_reward = []
        average_wrong_disease = []
        average_turn = []
        for index in range(0, len(performance.keys()),1):
            # print(performance[index].keys())
            success_rate.append(performance[index]["success_rate"])
            average_reward.append(performance[index]["average_reward"])
            #average_wrong_disease.append(performance[index]["match_rate2"])
            average_turn.append(performance[index]["average_turn"])

        smooth_num = 5
        d = [success_rate[i * smooth_num:i * smooth_num + smooth_num] for i in
             range(int(len(success_rate) / smooth_num))]

        success_rate_new = []
        cache = 0
        alpha = 0.8
        for i in d:
            cur = sum(i) / float(smooth_num)
            cache = cache * alpha + (1 - alpha) * cur
            success_rate_new.append(cache)
        return success_rate, success_rate[399]

    def get_mean_var(self, path,key_word_list=None, no_key_word_list=None):
        file_list = DrawCurve.get_dir_list(path=path, key_word_list=key_word_list, no_key_word_list=no_key_word_list)
        BBQ_datapoint = []
        data_point = []
        for file_name in file_list:
            data_list, data_scalar = self.read_performance_records(os.path.join(path,file_name))
            BBQ_datapoint.append(data_list)
            data_point.append(data_scalar)
            # BBQ_datapoint.append(self.read_performance_records(file_name,role,metric))
        min_len = min(len(i) for i in BBQ_datapoint)
        print([len(i) for i in BBQ_datapoint])
        data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        mean = np.mean(data, axis=0)
        var = np.std(data, axis=0)
        mean_data_point = np.mean(data_point)
        return mean, var, min_len, mean_data_point

    def plot(self):
        colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#a8d40f', '#b4546f',    '#6495ED', '#778899', '#48D1CC', '#00FA9A','#F4A460', '#8FBC8F','#C0C0C0']
        # colors = ['#FFB6C1', '#DB7093', '#DA70D6', '#8B008B','#4B0082', '#483D8B','#87CEFA']
        global_idx = 1500
        min_len_list = []
        ave_result = {}


        no_key_word_list = ['.DS_Store','.pdf','RID9']
        key_word_list = ['Agentdqn', '4999.p','wfrs88']
        mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        min_len_list.append(min_len)
        l1, = plt.plot(range(mean.shape[0]), mean, colors[0], label='baseline (after)', linewidth=linewidth,linestyle='-')
        plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[0], alpha=0.2)
        ave_result['RL-agent'] = mean_point

        no_key_word_list = ['.DS_Store','.pdf','RID9']
        key_word_list = ['AgentDQN', '9999.p','wfrs0']
        mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        min_len_list.append(min_len)
        l2, = plt.plot(range(mean.shape[0]), mean, colors[1], label='baseline (before)', linewidth=linewidth,linestyle='-')
        plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[1], alpha=0.2)
        ave_result['HRL-agent(var0, wc0, sdai0)'] = mean_point
        '''
        no_key_word_list = ['.DS_Store','.pdf']
        key_word_list = ['AgentWithGoal2', '4599.p','RID9']
        mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        min_len_list.append(min_len)
        l2, = plt.plot(range(mean.shape[0]), mean, colors[2], label='HRL-Goal, ex&im', linewidth=linewidth,linestyle='-')
        plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[2], alpha=0.2)
        ave_result['HRL-agent(var0, wc0, sdai0)'] = mean_point
        '''
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentDQN', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.2' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[2], label='Flat-DQN, wfrs0.2', linewidth=linewidth,linestyle='--')
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[2], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentDQN', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.5' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[3], label='Flat-DQN, wfrs0.5', linewidth=linewidth,linestyle='--')
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[3], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentDQN', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs1.0' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[4], label='Flat-DQN, wfrs1.0', linewidth=linewidth,linestyle='--')
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[4], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentDQN', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs2.0' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[5], label='Flat-DQN, wfrs2.0', linewidth=linewidth,linestyle='--')
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[5], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        #
        #
        #

        # no_key_word_list = ['.DS_Store','.pdf','RID9']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', 'RFNCY-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.0_']
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l1, = plt.plot(range(mean.shape[0]), mean, colors[7], label='AgentGoal, wfrs0', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[7], alpha=0.2)
        # ave_result['RL-agent'] = mean_point
        #
        # no_key_word_list = ['.DS_Store','.pdf','RID9']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'wfrs0.0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN']
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[8], label='AgentGoal, wfrs0.0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[8], alpha=0.2)
        # ave_result['HRL-agent(var0, wc0, sdai0)'] = mean_point
        #
        # no_key_word_list = ['.DS_Store','.pdf','RID9']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.2' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[9], label='AgentGoal, wfrs0.2', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[9], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.5' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[10], label='AgentGoal, wfrs0.5', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[10], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs1.0' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[11], label='AgentGoal, wfrs1.0', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[11], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        # #
        # #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs2.0' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[12], label='AgentGoal, wfrs2.0', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[12], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point
        #



        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', 'RFNCY-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.0_','RID9']
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l1, = plt.plot(range(mean.shape[0]), mean, colors[4], label='AgentGoal, embed, wfrs0', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[4], alpha=0.2)
        # ave_result['RL-agent'] = mean_point

        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'wfrs0.0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','RID9']
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[4], label='AgentGoal, wfrs0.0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[4], alpha=0.2)
        # ave_result['HRL-agent(var0, wc0, sdai0)'] = mean_point
        #
        # no_key_word_list = ['.DS_Store','.pdf']
        # key_word_list = ['AgentWithGoalJoint', '1999.p', 'RFIRS-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN','wfrs0.2','RID9' ]
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[5], label='AgentGoal, wfrs0.2', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[5], alpha=0.2)
        # ave_result['HRL-agent(var1, wc1, sdai0)'] = mean_point

        # min_len = min(min_len_list)
        min_len = 5000
        plt.grid(True)
        plt.ylabel('Success Rate')
        plt.xlabel('Simulation Epoch')
        plt.xlim([0, min_len])
        plt.legend(loc=4)
        # plt.savefig('learning_curve.png')
        # plt.show()
        plt.savefig(os.path.join(self.params['result_path'] + 'learning_curve' + str(min_len) + '.pdf'))
        print(ave_result)

    @staticmethod
    def get_dir_list(path, key_word_list=None, no_key_word_list=None):
        file_name_list = os.listdir(path)  # 获得原始json文件所在目录里面的所有文件名称
        if key_word_list == None and no_key_word_list == None:
            temp_file_list = file_name_list
        elif key_word_list != None and no_key_word_list == None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                if have_key_words == True:
                    temp_file_list.append(file_name)
        elif key_word_list == None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                if have_no_key_word == False:
                    temp_file_list.append(file_name)
        elif key_word_list != None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                    else:
                        pass
                if have_key_words == True and have_no_key_word == False:
                    temp_file_list.append(file_name)
        print(key_word_list, len(temp_file_list))
        # time.sleep(2)
        return temp_file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--result_path', dest='result_path', type=str, default='/Users/qianlong/Desktop/performance2/', help='the directory of the results.')
    parser.add_argument('--result_path', dest='result_path', type=str, default='D:/Desktop/对话系统/result/performance_DDQN/', help='the directory of the results.')
    parser.add_argument('--metric', dest='metric', type=str, default='recall', help='the metric to show')

    args = parser.parse_args()
    params = vars(args)
    drawer = DrawCurve(params)
    drawer.plot()
