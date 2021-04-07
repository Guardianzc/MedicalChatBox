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
        print(path)
        performance = pickle.load(file=open(path, 'rb'))

        success_rate = []
        average_reward = []
        average_wrong_disease = []
        average_turn = []
        for index in range(0, len(performance.keys()),1):
            print(performance[index].keys())
            success_rate.append(performance[index]["success_rate"])
            average_reward.append(performance[index]["average_reward"])
            average_wrong_disease.append(performance[index]["average_wrong_disease"])
            average_turn.append(performance[index]["average_turn"])

        smooth_num = 1
        d = [success_rate[i * smooth_num:i * smooth_num + smooth_num] for i in
             range(int(len(success_rate) / smooth_num))]

        success_rate_new = []
        cache = 0
        alpha = 0.8
        for i in d:
            cur = sum(i) / float(smooth_num)
            cache = cache * alpha + (1 - alpha) * cur
            success_rate_new.append(cache)
        return success_rate_new, success_rate[399]

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
        colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#a8d40f']
        global_idx = 1500
        min_len_list = []
        ave_result = {}


        no_key_word_list = ['.DS_Store','.pdf']

        key_word_list = ['dqn']
        mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        min_len_list.append(min_len)
        l1, = plt.plot(range(mean.shape[0]), mean, colors[0], label='RL-agent', linewidth=linewidth)
        plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[0], alpha=0.2)
        ave_result['RL-agent'] = mean_point


        # key_word_list = ['sim1', 'issdecay1', 'rac0']
        # mean, var, min_len,mean_point = self.get_mean_var(path=self.params['result_path'],
        #                                        key_word_list=key_word_list,
        #                                        no_key_word_list=no_key_word_list)
        # min_len_list.append(min_len)
        # l2, = plt.plot(range(mean.shape[0]), mean, colors[1], label='DQN-Sim(Decay=1)', linewidth=linewidth)
        # plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[1], alpha=0.2)
        # ave_result['DQN(Decay=1,RC=0)'] = mean_point

        min_len = min(min_len_list)
        plt.grid(True)
        plt.ylabel('Success Rate')
        plt.xlabel('Simulation Epoch')
        plt.xlim([0, min_len])
        plt.legend(loc=4)
        # plt.savefig('learning_curve.png')
        # plt.show()
        plt.savefig(os.path.join(self.params['result_path'] + '_sr_' + str(min_len) + '.pdf'))
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

    parser.add_argument('--result_path', dest='result_path', type=str, default='/Users/qianlong/Desktop/flat_dqn/', help='the directory of the results.')

    parser.add_argument('--metric', dest='metric', type=str, default='recall', help='the metric to show')

    args = parser.parse_args()
    params = vars(args)
    drawer = DrawCurve(params)
    drawer.plot()
