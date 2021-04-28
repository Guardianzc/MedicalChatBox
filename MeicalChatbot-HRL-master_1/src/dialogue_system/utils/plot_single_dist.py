# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import os
import pickle
import argparse
import numpy as np

# name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
# num_list = [.5, 0.6, 7.8, 6]
# num_list1 = [2, 3, 1,4]
# x = list(range(len(num_list)))
# total_width, n = 0.8, 3
# width = total_width / n
#
# plt.bar(x, num_list, width=width, label='boy', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
#
# x = [i + width for i in x]
# plt.bar(x, num_list, width=width, label='men', tick_label=name_list, fc='b')
#
# plt.legend()
# plt.show()


class PlotDistribution(object):
    def __init__(self, params):
        self.params = params

    def get_visitation_mean(self, path, key_word_list, no_key_word_list):
        file_list = PlotDistribution.get_dir_list(path=path, key_word_list=key_word_list, no_key_word_list=no_key_word_list)
        all_run_visitation = []
        for file_name in file_list:
            visitation_count = pickle.load(open(os.path.join(path, file_name), 'rb'))
            visitation_list = [visitation_count[key] for key in sorted(visitation_count.keys())]
            all_run_visitation.append(visitation_list)
        count = np.array(all_run_visitation)
        return count.mean(axis=0)

    def plot(self):
        colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#a8d40f', '#b4546f',    '#6495ED', '#778899', '#48D1CC', '#00FA9A','#F4A460', '#8FBC8F','#C0C0C0']
        no_key_word_list = ['.DS_Store','.pdf','RID9']
        key_word_list = ['AgentDQN', '4599.p']
        mean_point = self.get_visitation_mean(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        mean_point = mean_point[0:len(mean_point) - 4]
        name_list = [i for i in range(len(mean_point))]
        plt.bar(range(len(mean_point)), mean_point, label='Flat-DQN', fc=colors[0])
        plt.legend()
        plt.show()

        no_key_word_list = ['.DS_Store','.pdf','RID9']
        key_word_list = ['AgentWithGoal2', '4599.p']
        mean_point = self.get_visitation_mean(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        mean_point = mean_point[0:len(mean_point) - 4]
        name_list = [i for i in range(len(mean_point))]
        plt.bar(range(len(mean_point)), mean_point, label='HRL, ex', fc=colors[0])
        plt.legend()
        plt.show()

        no_key_word_list = ['.DS_Store','.pdf']
        key_word_list = ['AgentWithGoal2', '4599.p','RID9']
        mean_point = self.get_visitation_mean(path=self.params['result_path'],
                                               key_word_list=key_word_list,
                                               no_key_word_list=no_key_word_list)
        mean_point = mean_point[0:len(mean_point) - 4]
        name_list = [i for i in range(len(mean_point))]
        plt.bar(range(len(mean_point)), mean_point, label='HRL, ex&im', fc=colors[0])
        plt.legend()
        plt.show()

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

    parser.add_argument('--result_path', dest='result_path', type=str, default='/Users/qianlong/Desktop/visit2/', help='the directory of the results.')

    parser.add_argument('--metric', dest='metric', type=str, default='recall', help='the metric to show')

    args = parser.parse_args()
    params = vars(args)
    drawer = PlotDistribution(params)
    drawer.plot()
