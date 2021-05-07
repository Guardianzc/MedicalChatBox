# -*- coding:utf-8 -*-

import random
import copy
import pickle
import sys,os
sys.path.append(os.getcwd().replace("src/dialogue_system",""))


class User(object):
    def __init__(self, goal_set_path):
        self.goal_set = pickle.load(file=open(goal_set_path, "rb"))
        self.max_turn = 11
        self._init()

    def _init(self):
        """
        used for initializing an instance or an episode.
        """

        self.goal_slots = dict()
        self.request_times = dict()
        self.goal_set = list(self.goal_set.values())[0]
        for goal in self.goal_set:
            ids = goal['consult_id']
            self.request_times[ids] = 0
            self.goal_slots[ids] = dict()
            for key,items in goal['goal']["explicit_inform_slots"].items():
                self.goal_slots[ids][key] = items
            for key,items in goal['goal']["implicit_inform_slots"].items():
                self.goal_slots[ids][key] = items
                            

    def request(self, disease_id, slots):
        '''
        :return: 3 --- "Exceeded the maximum number of queries."
                 2 --- not know
                 1 --- True
                 0 --- False
        '''
        self.request_times[disease_id] += 1
        if self.request_times[disease_id] > self.max_turn:
            return 3
        else:
            if slots in self.goal_slots[disease_id].keys():
                if self.goal_slots[disease_id][slots]:
                    return 1
                else:
                    return 0
            else:
                return 2

    def request_time(self, disease_id):
        return self.request_times[disease_id]
    
if __name__ == '__main__':
    user = User('.\\goal_set_simul.p')
    print(user.request_time('10293005'))
    print(user.request('10293005','咳嗽'))
    print(user.request('10293005','出汗'))
    print(user.request('10293005','缺钙'))
    for i in range(20):
        user.request('10293005','支气管炎')
    print(user.request('10293005','缺钙'))
    print(user.request_time('10293005'))

