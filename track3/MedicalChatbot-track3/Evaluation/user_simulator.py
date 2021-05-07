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
        :return: raise error --- "Exceeded the maximum number of queries."
                 '3' --- not in the dialoge
                 '2' --- not_sure
                 '1' --- True
                 '0' --- False
        '''
        self.request_times[disease_id] += 1
        if self.request_times[disease_id] > self.max_turn:
            raise TypeError('Exceeded the maximum number of queries.')
        else:
            if slots in self.goal_slots[disease_id].keys():
                return self.goal_slots[disease_id][slots]
            else: 
                return '3'

    def request_time(self, disease_id):
        return self.request_times[disease_id]
    
if __name__ == '__main__':
    user = User('.\\goal_set_simul.p')
    '''
    self.goal_slots['10569463'] = 
        {'发热': '0', '消化不良': '2', '稀便': '1', '腹泻': '1'}
    '''
    print(user.request_time('10569463'))
    print(user.request('10569463','发热'))
    print(user.request('10569463','消化不良'))
    print(user.request('10569463','稀便'))
    print(user.request('10569463','缺钙'))
    for i in range(20):
        user.request('10569463','腹泻')
    print(user.request('10569463','缺钙'))
    print(user.request_time('10569463'))

