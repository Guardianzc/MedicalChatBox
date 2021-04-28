import logging
import pandas as pd
import random
from torch.utils import data
from utils import *
import torch
import pickle
import argparse
import logging
import os
from datetime import datetime
from torch.backends import cudnn


class DataLoader(data.Dataset):

    def __init__(self, slot_set, test_set, disease_set, goal_test_path, mode='train'):
        self.goal_test = pickle.load(file=open(goal_test_path, "rb"))
        self.mode = mode
        self.disease_set = disease_set
        self.dataset = self.goal_test[mode]
        self.num_slot = len(slot_set)
        self.num_test = len(test_set)
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        self.test_set = test_set
        self.disease_set = disease_set

        logging.info("image count in {} path :{}".format(self.mode, len(self.dataset)))
        
    def init_setting(self, state):
        goal = torch.zeros(self.num_slot + self.num_test)
        origin_state = torch.zeros(self.num_slot + self.num_test)
        explicit_symptom = list(state['goal']['explicit_inform_slots'])[0]
        origin_state[self.slot_set[explicit_symptom]] = 1
        goal[self.slot_set[explicit_symptom]] = 1
        implicit_inform_slots = list(state['goal']['implicit_inform_slots'])
        test_slots = list(state['goal']['test_slots'])
        for slot in implicit_inform_slots:
            goal[self.slot_set[slot]] = 1
        for slot in test_slots:
            goal[self.test_set[slot] + self.num_slot] = 1
        return origin_state, goal 
    
    def __getitem__(self, index):
        data = self.dataset[index]
        origin_state, goal = self.init_setting(data)
        goal_disease = self.disease_set[data['disease_tag']]

        '''
        ToTensor:
        - change dimension：img = torch.from_numpy(pic.transpose((2, 0, 1)))
        - change type ：float/int
        - divide 255: img.float().div(255)
        '''

        # for the reason  that torch classification criterion can only accept classes label [0, C-1]

        return origin_state, goal, goal_disease

    def __len__(self):
        return len(self.dataset)


def get_loader(slot_set, test_set, disease_set, goal_test_path, batch_size=16, num_workers=4, mode='train', shuffle = True):
    """
        Builds and returns Data loader.
        :param batch_size:
        :param num_workers:
        :param mode:
        :param augmentation_prob:
        :return:
    """
    dataset = DataLoader(slot_set, test_set, disease_set, goal_test_path,
                          mode=mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data_root = 'E:\大数据\医学影像\SRMM&MVMM\DATA\LGE_DATA'
    train_list, test_list, val_list = read_in_data_lists(data_root)
    
    test_loader = get_loader(data_list=test_list,
                             batch_size=4,
                             num_workers=8,
                             mode='test',
                             augmentation_prob=0.)
    for i, (img, lab) in enumerate(test_loader):
        label = lab.detach().cpu().numpy()
        print(np.sum(label[0, 0, ...]))
