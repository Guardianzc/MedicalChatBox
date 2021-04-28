# -*-coding: utf-8 -*-

import sys
import os
import pickle
import time
import json
from collections import deque
import copy

sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system import dialogue_configuration


class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """
    def __init__(self, parameter, checkpoint_path):
        self.epoch_size = parameter.get("simulation_size",100)
        self.parameter = parameter
        self.checkpoint_path = checkpoint_path
        self.learning_curve = {}

        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))

        user = User(goal_set=goal_set, disease_syptom=disease_symptom,parameter=parameter)
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom, parameter=parameter)
        self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)

        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0,"average_wrong_disease":10}

    def simulate(self, epoch_number, train_mode=False):
        """
        Simulating the dialogue session between agent and user simulator.
        :param agent: the agent used to simulate, an instance of class Agent.
        :param epoch_number: the epoch number of simulation.
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: nothing to return.
        """
        # initializing the count matrix for AgentWithGoal
        # print('Initializing the count matrix for AgentWithGoal')
        # self.simulation_epoch(epoch_size=500, train_mode=train_mode)
        save_model = self.parameter.get("save_model")
        save_performance = self.parameter.get("save_performance")
        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn=self.parameter.get('max_turn'))
        for index in range(0, epoch_number,1):
            # Training AgentDQN with experience replay
            if train_mode is True:
                self.dialogue_manager.train()
                # Simulating and filling experience replay pool.
                self.simulation_epoch(epoch_size=self.epoch_size)

            # Evaluating the model.
            result = self.evaluate_model(dataset="train", index=index)
            if result["success_rate"] > self.best_result["success_rate"] and \
                    result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD and \
                    result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and train_mode==True:
                self.dialogue_manager.state_tracker.agent.flush_pool()
                self.simulation_epoch(epoch_size=self.epoch_size)
                if save_model is True:
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                    print("The model was saved.")
                else:
                    pass
                self.best_result = copy.deepcopy(result)
        # The training is over and save the model of the last training epoch.
        if save_model is True and train_mode is True and epoch_number > 0:
            self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
        if save_performance is True and train_mode is True and epoch_number > 0:
            self.__dump_performance__(epoch_index=index)

    def simulation_epoch(self, epoch_size):
        """
        Simulating one epoch when training model.
        :param epoch_size: the size of each epoch, i.e., the number of dialogue sessions of each epoch.
        :return: a dict of simulation results including success rate, average reward, average number of wrong diseases.
        """
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        inform_wrong_disease_count = 0
        for epoch_index in range(0,epoch_size, 1):
            self.dialogue_manager.initialize(dataset="train")
            episode_over = False
            while episode_over is False:
                reward, episode_over, dialogue_status,slots_proportion_list = self.dialogue_manager.next(greedy_strategy=True, save_record=True)
                total_reward += reward
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / epoch_size))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / epoch_size))
        average_reward = float("%.3f" % (float(total_reward) / epoch_size))
        average_turn = float("%.3f" % (float(total_turns) / epoch_size))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / epoch_size))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn, "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        # print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        self.dialogue_manager.state_tracker.agent.train_mode() # for training
        return res

    def evaluate_model(self, dataset, index):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong diseases.
        """
        save_performance = self.parameter.get("save_performance")
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        match_symptom_count = 0
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        evaluate_session_number = len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        inform_wrong_disease_count = 0
        num_of_true_slots=0
        num_of_implicit_slots=0
        for goal_index in range(0,evaluate_session_number, 1):
            self.dialogue_manager.initialize(dataset=dataset, goal_index=goal_index)
            episode_over = False
            while episode_over == False:
                reward, episode_over, dialogue_status,slots_proportion_list = self.dialogue_manager.next(save_record=False,train_mode=train_mode,greedy_strategy=False)
                total_reward += reward
            assert len(slots_proportion_list)>0
            num_of_true_slots+=slots_proportion_list[0]
            num_of_implicit_slots+=slots_proportion_list[1]
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1

        success_rate = float("%.3f" % (float(success_count) / evaluate_session_number))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / evaluate_session_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_session_number))
        average_turn = float("%.3f" % (float(total_turns) / evaluate_session_number))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / evaluate_session_number))
        if num_of_implicit_slots>0:
            match_rate=float("%.3f" %(float(num_of_true_slots)/float(num_of_implicit_slots)))
        else:
            match_rate=0.0


        self.dialogue_manager.state_tracker.agent.train_mode() # for training.
        res = {
            "success_rate":success_rate,
            "average_reward": average_reward,
            "average_turn": average_turn,
            "average_wrong_disease":average_wrong_disease,
            "ab_success_rate":absolute_success_rate,
            "average_match_count":match_rate
        }
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"]=success_rate
        self.learning_curve[index]["average_reward"]=average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        self.learning_curve[index]["average_wrong_disease"]=average_wrong_disease
        self.learning_curve[index]["average_match_count"]=match_rate
        if index % 10 ==0:
            print('[INFO]', self.parameter["run_info"])
        if index % 1000 == 999 and save_performance == True:
            self.__dump_performance__(epoch_index=index)
        print("%3d simulation SR [%s], ABSR [%s], ave reward %s, ave turns %s, ave match rate %s" % (index,res['success_rate'], res["ab_success_rate"],res['average_reward'], res['average_turn'],res[ "average_match_count"]))
        return res

    def warm_start(self, epoch_number):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
                             equals to the simulation epoch.
        :return: nothing to return.
        """
        for index in range(0,epoch_number,1):
            res = self.simulation_epoch(epoch_size=self.epoch_size)
            print("%3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong disease %s" % (
            index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
            # if len(self.dialogue_manager.experience_replay_pool)==self.parameter.get("experience_replay_pool_size"):
            #     break

    def __dump_performance__(self, epoch_index):
        """
        Saving the performance of model.

        Args:
            epoch_index: int, indicating the current epoch.
        """
        file_name = self.parameter["run_info"] + "_" + str(epoch_index) + ".p"
        performance_save_path = self.parameter["performance_save_path"]
        if os.path.isdir(performance_save_path) is False:
            os.mkdir(performance_save_path)
        pickle.dump(file=open(os.path.join(performance_save_path,file_name), "wb"), obj=self.learning_curve)
        self.dialogue_manager.state_tracker.agent.save_visitation(epoch_index)