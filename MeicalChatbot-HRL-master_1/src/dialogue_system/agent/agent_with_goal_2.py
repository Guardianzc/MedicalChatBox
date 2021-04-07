# -*- coding: utf8 -*-
"""
Agent for hierarchical reinforcement learning. The master agent first generates a goal, and the goal will be inputted
into the lower agent.
"""

import numpy as np
import copy
import sys, os
import random
import math
from collections import namedtuple
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.policy_learning.dqn_torch import DQN
from src.dialogue_system.agent.utils import state_to_representation_last, reduced_state_to_representation_last
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.policy_learning.internal_critic import InternalCritic


class AgentWithGoal(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = disease_symptom
        self.disease_num = parameter.get("disease_number")
        self.slot_dim=1

        ##################
        # Master policy.
        #######################
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        self.output_size = parameter.get('goal_dim', 2*self.disease_num)
        self.dqn = DQN(input_size=input_size + self.output_size,
                       hidden_size=hidden_size,
                       output_size=self.output_size,
                       parameter=parameter,
                       named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        if parameter.get("train_mode") is False :
            self.dqn.restore_model(parameter.get("saved_model"))
            self.dqn.current_net.eval()
            self.dqn.target_net.eval()

        ###############################
        # Internal critic
        ##############################
        # symptom distribution by diseases.
        temp_slot_set = copy.deepcopy(slot_set)
        temp_slot_set.pop('disease')
        self.disease_to_symptom_dist = {}
        self.id2disease = {}
        total_count = np.zeros(len(temp_slot_set))
        for disease, v in self.disease_symptom.items():
            dist = np.zeros(len(temp_slot_set))
            self.id2disease[v['index']] = disease
            for symptom, count in v['symptom'].items():
                dist[temp_slot_set[symptom]] = count
                total_count[temp_slot_set[symptom]] += count
            self.disease_to_symptom_dist[disease] = dist

        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count
        goal_embed_value = [0] * len(disease_symptom)
        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count
            goal_embed_value[disease_symptom[disease]['index']] = list(self.disease_to_symptom_dist[disease])

        self.internal_critic = InternalCritic(input_size=len(temp_slot_set)*self.slot_dim + len(self.disease_symptom), hidden_size=hidden_size,
                                              output_size=len(temp_slot_set), goal_num=len(self.disease_symptom),
                                              goal_embedding_value=goal_embed_value, slot_set=temp_slot_set,
                                              parameter=parameter)
        print(os.getcwd())
        self.internal_critic.restore_model('../agent/pre_trained_internal_critic_dropout_both_one_hot512.pkl')

        #################
        # Lower agent.
        ##############
        temp_parameter = copy.deepcopy(parameter)
        temp_parameter['input_size_dqn'] = input_size + len(self.disease_symptom)
        path_list = parameter['saved_model'].split('/')
        path_list.insert(-1, 'lower')
        temp_parameter['saved_model'] = '/'.join(path_list)
        temp_parameter['gamma'] = temp_parameter['gamma_worker'] # discount factor for the lower agent.
        self.lower_agent = LowerAgent(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                                      parameter=temp_parameter,disease_as_action=False)
        named_tuple = ('state', 'agent_action', 'reward', 'next_state', 'episode_over','goal')
        self.lower_agent.dqn.Transition = namedtuple('Transition', named_tuple)
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]
        if temp_parameter.get("train_mode") is False:
            self.lower_agent.dqn.restore_model(temp_parameter.get("saved_model"))
            self.lower_agent.dqn.current_net.eval()
            self.lower_agent.dqn.target_net.eval()

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        # print('{} new session {}'.format('*'*20, '*'*20))
        # print('***' * 20)
        #self.subaction_history = []
        self.master_reward = 0.
        self.sub_task_terminal = True
        self.inform_disease = False
        self.master_action_index = None
        self.last_master_action_index = None
        self.worker_action_index = None
        self.last_worker_action_index = None
        self.intrinsic_reward = 0.0
        self.sub_task_turn = 0
        self.states_of_one_session = []
        self.master_previous_actions = set()
        self.worker_previous_actions = set()
        self.lower_agent.initialize()
        self.action = {'action': 'inform',
                       'inform_slots': {"disease": 'UNK'},
                        'request_slots': {},
                        "explicit_inform_slots": {},
                        "implicit_inform_slots": {}}

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        The master first select a goal, then the lower agent takes an action based on this goal and state.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        self.disease_tag = kwargs.get("disease_tag")
        self.sub_task_terminal, _, similar_score = self.intrinsic_critic(state, self.master_action_index, disease_tag=kwargs.get("disease_tag"))

        # The current sub-task is terminated or the first turn of the session.


        if self.sub_task_terminal is True or self.master_action_index is None:
            #self.subaction_history=[]
            self.master_reward = 0.0
            self.master_state = copy.deepcopy(state)
            self.sub_task_turn = 0
            self.last_master_action_index = copy.deepcopy(self.master_action_index)
            self.master_previous_actions.add(self.last_master_action_index)
            self.master_action_index = self.__master_next__(state, self.master_action_index, greedy_strategy)
        else:
            pass

        # Inform disease.
        if self.master_action_index > (self.disease_num-1):
            self.action["turn"] = turn
            self.action["inform_slots"] = {"disease": self.id2disease[self.master_action_index - self.disease_num]}
            self.action["speaker"] = 'agent'
            self.action["action_index"] = None
            return self.action, None

        # print('turn: {}, goal: {}, label: {}, sub-task finish: {}, inform disease: {}, intrinsic reward: {}, similar score: {}'.format(turn, self.master_action_index, self.disease_symptom[self.disease_tag]['index'], self.sub_task_terminal, self.inform_disease, self.intrinsic_reward, similar_score))

        # Lower agent takes an agent. Not inform disease.
        goal = np.zeros(len(self.disease_symptom))
        self.sub_task_turn += 1
        goal[self.master_action_index] = 1
        self.last_worker_action_index = copy.deepcopy(self.worker_action_index)
        self.worker_previous_actions.add(self.last_worker_action_index)
        agent_action, action_index = self.lower_agent.next(state, turn, greedy_strategy, goal=goal)
        self.worker_action_index = action_index
        #print(self.master_action_index,self.lower_agent.get_q_values(state=state,goal=goal))
        # print('action', agent_action)
        '''
        goal_reps={}
        for i in range(10):
            temp_goal=[0]*10
            temp_goal[i]=1
            goal_rep = self.lower_agent.get_q_values(state=state, goal=temp_goal)
            goal_reps[i] = goal_rep
        #print(state['turn'],goal_rep)
        #print('action', agent_action)
        outputfile=os.getcwd()+'/goal_file/goal_rep_turn'+str(state['turn'])+'.p'
        pickle.dump(file=open(outputfile,'wb'),obj=goal_reps)
        '''
        return agent_action, action_index

    def __master_next__(self, state, last_master_action, greedy_strategy):
        # disease_symptom are not used in state_rep.
        epsilon = self.parameter.get("epsilon")
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set) # sequence representation.
            #next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set)
        else:
            state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])  # sequence representation.
        last_action_rep = np.zeros(self.output_size)
        if last_master_action is not None:
            last_action_rep[last_master_action] = 1
        state_rep = np.concatenate((state_rep, last_action_rep), axis=0)
        # Master agent takes an action, i.e., selects a goal.
        if greedy_strategy is True:
            greedy = random.random()
            if greedy < epsilon:
                master_action_index = random.randint(0, self.output_size - 1)
            else:
                master_action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            master_action_index = self.dqn.predict(Xs=[state_rep])[1]
        return master_action_index

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sam ple used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter,weight_correction=self.parameter.get("weight_correction"))
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()
        self.lower_agent.update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/')
        self.lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
            batch = random.sample(self.experience_replay_pool, min(batch_size,len(self.experience_replay_pool)))
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s, " % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
        # Training of lower agents.
        self.lower_agent.train_dqn()
        # Training of internal critic.
        # self.internal_critic.buffer_replay()

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        """
        这里lower agent和master agent的sample都是在这里直接保存的，没有再另外调用函数。
        """
        # samples of internal critic.
        self.states_of_one_session.append(state)
        if episode_over is True:
            # current session is successful.
            if reward == self.parameter.get('reward_for_success'):
                for one_state in self.states_of_one_session:
                    # positive samples.
                    self.internal_critic.record_training_positive_sample(one_state, self.master_action_index)
                    # negative samples.
                    for index in range(self.output_size):
                        if index != self.master_action_index:
                            self.internal_critic.record_training_negative_sample(one_state, index)
            # current session is failed.
            elif reward == self.parameter.get('reward_for_fail') and state['turn'] <= self.parameter.get('max_turn') - 2:
                for one_state in self.states_of_one_session:
                    self.internal_critic.record_training_negative_sample(one_state, self.master_action_index)

        # reward shaping
        alpha = self.parameter.get("weight_for_reward_shaping")
        # if episode_over is True: shaping = self.reward_shaping(agent_action, self.master_action_index)
        # else: shaping = 0
        shaping = 0
        # Reward shaping only when non-terminal state.
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping

        # state to vec.
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set)
            master_state_rep = reduced_state_to_representation_last(state=self.master_state, slot_set=self.slot_set)
        else:
            state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set,disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
            next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
            master_state_rep = state_to_representation_last(state=self.master_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])


        # samples of master agent.
        sub_task_terminal, intrinsic_reward, _ = self.intrinsic_critic(next_state, self.master_action_index,disease_tag=self.disease_tag)
        self.master_reward += reward
        if self.sub_task_terminal is True and sub_task_terminal is True:
            last_master_action_rep = np.zeros(self.output_size)
            current_master_action_rep = np.zeros(self.output_size)
            # 将master所有已经选择的动作加入到状态表示中。
            for last_master_action_index in self.master_previous_actions:
                if last_master_action_index is not None:
                    last_master_action_rep[last_master_action_index] = 1
                    current_master_action_rep[last_master_action_index] = 1
            if self.master_action_index is not None: current_master_action_rep[self.master_action_index] = 1
            master_state_rep = np.concatenate((master_state_rep, last_master_action_rep), axis=0)
            next_master_state_rep = np.concatenate((next_state_rep, current_master_action_rep), axis=0)

            self.experience_replay_pool.append((master_state_rep, self.master_action_index, self.master_reward, next_master_state_rep, episode_over))

            # # master repeated action.
            # if self.master_action_index in self.master_previous_actions:
            #     temp_reward = - self.parameter.get("max_turn") / 2
            #     self.experience_replay_pool.append( (master_state_rep, self.master_action_index, temp_reward, next_master_state_rep, episode_over))
            # else:
            #     self.experience_replay_pool.append((master_state_rep, self.master_action_index, self.master_reward, next_master_state_rep, episode_over))

        # samples of lower agent.
        if agent_action is not None: # session is not over. Otherwise the agent_action is not one of the lower agent's actions.
            goal = np.zeros(len(self.disease_symptom))
            goal[self.master_action_index] = 1
            state_rep = np.concatenate((state_rep, goal), axis=0)
            next_state_rep = np.concatenate((next_state_rep, goal), axis=0)
            # reward shaping for lower agent on intrinsic reward.

            shaping = self.reward_shaping(state, next_state)
            intrinsic_reward += alpha * shaping
            self.lower_agent.experience_replay_pool.append((state_rep, agent_action, intrinsic_reward, next_state_rep, sub_task_terminal, self.master_action_index))
            # visitation count.
            self.lower_agent.action_visitation_count.setdefault(agent_action, 0)
            self.lower_agent.action_visitation_count[agent_action] += 1

            # # repeated action
            # if agent_action in self.worker_previous_actions:
            #     temp_reward = -0.5
            #     self.lower_agent.experience_replay_pool.append((state_rep, agent_action, temp_reward, next_state_rep, sub_task_terminal, self.master_action_index))
            # else:
            #     self.lower_agent.experience_replay_pool.append((state_rep, agent_action, intrinsic_reward, next_state_rep, sub_task_terminal, self.master_action_index))

            # 如果达到固定长度，同时去掉即将删除transition的计数。
            self.visitation_count[self.master_action_index, agent_action] += 1
            if len(self.lower_agent.experience_replay_pool) == self.lower_agent.experience_replay_pool.maxlen:
                _, pre_agent_action, _, _, _, pre_master_action = self.lower_agent.experience_replay_pool.popleft()
                self.visitation_count[pre_master_action, pre_agent_action] -= 1

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.lower_agent.flush_pool()
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]

    def intrinsic_critic(self, state, master_action_index, disease_tag):
        # For the first turn.
        if master_action_index is None:
            return True, 0, 0

        if master_action_index > (self.disease_num-1):
            intrinsic_reward = self.parameter.get('reward_for_success') / 2
            return True, intrinsic_reward, 0

        self.internal_critic.critic.eval()
        sub_task_terminate = False
        #intrinsic_reward = -1
        intrinsic_reward = 0

        goal_list = [i for i in range(len(self.disease_symptom))]
        state_batch = [state] * len(self.disease_symptom)
        similarity_score = self.internal_critic.get_similarity_state_dict(state_batch, goal_list)[master_action_index]

        if similarity_score > self.parameter.get("upper_bound_critic"):
            sub_task_terminate = True
            intrinsic_reward = self.parameter.get('reward_for_success') / 2
            #print("the number %d subtask is over, simialrity score is high" % self.master_action_index)

        elif self.sub_task_turn >= 4:
            sub_task_terminate = True
            #intrinsic_reward = self.parameter.get('reward_for_fail') / 2
            intrinsic_reward = -11
            #print("the number %d subtask is over, subtask turn is over 4" % self.master_action_index)

        elif similarity_score < self.parameter.get("lower_bound_critic"):
            sub_task_terminate = True
            intrinsic_reward = self.parameter.get('reward_for_success') / 2
            #print("the number %d subtask is over, similarity score is low" % self.master_action_index)

        if self.worker_action_index in self.worker_previous_actions:
            intrinsic_reward += -5.5

        self.internal_critic.critic.train()
        return sub_task_terminate, intrinsic_reward, similarity_score

    def reward_shaping1(self, lower_agent_action, goal):
        prob_action_goal = self.visitation_count[goal, lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        prob_goal = self.visitation_count.sum(1)[goal] / (self.visitation_count.sum() + 1e-8)
        prob_action = self.visitation_count.sum(0)[lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        return np.log(prob_action_goal / (prob_action * prob_goal + 1e-8))

    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict.update(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        slot_dict.update(state["current_slots"]["proposed_slots"])
        slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["proposed_slots"])
        next_slot_dict.update(next_state["current_slots"]["agent_request_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def train_mode(self):
        self.dqn.current_net.train()
        self.lower_agent.dqn.current_net.train()
        self.internal_critic.critic.train()

    def eval_mode(self):
        self.dqn.current_net.eval()
        self.lower_agent.dqn.current_net.eval()
        self.internal_critic.critic.eval()

    def save_visitation(self, epoch_index):
        self.lower_agent.save_visitation(epoch_index)