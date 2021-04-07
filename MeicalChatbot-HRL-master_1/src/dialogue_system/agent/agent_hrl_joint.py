import numpy as np
import copy
import sys, os
import random
import re
import pickle
import math
from collections import deque, Counter
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.policy_learning.dqn_torch import DQN,DQN2
from src.dialogue_system.agent.utils import state_to_representation_last, reduced_state_to_representation_last
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent.prioritized_new import PrioritizedReplayBuffer

class AgentHRL_joint(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.parameter = parameter
        self.action_set = action_set
        self.slot_set = slot_set
        self.slot_set.pop("disease")
        self.disease_symptom = disease_symptom
        self.master_experience_replay_size = 10000
        if parameter.get('prioritized_replay'):
            self.experience_replay_pool = PrioritizedReplayBuffer(buffer_size=self.master_experience_replay_size)
        else:
            self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)

        self.input_size_dqn_all = {1:374, 4:494, 5:389, 6:339, 7:279, 12:304, 13:359, 14:394, 19:414}

        self.id2disease = {}
        self.id2lowerAgent = {}
        self.master_action_space = []
        temp_parameter = {}
        for key,value in self.input_size_dqn_all.items():
            label = str(key)
            #print(label)
            self.master_action_space.append(label)
            #assert len(label) == 1
            #label = label[0]
            label_all_path = self.parameter.get("file_all")
            label_new_path = os.path.join(label_all_path, 'label'+str(label))
            disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'),'rb'))
            slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'),'rb'))
            action_set = pickle.load(open(os.path.join(label_new_path, 'action_set.p'), 'rb'))

            temp_parameter[label] = copy.deepcopy(parameter)
            #print(parameter["saved_model"])
            #if parameter.get("train_mode"):
            #    temp_parameter["saved_model"] = parameter["saved_model"].split('model_d10_agent')[0] + 'lower/' + str(
            #        label) + '/model_d10_agent' + parameter["saved_model"].split('model_d10_agent')[1]
            #else:
            #temp_parameter["saved_model"] = parameter["saved_model"].split('model_d10agent')[0] + 'lower/' + str(
            #        label) + '/model_d10agent' + parameter["saved_model"].split('model_d10agent')[1]
            path_list = parameter['saved_model'].split('/')
            path_list.insert(-1, 'lower')
            path_list.insert(-1, str(label))
            temp_parameter[label]['saved_model'] = '/'.join(path_list)
            temp_parameter[label]['gamma'] = temp_parameter[label]['gamma_worker']  # discount factor for the lower agent.

            temp_parameter[label]["input_size_dqn"] = value
            self.id2lowerAgent[label] = LowerAgent(action_set=action_set, slot_set=slot_set,
                                                        disease_symptom=disease_symptom, parameter=temp_parameter[label],
                                                   disease_as_action=False)
            #model_path = os.path.join(self.parameter.get("label_all_model_path"), label)

        # Master policy.
        if parameter.get("state_reduced"):
            input_size = len(self.slot_set) * 3
        else:
            input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 300)
        output_size = len(self.id2lowerAgent)
        if self.parameter.get("disease_as_action")==False:
            output_size = len(self.id2lowerAgent)+1
        #print("input_size",input_size)
        self.master= DQN2(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=output_size,
                       parameter=parameter,
                       named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.parameter = parameter
        #self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.current_lower_agent_id = -1
        self.behave_prob = 1
        print("master:",self.master_action_space)
        self.count = 0
        self.past_lower_agent_pool = {key:0 for key in self.id2lowerAgent.keys()}

        if parameter.get("train_mode") is False :
            print("########## master model is restore now ##########")
            self.master.restore_model(parameter.get("saved_model"))
            self.master.current_net.eval()
            self.master.target_net.eval()
            for label, agent in self.id2lowerAgent.items():
                #print(temp_parameter[label])
                self.id2lowerAgent[label].dqn.restore_model(temp_parameter[label]['saved_model'])
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()



        self.agent_action = {
            "turn": 1,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn": None,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # disease_symptom are not used in state_rep.
        epsilon = self.parameter.get("epsilon")
        #print(state["turn"])
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
        else:
            state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"]) # sequence representation.
        #print(len(state_rep))
        # Master agent takes an action.
        if self.parameter.get("initial_symptom") and state["turn"]>0:
            pass
        else:
            #print("####")
            if greedy_strategy == True:
                greedy = random.random()
                if greedy < epsilon:
                    self.master_action_index = random.randint(0, len(self.id2lowerAgent))
                    #print(self.master_action_index)
                    #master_action_index = random.sample(list(self.id2lowerAgent.keys()),1)[0]
                else:
                    self.master_action_index = self.master.predict(Xs=[state_rep])[1]
            # Evaluating mode.
            else:
                self.master_action_index = self.master.predict(Xs=[state_rep])[1]
            self.behave_prob = 1 - epsilon + epsilon / (len(self.id2lowerAgent) - 1)
            #print(master_action_index)

            if self.parameter.get("prioritized_replay"):
                # print('2')
                Ys = self.master.predict(Xs=[state_rep])[0]
                self.current_action_value = Ys.detach().cpu().numpy()[0][self.master_action_index]
        #print(self.master_action_index)  这里还存在9

        # Lower agent takes an agent.
        #symptom_dist = self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]]
        # 在state_to_representation_last的步骤中，可以自动将不属于slot set中的slot去除掉
        if self.parameter.get("disease_as_action"):
            self.current_lower_agent_id = self.master_action_space[self.master_action_index]
            agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, turn, greedy_strategy=greedy_strategy)

        else:
            if self.master_action_index > (len(self.id2lowerAgent) - 1):
                agent_action = {'action': 'inform', 'inform_slots': {"disease": 'UNK'}, 'request_slots': {},
                       "explicit_inform_slots": {}, "implicit_inform_slots": {}}
                agent_action["turn"] = turn
                agent_action["inform_slots"] = {"disease": None}
                agent_action["speaker"] = 'agent'
                agent_action["action_index"] = None
                lower_action_index = -1
            else:
                self.current_lower_agent_id = self.master_action_space[self.master_action_index]
                #print(self.current_lower_agent_id)
                agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, turn, greedy_strategy=greedy_strategy)
                assert len(list(agent_action["request_slots"].keys())) == 1
            #print(self.current_lower_agent_id, lower_action_index)
            #print(agent_action)
        return agent_action, self.master_action_index, lower_action_index

    def next_state_values_DDQN(self, next_state):
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=next_state,
                                                 slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
        else:
            state_rep = state_to_representation_last(state=next_state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
        action_index = self.master.predict(Xs=[state_rep])[1]
        Ys = self.master.predict_target(Xs=[state_rep])
        next_action_value = Ys.detach().cpu().numpy()[0][action_index]

        return next_action_value

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.master.singleBatch(batch=batch,params=self.parameter,weight_correction=self.parameter.get("weight_correction"))
        return loss

    def update_target_network(self):
        self.master.update_target_network()
        for key in self.id2lowerAgent.keys():
            self.id2lowerAgent[key].update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.master.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        #Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
            lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        #print(batch_size)

        priority_scale = self.parameter.get("priority_scale")
        if self.parameter.get("prioritized_replay"):
            for iter in range(math.ceil(self.experience_replay_pool.__len__() / batch_size)):
                batch = self.experience_replay_pool.sample(
                    batch_size=min(batch_size, self.experience_replay_pool.__len__()), priority_scale=priority_scale)
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
            print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / (self.experience_replay_pool.__len__() + 1e-10), self.experience_replay_pool.__len__()))
            for disease_id, lower_agent in self.id2lowerAgent.items():
                if len(lower_agent.experience_replay_pool) > 120:
                    lower_agent.train_dqn()
        else:
            for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
                batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
            print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
            if self.count % 10 == 9:
                for group_id, lower_agent in self.id2lowerAgent.items():
                    #if len(lower_agent.experience_replay_pool) ==10000 or (len(lower_agent.experience_replay_pool)-self.past_lower_agent_pool[group_id])>100:
                    if len(lower_agent.experience_replay_pool) > 100:
                        lower_agent.train_dqn(label = group_id)
                        self.past_lower_agent_pool[group_id] = len(lower_agent.experience_replay_pool)

        self.count += 1
        # Training of lower agents.
        #for disease_id, lower_agent in self.id2lowerAgent.items():
        #    lower_agent.train_dqn()

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


    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, lower_reward):
        # samples of master agent.
        #print(state)
        #print(reward)

        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        '''
        if reward == self.parameter.get("reward_for_repeated_action"):
            lower_reward = reward
            #reward = reward * 2
        else:
            lower_reward = max(0, shaping * alpha)
            #lower_reward = shaping * alpha
        '''

        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping

        # samples of lower agent.
        #print(agent_action)
        if int(agent_action) >= 0:
            #print(lower_reward)
            self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, lower_reward,
                                                                                   next_state, episode_over)

        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter=self.parameter)
        else:
            state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
            next_state_rep = state_to_representation_last(state=next_state,
                                                      action_set=self.action_set,
                                                      slot_set=self.slot_set,
                                                      disease_symptom=self.disease_symptom,
                                                      max_turn=self.parameter["max_turn"])
        #print("state", [idx for idx,x in enumerate(state_rep) if x==1], agent_action)
        #print("nexts", [idx for idx,x in enumerate(next_state_rep) if x==1], reward)
        if self.parameter.get("value_as_reward") is True:
            q_values = self.id2lowerAgent[self.current_lower_agent_id].get_q_values(state)
            q_values.reshape(q_values.shape[1])
            master_reward = np.max(q_values, axis=1)[0]
        else:
            master_reward = reward
        #print(master_reward)
        self.experience_replay_pool.append((state_rep, self.master_action_index, master_reward, next_state_rep, episode_over))

    def record_prioritized_training_sample(self, state, agent_action, reward, next_state, episode_over, TD_error, **kwargs):
        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        # if True:
        # print('shaping', shaping)
        # Reward shaping only when non-terminal state.
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping

        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter=self.parameter)
        else:
            state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
            next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set,
                                                      slot_set=self.slot_set, disease_symptom=self.disease_symptom,
                                                      max_turn=self.parameter["max_turn"])
        self.experience_replay_pool.add(state_rep, agent_action, reward, next_state_rep, episode_over, TD_error)

    def flush_pool(self):
        if self.parameter.get('prioritized_replay'):
            self.experience_replay_pool = PrioritizedReplayBuffer(buffer_size=self.master_experience_replay_size)
        else:
            self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        #for key, lower_agent in self.id2lowerAgent.items():
        #    self.id2lowerAgent[key].flush_pool()

    def train_mode(self):
        self.master.current_net.train()

    def eval_mode(self):
        self.master.current_net.eval()




