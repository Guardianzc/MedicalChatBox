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

class AgentHRL_new2(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.parameter = parameter
        self.action_set = action_set
        self.slot_set = slot_set
        self.slot_set.pop("disease")
        self.disease_symptom = disease_symptom
        if parameter.get('prioritized_replay'):
            self.experience_replay_pool = PrioritizedReplayBuffer(buffer_size=parameter.get("experience_replay_pool_size"))
        else:
            self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))

        #restore the pre-trained model for several lower agents
        self.input_size_dqn_all = {1:374, 4:494, 5:389, 6:339, 7:279, 9:409, 10:254, 11:304, 12:304, 13:359, 14:394, 19:414}

        self.id2disease = {}
        for disease, v in self.disease_symptom.items():
            self.id2disease[v['index']] = disease
        self.id2lowerAgent = {}
        self.master_action_space = []
        dirs = os.listdir(self.parameter.get("label_all_model_path"))
        for model in dirs:
            #pattern = re.compile(r'(?<=label=)\d+\.?\d*')
            #label = pattern.findall(model)
            reg = re.compile(r"(?<=label)\d+")
            match = reg.search(model)
            label = match.group(0)
            #print(label)
            self.master_action_space.append(label)
            #assert len(label) == 1
            #label = label[0]
            label_all_path = self.parameter.get("file_all")
            label_new_path = os.path.join(label_all_path, 'label'+label)
            disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'),'rb'))
            slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'),'rb'))
            action_set = pickle.load(open(os.path.join(label_new_path, 'action_set.p'), 'rb'))

            temp_parameter = copy.deepcopy(parameter)
            #print(parameter["saved_model"])
            #if parameter.get("train_mode"):
            #    temp_parameter["saved_model"] = parameter["saved_model"].split('model_d10_agent')[0] + 'lower/' + str(
            #        label) + '/model_d10_agent' + parameter["saved_model"].split('model_d10_agent')[1]
            #else:
            temp_parameter["saved_model"] = parameter["saved_model"].split('model_d10agent')[0] + 'lower/' + str(
                    label) + '/model_d10agent' + parameter["saved_model"].split('model_d10agent')[1]
            temp_parameter["input_size_dqn"] = self.input_size_dqn_all[int(label)]
            self.id2lowerAgent[label] = LowerAgent(action_set=action_set, slot_set=slot_set,
                                                        disease_symptom=disease_symptom, parameter=temp_parameter,
                                                   disease_as_action=parameter.get("disease_as_action"))
            #model_path = os.path.join(self.parameter.get("label_all_model_path"), label)

            self.id2lowerAgent[label].dqn.restore_model(os.path.join(self.parameter.get("label_all_model_path"),model))
            self.id2lowerAgent[label].dqn.current_net.eval()
            self.id2lowerAgent[label].dqn.target_net.eval()

        # Master policy.
        if parameter.get("state_reduced"):
            input_size = len(self.slot_set) * 3
        else:
            input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.id2lowerAgent) + len(self.disease_symptom)
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
        if parameter.get("train_mode") is False :
            print("#################### master model is restored now ####################")
            self.master.restore_model(parameter.get("saved_model"))
            self.master.current_net.eval()
            self.master.target_net.eval()


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
        self.action = {'action': 'inform',
                       'inform_slots': {"disease": 'UNK'},
                       'request_slots': {},
                       "explicit_inform_slots": {},
                       "implicit_inform_slots": {}}

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # disease_symptom are not used in state_rep.
        #在这里master先于lower agent一步进行判断，如果master可以inform疾病就不需要激活lower agent，反之再交由lower agent决策
        min_epsilon = self.parameter.get("epsilon")
        index = kwargs.get("index")
        if index<1000:
            epsilon = 0.3- (0.3-min_epsilon)*index/1000
        else:
            epsilon = min_epsilon
        #print(index, epsilon)
        #print(state["turn"])
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set) # sequence representation.
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
                    self.master_action_index = random.randint(0, len(self.id2lowerAgent) - 1)
                    #master_action_index = random.sample(list(self.id2lowerAgent.keys()),1)[0]
                else:
                    self.master_action_index = self.master.predict(Xs=[state_rep])[1]
            # Evaluating mode.
            else:
                self.master_action_index = self.master.predict(Xs=[state_rep])[1]
            self.behave_prob = 1 - epsilon + epsilon / (len(self.id2lowerAgent) - 1)

            if self.master_action_index > (len(self.id2lowerAgent) - 1):
                self.action["turn"] = turn
                self.action["inform_slots"] = {"disease": self.id2disease[self.master_action_index - len(self.id2lowerAgent)]}
                self.action["speaker"] = 'agent'
                self.action["action_index"] = None
                return self.action, self.master_action_index
            #print(master_action_index)
            self.current_lower_agent_id = self.master_action_space[self.master_action_index]
            if self.parameter.get("prioritized_replay"):
                # print('2')
                Ys = self.master.predict(Xs=[state_rep])[0]
                self.current_action_value = Ys.detach().cpu().numpy()[0][self.master_action_index]

        # Lower agent takes an agent.
        #symptom_dist = self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]]
        # 在state_to_representation_last的步骤中，可以自动将不属于slot set中的slot去除掉
        agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, turn, greedy_strategy=False)
        #assert len(agent_action["request_slots"])>0 and len(agent_action["inform_slots"])==0
        return agent_action, self.master_action_index, lower_action_index

    def next_state_values_DDQN(self, next_state):
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=next_state,
                                                 slot_set=self.slot_set) # sequence representation.
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

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.master.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        #for key, lower_agent in self.id2lowerAgent.items():
        #    temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
        #    lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        priority_scale = self.parameter.get("priority_scale")
        if self.parameter.get("prioritized_replay"):
            for iter in range(math.ceil(self.experience_replay_pool.__len__() / batch_size)):
                batch = self.experience_replay_pool.sample(
                    batch_size=min(batch_size, self.experience_replay_pool.__len__()), priority_scale=priority_scale)
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
            print("cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / (self.experience_replay_pool.__len__() + 1e-10),
            self.experience_replay_pool.__len__()))
        else:
            for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
                batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
                loss = self.train(batch=batch)
                cur_bellman_err += loss["loss"]
            print("cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

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


    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
       # samples of master agent.
        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")

        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set)
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
        ##print("nexts", [idx for idx,x in enumerate(next_state_rep) if x==1], reward)
        #print(self.master_action_index, reward)
        if self.parameter.get("value_as_reward") is True:
            q_values = self.id2lowerAgent[self.current_lower_agent_id].get_q_values(state)
            q_values.reshape(q_values.shape[1])
            master_reward = np.max(q_values, axis=1)[0]
        else:
            master_reward = reward
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
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set)
        else:
            state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
            next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set,
                                                      slot_set=self.slot_set, disease_symptom=self.disease_symptom,
                                                      max_turn=self.parameter["max_turn"])
        self.experience_replay_pool.add(state_rep, self.master_action_index, reward, next_state_rep, episode_over, TD_error)

    def flush_pool(self):
        if self.parameter.get('prioritized_replay'):
            self.experience_replay_pool = PrioritizedReplayBuffer(buffer_size=self.parameter.get("experience_replay_pool_size"))
        else:
            self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        #for key, lower_agent in self.id2lowerAgent.items():
        #    self.id2lowerAgent[key].flush_pool()

    def train_mode(self):
        self.master.current_net.train()

    def eval_mode(self):
        self.master.current_net.eval()


