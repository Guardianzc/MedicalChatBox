# -*- coding:utf-8 -*-
"""
Basic user simulator, random choice action.
# Structure of agent_action:
agent_action = {
    "turn":0,
    "speaker":"agent",
    "action":"request",
    "request_slots":{},
    "inform_slots":{},
    "explicit_inform_slots":{},
    "implicit_inform_slots":{}
}
# Structure of user_action:
user_action = {
    "turn": 0,
    "speaker": "user",
    "action": "request",
    "request_slots": {},
    "inform_slots": {},
    "explicit_inform_slots": {},
    "implicit_inform_slots": {}
}
# Structure of user goal.
{
  "consult_id": "10002219",
  "disease_tag": "上呼吸道感染",
  "goal": {
    "request_slots": {
      "disease": "UNK"
    },
    "explicit_inform_slots": {
      "呼吸不畅": true,
      "发烧": true
    },
    "implicit_inform_slots": {
      "厌食": true,
      "鼻塞": true
    }
  }
"""

import random
import copy

import sys,os
sys.path.append(os.getcwd().replace("src/dialogue_system",""))

from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent.agent import Agent


class User(object):
    def __init__(self, goal_set, disease_symptom, parameter):
        self.goal_set, self.disease_sample_count = self.__prepare_goal_set__(goal_set,parameter)
        self.max_turn = parameter["max_turn"]
        self.parameter = parameter
        self.disease_symptom = Agent.disease_symptom_clip(disease_symptom=disease_symptom,denominator=20,parameter=parameter)
        # self._init()

    def initialize(self, dataset, goal_index=None):
        self._init(dataset=dataset, goal_index=goal_index)

        # Initialize rest slot for this user.
        # 初始的时候request slot里面必有disease，然后随机选择explicit_inform_slots里面的slot进行用户主诉的构建，若explicit里面没
        # 有slot，初始就只有一个request slot，里面是disease，因为implicit_inform_slots是需要与agent交互的过程中才能发现的，患者自己并
        # 不能发现自己隐含的一些症状。
        goal = self.goal["goal"]
        self.state["action"] = "request"
        self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN

        # randomly select several slots from explicit symptoms.
        # if len(goal["explicit_inform_slots"].keys()) > 0:
        #     first_inform_number = random.randint(0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN,len(goal["explicit_inform_slots"].keys()))
        #     inform_slots = random.sample(list(goal["explicit_inform_slots"].keys()),k=first_inform_number)
        # else:
        #     inform_slots = []

        # inform all explicit_symptoms at first.
        inform_slots = list(goal["explicit_inform_slots"].keys())
        for slot in list(goal["explicit_inform_slots"].keys()):
            if slot in inform_slots:
                self.state["inform_slots"][slot] = goal["explicit_inform_slots"][slot]

        # # inform all implicit_symptoms at first.
        # inform_slots = list(goal["implicit_inform_slots"].keys())
        # for slot in list(goal["implicit_inform_slots"].keys()):
        #     if slot in inform_slots:
        #         self.state["inform_slots"][slot] = goal["implicit_inform_slots"][slot]

        for slot in goal["implicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "implicit_inform_slots" # Remember where the rest slot comes from.
        for slot in goal["explicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "explicit_inform_slots"
        for slot in goal["request_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "request_slots"
        user_action = self._assemble_user_action()
        return user_action

    def _init(self,dataset, goal_index=None):
        """
        Initializing an instance or an episode. Choosing one goal for a new dialogue session.
        :return: Nothing
        """
        self.state = {
            "turn":0,
            "action":None,
            "history":{}, # For slots that have been informed.
            "request_slots":{}, # For slots that user requested in this turn.
            "inform_slots":{}, # For slots that belong to goal["request_slots"] or other slots not in explicit/implicit_inform_slots.
            "explicit_inform_slots":{}, # For slots that belong to goal["explicit_inform_slots"]
            "implicit_inform_slots":{}, # For slots that belong to goal["implicit_inform_slots"]
            "rest_slots":{} # For slots that have not been informed.
        }

        if goal_index is None:
            self.goal = random.choice(self.goal_set[dataset])
        else:
            self.goal = self.goal_set[dataset][goal_index]

        self.episode_over = False
        self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET
        self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_FAILURE

    def _assemble_user_action(self):
        """
        Assembling the user action according to the current status.
        Returns:
            A dict, containing the information of this turn and the user's current state.
        """
        user_action = {
            "turn":self.state["turn"],
            "action":self.state["action"],
            "speaker":"user",
            "request_slots":self.state["request_slots"],
            "inform_slots":self.state["inform_slots"],
            "explicit_inform_slots":self.state["explicit_inform_slots"],
            "implicit_inform_slots":self.state["implicit_inform_slots"]
        }
        return user_action

    def next(self, agent_action, turn):
        """
        Responding to the agent. Call different responding functions for different action types.
        Args:
            agent_action: a dict, the action of agent, see the definition of agent action in the Agents.
            turn: int, indicating the current turn of this dialgue session.
        Returns:
            A tuple:
                user_action: a dict, the user action returned by the _assemble_action funciton.
                reward: float, the immediate reward for this turn.
                episode_over: bool, indicating whether the current session is terminated or not.
                dialogue_status: string, indicating the dialogue status after this turn.
        """
        # Exceed the limited maximum dialogue turn. This session terminated as failure.
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn
        if self.parameter.get("use_all_labels"):
            temp_turn = self.max_turn + 20
        else:
            temp_turn = self.max_turn - 2
        #if self.state["turn"] == (self.max_turn - 2):
        if self.state["turn"] == temp_turn:
            self.episode_over = True
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            #self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN
            #print('##########################################')
        else:
            pass

        # Within the maximum dialogue turn and the session does not terminate.
        if self.episode_over is not True:
            # Updating the history of state with the mentioned slots in the user's action in this turn.
            # TODO: the request_slots in state should be considered?
            self.state["history"].update(self.state["inform_slots"])
            self.state["history"].update(self.state["explicit_inform_slots"])
            self.state["history"].update(self.state["implicit_inform_slots"])

            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()

            # Response according to different action type.
            if agent_act_type == dialogue_configuration.CLOSE_DIALOGUE:
                self._response_closing(agent_action=agent_action)
            elif agent_act_type == dialogue_configuration.THANKS:
                self._response_thanks(agent_action=agent_action)
            elif agent_act_type == "confirm_answer":
                self._response_confirm_answer(agent_action=agent_action)
            elif agent_act_type == "confirm_question": # The "confirm_question" case is similar to "confirm_answer".
                self._response_confirm_answer(agent_action=agent_action)
            elif agent_act_type == "deny": # The "deny" case is similar to "confirm_answer".
                self._response_confirm_answer(agent_action=agent_action)
            elif agent_act_type == "inform":
                self._response_inform(agent_action=agent_action)
            elif agent_act_type == "explicit_inform":
                self._response_inform(agent_action=agent_action)
            elif agent_act_type == "implicit_inform":
                self._response_inform(agent_action=agent_action)
            elif agent_act_type == "request":
                self._response_request(agent_action=agent_action)
        else:# this session is terminated, return the results directly.
            pass

        # Check the related symptoms if the dialogue status is success.
        if self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS and self.parameter.get("check_related_symptoms")==True:
            self.check_disease_related_symptoms()
        user_action = self._assemble_user_action()
        reward = self._reward_function()
        return user_action, reward, self.episode_over, self.dialogue_status

    def _response_closing(self, agent_action):
        self.state["action"] = dialogue_configuration.THANKS
        self.episode_over = True


    #############################################
    # Response for request where explicit_inform_slots and implicit_slots are handled in the same way.
    ##############################################
    def _response_request(self, agent_action):
        """
        The user informs slot must be one of implicit_inform_slots, because the explicit_inform_slots are all informed
        at beginning.
        # It would be easy at first whose job is to answer the implicit slot requested by agent.
        :param agent_action:
        :return:
        """
        # TODO (Qianlong): response to request action.
        if len(agent_action["request_slots"].keys()) > 0:
            for slot in agent_action["request_slots"].keys():
                # The requested slots are come from explicit_inform_slots.
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    # For requesting right symptoms of the user goal.
                    self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    # For requesting right symptoms of the user goal.
                    self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                # The requested slots not in the user goals.
                else:
                    # All the symptom in the user goal are mentioned and the diseased is also informed by the agent.
                    if len(self.state["request_slots"].keys()) == 0 and len(self.state["rest_slots"].keys()) == 0:
                        self.state["action"] = dialogue_configuration.THANKS
                    else:
                        self.state["action"] = "not_sure"
                        self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_KNOW

        # The case where the agent action type is request, but nothing in agent request_slots, which should not appear.
        # A randomized slot will be chosen to inform agent if the rest_slots is not empty.
        else:
            if len(self.state["rest_slots"].keys()) > 0:
                rest_slots = copy.deepcopy(self.state["rest_slots"])
                rest_slots.pop("disease")
                slot = random.choice(rest_slots.keys())
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():# The case should not appear.
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                # The selected slot comes from neither explicit slots nor implicit slots, this case should not appear.
                else:
                    self.state["action"] = "not_sure"
                    self.state["inform_slots"] = dialogue_configuration.I_DO_NOT_KNOW
            else:
                self.state["action"] = dialogue_configuration.THANKS

    #############################################
    # Response confirm_answer where explicit_inform_slots and implicit_inform_slots are handled in the same way.
    #############################################
    def _response_confirm_answer(self, agent_action):
        # TODO (Qianlong): response to confirm answer action. I don't think it is logically right, but there is no slot
        # TODO: in either inform_slots or request_slots when the action type is "confirm answer" in the action space of agent.
        if len(self.state["rest_slots"].keys()) > 0:
            slot = random.choice(list(self.state["rest_slots"].keys()))
            if slot in self.goal["goal"]["request_slots"].keys():
                self.state["action"] = "request"
                self.state["request_slots"][slot] = dialogue_configuration.VALUE_UNKNOWN
            elif slot in self.goal["goal"]["explicit_inform_slots"].keys():
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
            elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
        else:
            self.state["action"] = dialogue_configuration.THANKS

    ##########################################
    # Response for thanks.
    ##########################################
    def _response_thanks(self, agent_action):
        # TODO (Qianlong): response to thanks action.
        self.episode_over = True
        self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_SUCCESS

        request_slot_set = copy.deepcopy(list(self.state["request_slots"].keys()))
        if "disease" in request_slot_set:
            request_slot_set.remove("disease")
        rest_slot_set = copy.deepcopy(list(self.state["rest_slots"].keys()))
        if "disease" in rest_slot_set:
            rest_slot_set.remove("disease")

        # The dialogue is failed if there are still slots in rest_slots or request_slots.
        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED

        # The dialogue is failed if the value for any slot is wrong.
        for slot in self.state["history"].keys():
            if slot in self.goal["goal"]["explicit_inform_slots"].keys() and \
                self.state["history"][slot] != self.goal["goal"]["explicit_inform_slots"][slot]:
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
            elif slot in self.goal["goal"]["implicit_inform_slots"].keys() and \
                self.state["history"][slot] != self.goal["goal"]["implicit_inform_slots"][slot]:
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
        # The dialogue is failed if wrong disease is informed.
        if "disease" in agent_action["inform_slots"].keys():
            if agent_action["inform_slots"]["disease"] != self.goal["disease_tag"]:
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED

    ##########################################
    # Response for inform where explicit_inform_slots and implicit_inform_slots are handled in the same way.
    ##########################################
    def _response_inform(self, agent_action):
        # TODO (Qianlong): response to inform action.
        agent_all_inform_slots = copy.deepcopy(agent_action["inform_slots"])
        agent_all_inform_slots.update(agent_action["explicit_inform_slots"])
        agent_all_inform_slots.update(agent_action["implicit_inform_slots"])

        user_all_inform_slots = copy.deepcopy(self.goal["goal"]["explicit_inform_slots"])
        user_all_inform_slots.update(self.goal["goal"]["implicit_inform_slots"])

        # The agent informed the right disease and the current dialogue session is over.
        if "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] == self.goal["disease_tag"]:
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_SUCCESS
            self.state["history"]["disease"] = agent_action["inform_slots"]["disease"]
            self.episode_over = True
            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
            self.state["request_slots"].pop("disease")
            if "disease" in self.state["rest_slots"]: self.state["rest_slots"].pop("disease")
        # The agent informed wrong disease and the dialogue will go on if not reach the max_turn.
        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] != self.goal["disease_tag"]:
            # The user denys the informed disease, and the dialogue will going on.
            if self.parameter["allow_wrong_disease"] == True:
                self.state["action"] = "deny"
                self.state["inform_slots"]["disease"] = agent_action["inform_slots"]["disease"]
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE
            # The informed disease is wrong, and the dialogue is failed.
            else:
                self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
                self.episode_over = True
                self.state["inform_slots"].clear()
                self.state["explicit_inform_slots"].clear()
                self.state["implicit_inform_slots"].clear()

        # No disease is informed in the agent action.
        else: # Task is not completed.
            for slot in agent_all_inform_slots.keys():
                # The slot comes from explicit/implicit_inform_slots of user.
                if slot in user_all_inform_slots.keys():
                    # Agent informed correct slot.
                    if agent_all_inform_slots[slot] == user_all_inform_slots[slot]:
                        # dialogue status changes into DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM.
                        self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM
                        self.state["history"][slot] = agent_all_inform_slots[slot]
                        if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot) # pop this slot from left slots.

                        if len(self.state["request_slots"].keys()) > 0: # the act type is `request` if there exists slot in `request_slots`
                            self.state["action"] = "request"
                            # confirm the slot-value.
                            self.state["action"] = "confirm_answer"
                            self.state["inform_slots"][slot] = agent_all_inform_slots[slot]

                        elif len(self.state["rest_slots"]) > 0:# The state["rest_slots"] is not empty.
                            rest_slot_set = copy.deepcopy(list(self.state['rest_slots'].keys()))
                            if "disease" in rest_slot_set:
                                rest_slot_set.remove("disease")

                            if len(rest_slot_set) > 0: # Not the `disease` slots.
                                inform_slot = random.choice(rest_slot_set)
                                if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                    self.state["inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                    self.state["action"] = "inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                    self.state["inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                    self.state["action"] = "inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                # This case will not appear in medical dialogue system.
                                elif inform_slot in self.goal["goal"]["request_slots"].keys():
                                    self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                                    self.state["rest_slots"].pop(inform_slot)
                            # No slots left except for "disease" which is used for request.
                            else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"

                    # Agent informed wrong slot-value. In this system this case would never appear. Because the value of
                    # informed slots in the action space are True.
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                            self.state["action"] = "inform"
                            self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                        elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                            self.state["action"] = "inform"
                            self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                        else:
                            self.state["action"] = "inform"
                            self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_KNOW

                        if slot in self.state["rest_slots"]: self.state["rest_slots"].pop(slot)

                # The slot agent informed is not in the user explicit/implicit slots, which means the informed slots may
                # in the user["request_slots"] or not in the request_slots and explicit/implicit_slots of the user.
                # the "disease" case has been handled specially.
                # TODO: I think this should deny the wrong informed slot. Attention.
                else:
                    if slot in self.state["request_slots"].keys(): self.state["request_slots"].pop(slot)
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

                    # The informed slot is neither disease nor the symptom that the user has.
                    if slot != "disease" and slot not in user_all_inform_slots:
                        self.state["action"] = "not_sure"
                        self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_KNOW
                    elif len(self.state["request_slots"]) > 0:
                        request_slot_set = list(self.state["request_slots"].keys())
                        if "disease" in request_slot_set:
                            request_slot_set.remove("disease")

                        if len(request_slot_set) > 0:
                            request_slot = random.choice(request_slot_set)
                        else:
                            request_slot = "disease"
                        self.state["request_slots"][request_slot] = dialogue_configuration.VALUE_UNKNOWN
                        self.state["action"] = "request"
                    elif len(self.state["rest_slots"].keys()) > 0:
                        rest_slot_set = list(self.state["rest_slots"].keys())
                        if "disease" in rest_slot_set: rest_slot_set.remove("disease")
                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set)
                            if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                self.state["inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                self.state["action"] = "inform"
                                self.state["rest_slots"].pop(inform_slot)
                            elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                self.state["inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                self.state["action"] = "inform"
                                self.state["rest_slots"].pop(inform_slot)
                            elif inform_slot in self.goal["goal"]["request_slots"].keys():  # This case will not appear
                                self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                        else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                    else:
                        self.state["action"] = dialogue_configuration.THANKS

    def _check_slots(self):
        """
        TODO: the same as the next function?
        Check whether all the explicit slots, implicit slots and request slots are correctly informed.
        Returns:
            bool, True:
        """
        informed_slots = list(self.state["history"].keys())
        all_slots = copy.deepcopy(self.goal["goal"]["request_slots"])
        all_slots.update(self.goal["goal"]["explicit_inform_slots"])
        all_slots.update(self.goal["goal"]["implicit_inform_slots"])

        for slot in all_slots.keys():
            if slot not in informed_slots:
                return False
        return True

    def _informed_all_slots_or_not_(self):
        """
        Whether all the inform_slots and request_slots in the user goal are informed.
        Returns:
            bool, True: all the slots have been mentioned, False: not all slots have been mentioned.
        """
        if len(self.state["rest_slots"].keys()) > 0:
            return False
        else:
            return True

    def _reward_function(self):
        """
        Return a reward for this turn according to the dialoge status.
        Returns:
            A float, the immediate reward for this turn.
        """
        if self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
            return self.parameter.get("reward_for_not_come_yet")
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
            success_reward = self.parameter.get("reward_for_success")
            if self.parameter.get("minus_left_slots") == True:
                return success_reward - len(self.state["rest_slots"])
            else:
                return success_reward
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
            return self.parameter.get("reward_for_fail")
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_DISEASE
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM:
            return self.parameter.get("reward_for_inform_right_symptom")
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            return self.parameter.get("reward_for_reach_max_turn")

    def get_goal(self):
        return self.goal

    def __prepare_goal_set__(self, goal_set, parameter):
        explicit_number = parameter.get('explicit_number')
        implicit_number = parameter.get('implicit_number')
        temp_goal_set = {}
        disease_sample_count = {}
        for key in goal_set.keys():
            temp_goal_set[key] = []
            for goal in goal_set[key]:
                append_or_not = False
                if len(goal["goal"]["explicit_inform_slots"].keys()) >= explicit_number and \
                        len(goal["goal"]["implicit_inform_slots"].keys()) >= implicit_number:
                    append_or_not = True

                if append_or_not:
                    temp_goal_set[key].append(goal)
                    disease_sample_count.setdefault(goal["disease_tag"],0)
                    disease_sample_count[goal["disease_tag"]] += 1
            print(key, len(temp_goal_set[key]))
        return temp_goal_set, disease_sample_count

    def set_max_turn(self, max_turn):
        self.max_turn = max_turn

    def check_disease_related_symptoms(self):
        """
        This function will be called only if dialogue status is successful to check whether the symptoms that related to the
        predicted disease have been all mentioned so far. If yes, the dialogue status still be success, otherwise, it
        will be changed into fail.
        Raise:
            Raise key error if the 'disease' not in the key of state['history'], i.e., the agent has not informed the
            right disease yet.
        """
        '''
        # inform all related symptoms.
        all_mentioned_slots = self.state["history"].keys()
        if "disease" not in all_mentioned_slots:
            raise KeyError("'disease' not in the keys of state['history']")
        disease_pred = self.state["history"]["disease"]
        # Get the related symptoms.
        related_symptoms = self.disease_symptom[disease_pred]["symptom"]
        # print("mentioned slots", all_mentioned_slots)
        # print("related slots", related_symptoms)
        for slot in related_symptoms:
            if slot not in all_mentioned_slots:
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
                break
        '''

        # Mentioned at least two symptoms which are not in the `explicit_inform_slots` of user goal.
        all_mentioned_slots = copy.deepcopy(self.state["history"])
        count = 0
        all_mentioned_slots.pop("disease")
        for key in all_mentioned_slots.keys():
            if key not in self.goal["goal"]["explicit_inform_slots"].keys():
                count += 1

        if count < 2:
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED