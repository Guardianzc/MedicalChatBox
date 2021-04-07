
import pickle

class Goal2Slot(object):
    def __init__(self):
        pass

    def load_goal(self,goal_file):
        slot_set = set()
        goal_set = pickle.load(open(goal_file,"rb"))
        for key in goal_set.keys():
            for goal in goal_set[key]:
                for symptom in goal["goal"]["explicit_inform_slots"].keys():
                    slot_set.add(symptom)
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    slot_set.add(symptom)
        self.slot_set = list(slot_set)
        print(len(self.slot_set))


if __name__ == "__main__":
    goal_file = "./../data/dataset/1200/0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN/goal_set_2.p"
    goal2slot = Goal2Slot()
    goal2slot.load_goal(goal_file)
