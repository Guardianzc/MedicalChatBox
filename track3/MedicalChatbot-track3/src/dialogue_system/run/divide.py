import pickle

goal_set_all = pickle.load(file=open('./src/dialogue_system/data/dataset/label/goal_set.p', "rb"))
goal_set = dict()
goal_set['train'] = goal_set_all['train']
goal_set['dev'] = goal_set_all['dev']
goal_set_test = dict()
goal_set_test['test'] = goal_set_all['test']
pickle.dump(goal_set, open('./src/dialogue_system/data/dataset/label/goal_train_set.p','wb'))
pickle.dump(goal_set_test, open('./src/dialogue_system/data/dataset/label/goal_test_set.p', 'wb'))

action = dict()
action['request'] = 0
action['inform'] = 1
action['closing'] = 2
pickle.dump(action, open('./src/dialogue_system/data/dataset/label/action_set_1.p', "wb"))
pass