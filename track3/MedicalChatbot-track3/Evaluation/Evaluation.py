import pickle
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--result_path", dest="result_path", type=str, default='./result.json',help="the path of result .json")
parser.add_argument("--goal_set_path", dest="goal_set_path", type=str, default=".//goal_set_simul.p", help="the device for tensorflow running on.")

args = parser.parse_args()
parameter = vars(args)

result_dict = json.load(open(parameter.get('result_path'),'r'))
goal_dict = pickle.load(file=open(parameter["goal_set_path"], "rb"))['test']

ture_predict = 0
recall_score = 0

for value in goal_dict:
    ids = value['consult_id']
    if value['disease_tag'] == result_dict[ids]['Disease']:
        ture_predict += 1
    goal_symptom = list(value['goal']['explicit_inform_slots']) + list(value['goal']['implicit_inform_slots'])
    result_symptom = result_dict[ids]['Symptoms']
    recall_score += len(list(set(goal_symptom).intersection(set(result_symptom)))) / len(set(goal_symptom)) 
    #print(2 * len(list(set(goal_symptom).intersection(set(result_symptom)))) / (len(set(goal_symptom)) + len(set(result_symptom))))
print('Acc = ', ture_predict / len(goal_dict), 'Recall_symptom = ', recall_score / len(goal_dict))