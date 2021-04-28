
Loading this dataset using the following command in Python:
    import pickle
    data_set = pickle.load(open(file_name, 'rb'))

1. goal_set.p: the goal set used in the code. The goal_set contains training set and testing set, which can be visited with goal_set["train"] and goal_set["test"]. Each sub-set is a list of user goals expalined in our paper and each user goal is an dictionary which has three keys, "consult_id" is the user id, "group_id" is the group to which the user goal belongs, "disease_tag" is the disease that the user suffers and  "goal" is the combination of slots (request slots, implicit symptoms and explicit symptoms).

2. action_set.p: the types of action pre-defined for this medical DS.

3. disease_symptom.p: the collection of symptoms for each disease.

4. slot_set.p: the set of slots, which consists of normalized symptoms and a special slot diseaseas as explained in our paper.

Now the goal set contains 30,000 user goals and 90 diseases in total, in addition, it contains 9 groups of diseases and each group includes 10 diseases.

The group id we selected is among [1,4,5,6,7,12,13,14,19], which corresponds to the chapter number in ICD-10-CM.

The folder from label1 to label19 is used for the training of low-level policy for each worker. 

Each folder contains its own  "goal_set.p", "action_set.p", "disease_symptom.p" and "slot_set.p", which only corresponds to the diseases in this gorup.

Please see our paper for details.