# -*- coding: utf-8 -*-
"""
使用主诉里面获得症状进行分类，把疾病判断看成一个分类任务；
"""

import copy
import json
import random
import pickle
import numpy as np
import torch
import sys, os
from sklearn import svm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

sys.path.append(os.getcwd().replace("src/classifier/symptom_as_feature",""))

def progress_bar(bar_len, acc_train, SR, best_train, best_SR,  currentNumber, wholeNumber):
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mtrain_SR\033[0m = %3f \033[31;1mSR\033[0m = %3f \033[33;1mBest_train\033[0m = %3f \033[33;1mBest_SR\033[0m= %3f  \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', acc_train, SR, best_train, best_SR))
    sys.stdout.flush()



class MLP(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # different layers. Two layers.???
        self.MLP_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        embedding = self.MLP_layer(x)
        return embedding


class SymptomClassifier(object):
    def __init__(self, goal_set,symptom_set, disease_symptom, hidden_size,parameter, fold, fold_num):
        #self.k_fold = k_fold
        self.goal_set = goal_set
        self.hidden_size = hidden_size
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self.wrong_samples = {}
        self.fold = fold
        self.fold_num = fold_num
        self._disease_index(disease_symptom=disease_symptom)
        self._symptom_index(symptom_set=symptom_set)
        self._prepare_data_set()

        print(self.disease_to_index)

    def _symptom_index(self, symptom_set):
        """
        Mapping symptom to index and index to symptom.
        :param symptom_set:
        :return:
        """
        index = 0
        symptom_to_index = {}
        index_to_symptom = {}
        if "disease" in symptom_set.keys():
            symptom_set.pop("disease")
        for key, value in symptom_set.items():
            symptom_to_index[key] = index
            index_to_symptom[index] = key
            index += 1
        self.symptom_to_index = symptom_to_index
        self.index_to_symptom = index_to_symptom

    def _disease_index(self, disease_symptom):
        """
        Mapping disease to index and index to disease.
        :param disease_symptom:
        :return:
        """
        index = 0
        disease_to_index = {}
        index_to_disease = {}
        for key in disease_symptom.keys():
            disease_to_index[key] = index
            index_to_disease[index] = key
            index += 1
        self.disease_to_index = disease_to_index
        self.index_to_disease = index_to_disease

    def split_data(self, dataset):
        return_dataset = list()
        for goal in dataset:
            '''
            append_or_not = self.__keep_sample_or_not__(goal)
            if append_or_not:
            '''
            return_dataset.append(goal)  
        return return_dataset

    def _prepare_data_set(self):
        """
        Preparing the dataset for training and evaluating.
        :return:
        """
        disease_sample_count = {}
        sample_by_disease = {}
        data_set = {}
        fold = self.fold
        fold_num = self.fold_num
        self.goal_set["train"] = self.split_data(self.goal_set["train"])
        #self.goal_set["test"] = self.split_data(self.goal_set["test"])
        self.goal_set["dev"] = self.split_data(self.goal_set["dev"])
        train_sample = self.goal_set["train"] + self.goal_set["dev"] #+ self.goal_set["test"] 
        if fold == False:            
            fold_list = dict()
            fold_list[0] = self.goal_set['train']
            fold_list[1] = self.goal_set['dev']
            #fold_list[2] = self.goal_set['test']
            fold_num = 2
        else:
            all_sample = self.goal_set["train"] + self.goal_set["dev"] #+ self.goal_set["test"] 
            random.shuffle(all_sample)
            fold_size = int(len(all_sample) / fold_num)
            fold_list = [all_sample[i:i+fold_size] for i in range(0,len(all_sample),fold_size)]

            
        for k in range(fold_num):
            data_set[k] = {
                "x_ex":[],
                "x_im":[],
                "x_ex_im":[],
                "y":[],
                "consult_id":[]
            }
            fold = fold_list[k]
            for goal in fold:
                disease_rep = np.zeros(len(self.disease_to_index.keys()))
                disease_rep[self.disease_to_index[goal["disease_tag"]]] = 1
                symptom_rep_ex = np.zeros(len(self.symptom_to_index.keys()))
                symptom_rep_im = np.zeros(len(self.symptom_to_index.keys()))
                symptom_rep_ex_im = np.zeros(len(self.symptom_to_index.keys()))
                # explicit
                for symptom, value in goal["goal"]["explicit_inform_slots"].items():
                    if value == True:
                        symptom_rep_ex[self.symptom_to_index[symptom]] = 1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = 1
                    else:
                        symptom_rep_ex[self.symptom_to_index[symptom]] = -1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1

                # implicit
                for symptom, value in goal["goal"]["implicit_inform_slots"].items():
                    if value == True:
                        symptom_rep_im[self.symptom_to_index[symptom]] = 1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = 1
                    else:
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1
                        symptom_rep_im[self.symptom_to_index[symptom]] = -1
                # print(data_set)
                '''
                append_or_not = self.__keep_sample_or_not__(goal)
                if append_or_not:
                '''
                sample_by_disease.setdefault(goal["disease_tag"], dict())
                sample_by_disease[goal["disease_tag"]][goal["consult_id"]] = goal

                disease_sample_count.setdefault(goal["disease_tag"],0)
                disease_sample_count[goal["disease_tag"]] += 1

                data_set[k]["x_ex"].append(symptom_rep_ex)
                data_set[k]["x_im"].append(symptom_rep_im)
                data_set[k]["x_ex_im"].append(symptom_rep_ex_im)
                data_set[k]["y"].append(disease_rep)
                data_set[k]["consult_id"].append(goal["consult_id"])

        self.data_set = data_set
        self.sample_by_disease = sample_by_disease
        self.disease_sample_count = disease_sample_count

    def MLP_dataset(self, dataset, batch_size):
        train_feature = self.parameter.get("train_feature")
        test_feature = self.parameter.get("test_feature")
        Xs = []
        Ys = []
        Ys = Ys + dataset['y']
        if train_feature == "ex":
            Xs = Xs + dataset["x_ex"]
        elif train_feature == "im":
            Xs = Xs + dataset["x_im"]
        elif train_feature == "ex&im":
            Xs = Xs + dataset["x_ex_im"]
        Xs = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        Ys = torch.tensor(Ys, dtype=torch.float32).to(self.device)
        ids = TensorDataset(Xs, Ys) 
        loader = DataLoader(dataset=ids, batch_size = batch_size, shuffle=True)
        return loader
    
    def MLP_train(self):
        fold = self.fold
        fold_num = self.fold_num
        if fold:
            for key in self.data_set.keys():
                print("fold index:", key)
                train_set = copy.deepcopy(self.data_set)
                test_set = train_set.pop(key)
                new_train_set = {'x_ex':list(), 'x_ex_im':list(), 'x_im':list(), 'y':list(), 'consult_id':list()}
                for key,values in train_set.items():
                    for list_class, value in values.items():
                        new_train_set[list_class] += value
                self.MLP_build(new_train_set, test_set, test_set)
        else:
            self.MLP_build(self.data_set[0], self.data_set[1], self.data_set[1])
            
    def MLP_build(self, train_set, dev_set, test_set):

        self.MLP = MLP(254, 4)
        self.device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(list(self.MLP.parameters()), 0.001, (0.5, 0.999), weight_decay=0.01)
        self.MLP.to(self.device)
        self.criterion = torch.nn.BCELoss()

        train_loader = self.MLP_dataset(dataset=train_set, batch_size=16)  
        dev_loader = self.MLP_dataset(dataset=dev_set, batch_size=1)  
        test_loader = self.MLP_dataset(dataset=test_set, batch_size=1)  
        best_acc = 0
        for i in range(1000):
            acc_train = 0
            length = 0
            self.MLP.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.type(torch.float32)
                batch_y = batch_y.type(torch.float32)
                predict_y = self.MLP(batch_x)
                loss = self.criterion(predict_y, batch_y)
                for j in range(batch_y.shape[0]):
                    if torch.argmax(predict_y[j]) == torch.argmax(batch_y[j]):
                        acc_train += 1
                length += batch_y.shape[0]
                self.MLP.zero_grad()  # zero the gradient buffers, not zero the parameters
                loss.backward()
                self.optimizer.step()
            acc_train = acc_train / length
            acc_dev = 0
            length = 0
            self.MLP.eval()
            for step, (batch_x, batch_y) in enumerate(dev_loader):
                batch_x = batch_x.type(torch.float32)
                predict_y = self.MLP(batch_x)
                if torch.argmax(predict_y) == torch.argmax(batch_y):
                    acc_dev += 1
                length += 1
            acc = acc_dev / length
            if best_acc < acc:
                best_train = acc_train
                best_acc = acc
                torch.save(self.MLP, 'best_MLP.pkl')
            progress_bar(10, acc_train, acc, best_train ,best_acc,  i, 1000)
        print('dev = ', best_acc)
        #######################  test ######################
        acc_num = 0
        length = 0
        best_MLP = torch.load('best_MLP.pkl')
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.type(torch.float32)
            predict_y = best_MLP(batch_x)
            if torch.argmax(predict_y) == torch.argmax(batch_y):
                acc_num += 1
            length += 1
        print('test = ', acc_num / length)
        

    def SVM_train(self):
        fold = self.fold
        fold_num = self.fold_num
        if fold:
            for key in self.data_set.keys():
                print("fold index:", key)
                train_set = copy.deepcopy(self.data_set)
                test_set = train_set.pop(key)
                self.train_sklearn_svm(train_set, test_set)
        else:
            print('dev = ')
            self.train_sklearn_svm({'train':self.data_set[0]}, self.data_set[1])
            '''
            print('test = ')
            self.train_sklearn_svm({'train':self.data_set[0]}, self.data_set[2])
            '''
    def train_sklearn_svm(self, train_set, test_set):
        disease_accuracy = {}
        disease_accuracy["total_accuracy"] = {}
        disease_accuracy["total_accuracy"]["ex&im"] = 0.0
        disease_accuracy["total_accuracy"]["ex"] = 0.0

        for key in self.sample_by_disease.keys():
            disease_accuracy[key] = {}
            disease_accuracy[key]["ex&im"] = 0.0
            disease_accuracy[key]["ex"] = 0.0

        for key in self.data_set.keys():
            temp_accuracy_ex_im,temp_accuracy_ex = self._train_and_evaluate_svm_one_fold_(train_set, test_set)
            '''
            print(temp_accuracy_ex_im)
            print(temp_accuracy_ex)
            '''
            for key in temp_accuracy_ex_im.keys():
                disease_accuracy[key]["ex&im"] += temp_accuracy_ex_im[key]["accuracy"]
                disease_accuracy[key]["ex"] += temp_accuracy_ex[key]["accuracy"]

        for key,value in disease_accuracy.items():
            disease_accuracy[key]["ex&im"] = float("%.4f" % (value["ex&im"] / len(self.data_set.keys())))
            disease_accuracy[key]["ex"] = float("%.4f" % (value["ex"] / len(self.data_set.keys())))

        print(disease_accuracy)

    def _train_and_evaluate_svm_one_fold_(self, train_set, test_set):
        """

        :param train_set: dict, {"fold_index":{"x":[],"x_ex":[]]}
        :param test_set: a list of batches.
        :return:
        """
        train_feature = self.parameter.get("train_feature")
        test_feature = self.parameter.get("test_feature")
        clf = svm.SVC(decision_function_shape="ovo", gamma='auto')
        Xs = []
        Ys = []
        for fold in train_set.values():
            Ys = Ys + list(np.argmax(fold['y'], axis=1))
            if train_feature == "ex":
                Xs = Xs + fold["x_ex"]
            elif train_feature == "im":
                Xs = Xs + fold["x_im"]
            elif train_feature == "ex&im":
                Xs = Xs + fold["x_ex_im"]
        clf.fit(X=Xs, y=Ys)

        # Test
        IDs = test_set["consult_id"]
        Ys = list(np.argmax(test_set['y'],axis=1))
        Xs_ex = test_set["x_ex"]
        Xs_im = test_set["x_im"]
        Xs_ex_im = test_set["x_ex_im"]
        predict_ys_ex = clf.predict(Xs_ex)
        predict_ys_ex_im = clf.predict(Xs_ex_im)
        disease_accuracy_ex = self._accuracy_for_each_disease(labels=Ys,predicted_ys=predict_ys_ex, IDs=IDs)
        disease_accuracy_ex_im = self._accuracy_for_each_disease(labels=Ys,predicted_ys=predict_ys_ex_im, IDs=IDs)
        return disease_accuracy_ex_im, disease_accuracy_ex


    def _accuracy_for_each_disease(self, labels, predicted_ys,IDs):
        disease_accuracy = {}
        disease_accuracy["total_accuracy"]={}
        disease_accuracy["total_accuracy"]["accuracy"] = 0.0
        count = 0.0

        import csv
        train_feature = self.parameter.get("train_feature")
        test_feature = self.parameter.get("test_feature")
        # error_id_file = open("/Users/qianlong/Desktop/error_id_"+train_feature+"_" + test_feature+".csv","a+",encoding="utf8")
        # writer = csv.writer(error_id_file)

        for disease in self.disease_sample_count.keys():
            disease_accuracy[disease] = {}
            disease_accuracy[disease]["success_count"] = 0.0
            disease_accuracy[disease]["total"] = 0.0
            disease_accuracy["total_accuracy"]["accuracy"] = 0.0
        for sample_index in range(0, len(labels), 1):
            disease_accuracy[self.index_to_disease[labels[sample_index]]]["total"] += 1
            if labels[sample_index] == predicted_ys[sample_index]:
                count += 1
                disease_accuracy[self.index_to_disease[labels[sample_index]]]["success_count"] += 1

        for disease in self.disease_sample_count.keys():
            disease_accuracy[disease]["accuracy"] = disease_accuracy[disease]["success_count"] / disease_accuracy[disease]["total"]
        disease_accuracy["total_accuracy"]["accuracy"] = count / len(labels)
        # error_id_file.close()
        return disease_accuracy
