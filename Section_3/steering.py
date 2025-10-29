"""
This file is responsible for the intervention by detection
"""

import functools
import gc
import json
import os
import pickle
import subprocess
import time

import psutil
from torch.nn import CrossEntropyLoss
from typing import List, Tuple

import numpy as np
import torch
import transformers

from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from models_config import get_model_config
import model_inside

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random

set_seed(42)
torch.manual_seed(42)
generation_length = 5


class InterventionByDetection():

    def __init__(self, path_to_results, data_path_without_hallucinations, data_path_with_hallucinations, model_name,
                 dataset_size, dataset_name, threshold_of_data, use_mlp, use_attention, use_heads, use_residual,
                 alpha=5, static_intervention=False, concatenate_answer=False,
                 on_test_set=False, seed_train_val=None, general_data_path=None,
                 use_classifier_for_intervention=False,path_general_test_set=None,alice=False, persona=False, truthfulness=False, fake_alignment = False, realistic=False, hk_=False):
        self.seed_train_val = seed_train_val

        self.path_general_test_set = path_general_test_set
        self.model_name = model_name
        self.intervention = None
        self.alpha = alpha
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.hallucinations_examples = self.load_dataset(data_path_with_hallucinations)
        self.non_hallucinations_examples = self.load_dataset(data_path_without_hallucinations)
        self.general_examples = self.load_dataset(general_data_path) if general_data_path is not None else None
        self.path_to_save_results = path_to_results + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold_of_data}/concat_answer{concatenate_answer}_size{dataset_size}/"
        start_time = time.time()
        MLPCheck = ModelInside.ModelInside(path_to_results,
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=threshold_of_data,concat_answer=concatenate_answer,alice=alice, persona=persona, truthfulness=truthfulness, fake_alignment = False, realistic=realistic)
        self.all_mlp_vector_with_hall, self.all_attention_vector_with_all, self.all_mlp_vector_without_hall, self.all_attention_vector_without_hall, self.heads_vectors_with, self.heads_vectors_without, self.all_residual_with, self.all_residual_without = MLPCheck.load_all_data()
        if not hk_:
            self.bad_indecess = []
            for i, e in enumerate(self.general_examples):
                if e[-2] != 0:
                    self.bad_indecess.append(i)
            torch.cuda.empty_cache()
            gc.collect()
            self.general_examples = [self.general_examples[i] for i, e in enumerate(self.general_examples) if i not in self.bad_indecess]
        #this is to run on HK- hall
        else:
            # self.bad_indecess = [i for i,e in enumerate(self.hallucinations_examples) if e[-2]!=0]
            self.bad_indecess = [] # using hk- examples
            for i, e in enumerate(self.hallucinations_examples):
                if e[-2] != 0 :
                    self.bad_indecess.append(i)
            self.hallucinations_examples = [self.hallucinations_examples[i] for i, e in enumerate(self.hallucinations_examples) if i not in self.bad_indecess]

            torch.cuda.empty_cache()
            gc.collect()
            self.all_mlp_vector_type1, self.all_attention_vector_type1, self.heads_vectors_type1, self.all_residual_vectors_type1 = MLPCheck.get_type_1_data(
                path_to_results, self.bad_indecess)

            self.all_mlp_vector_with_hall = self.all_mlp_vector_type1[
                                            :min(len(self.all_mlp_vector_with_hall), len(self.all_mlp_vector_type1))]
            self.all_attention_vector_with_all = self.all_attention_vector_type1[
                                                 :min(len(self.all_attention_vector_with_all),
                                                      len(self.all_attention_vector_type1))]
            self.heads_vectors_with = self.heads_vectors_type1[
                                      :min(len(self.heads_vectors_with), len(self.heads_vectors_type1))]
            self.all_residual_with = self.all_residual_vectors_type1[
                                     :min(len(self.all_residual_with), len(self.all_residual_vectors_type1))]
        # those are the else examples:

        assert len(self.all_mlp_vector_with_hall) == len(self.all_attention_vector_with_all) == len(
            self.all_mlp_vector_without_hall) == len(self.all_attention_vector_without_hall) == len(
            self.heads_vectors_with) == len(self.heads_vectors_without) == len(self.all_residual_with) == len(
            self.all_residual_without)



        # create a classifier for each type of information - mlp,attention,residual,heads
        self.classifiers_dict, self.acc_classification_dict = self.get_classifier_dicts(use_mlp=use_mlp,
                                                                                        use_heads=use_heads,
                                                                                        use_attention=use_attention,
                                                                                        use_residual=use_residual)
        # create an intervention dict
        self.intervention_dict = self.get_intervention_dict(use_mlp=use_mlp, use_heads=use_heads,
                                                            use_attention=use_attention, use_residual=use_residual)
        start_time = time.time()
        self.intervention = ModelIntervention(path_to_results=path_to_results, model_name=model_name,
                                              dataset_size=dataset_size, dataset_name=dataset_name,
                                              threshold_of_data=threshold_of_data, use_mlp=use_mlp,
                                              use_attention=use_attention, use_heads=use_heads,
                                              use_residual=use_residual, intervention_all_dict=self.intervention_dict,
                                              classifier_dict=self.classifiers_dict,
                                              alpha=alpha,
                                              acc_classification_val=self.acc_classification_dict,
                                              static_intervention=static_intervention)
        self.generate_examples_for_intervention(on_test_set=on_test_set)

        # create an intervention dict using classifier
        if use_classifier_for_intervention:
            self.intervention_with_classifier = self.get_intervention_for_all_data_types_based_classifier(
                self.all_mlp_vector_with_hall,
                self.all_mlp_vector_without_hall,
                self.all_attention_vector_with_all,
                self.all_attention_vector_without_hall,
                self.all_residual_with,
                self.all_residual_without,
                self.heads_vectors_with,
                self.heads_vectors_without)
            self.intervention.all_interventions = self.intervention_with_classifier

        # create an intervention dict using concatenate answer examples (post answer)
        if concatenate_answer:
            self.get_intervention_concat(MLPCheck, use_mlp, use_attention, use_heads, use_residual)

    def get_classifier_dicts(self, use_mlp, use_attention, use_heads, use_residual):
        """
        get the classifier dict and the accuracy classification dict on the validation set
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        start_time = time.time()
        classifier_dict_path = self.path_to_save_results + f"classifier_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        acc_classification_dict_path = self.path_to_save_results + f"acc_classification_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        if self.seed_train_val is not None:
            classifier_dict_path = classifier_dict_path.replace(".pkl", f"seed_{self.seed_train_val}.pkl")
            acc_classification_dict_path = acc_classification_dict_path.replace(".pkl",
                                                                                f"seed_{self.seed_train_val}.pkl")

        self.classifiers_dict, self.acc_classification_dict = self.create_classifier_for_each_type_of_info()



        return self.classifiers_dict, self.acc_classification_dict

    def get_intervention_dict(self, use_mlp, use_attention, use_heads, use_residual):
        """
        get the intervention dict, the direction to add to mitigate the hallucinations
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        start_time = time.time()
        intervention_dict_path = self.path_to_save_results + f"intervention_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        if self.seed_train_val is not None:
            intervention_dict_path = intervention_dict_path.replace(".pkl", f"seed_{self.seed_train_val}.pkl")

        self.intervention_dict = self.get_intervention_for_all_data_types(self.all_mlp_vector_with_hall,
                                                                          self.all_mlp_vector_without_hall,
                                                                          self.all_attention_vector_with_all,
                                                                          self.all_attention_vector_without_hall,
                                                                          self.all_residual_with,
                                                                          self.all_residual_without,
                                                                          self.heads_vectors_with,
                                                                          self.heads_vectors_without)

        return self.intervention_dict

    def get_intervention_concat(self, MLPCheck, use_mlp, use_attention, use_heads, use_residual):
        """
        get the intervention dict using concatenate answer examples (post-answer)
        :param MLPCheck:
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        MLPCheck.path_to_save_results = MLPCheck.path_to_save_results.replace("concat_answerFalse",
                                                                              "concat_answerTrue")
        assert "concat_answerTrue" in MLPCheck.path_to_save_results
        # self.all_mlp_vector_with_hall_concatenate_answer, self.all_attention_vector_with_all_concatenate_answer, self.all_mlp_vector_without_hall_concatenate_answer, self.all_attention_vector_without_hall_concatenate_answer, self.heads_vectors_with_concatenate_answer, self.heads_vectors_without_concatenate_answer, self.all_residual_with_concatenate_answer, self.all_residual_without_concatenate_answer = MLPCheck.load_all_data()
        self.intervention_dict_concatenate_answer = self.get_intervention_for_all_data_types(
            self.all_mlp_vector_with_hall, self.all_mlp_vector_without_hall,
            self.all_attention_vector_with_all,
            self.all_attention_vector_without_hall, self.all_residual_with,
            self.all_residual_without, self.heads_vectors_with,
            self.heads_vectors_without)


        # does intervention based on value with concatenation of answer
        self.intervention.all_interventions = self.intervention_dict_concatenate_answer


    def generate_examples_for_intervention(self, on_test_set):
        """
        generate examples for intervention
        :return:
        """
        train_with_indeces, val_with_indeces, test_with_indeces = self.split_data_to_train_val_test_for_all_data_types(
            [i for i in range(len(self.all_mlp_vector_with_hall))])
        train_with_again, _, _ = self.split_data_to_train_val_test_for_all_data_types([i for i in range(len(self.all_residual_without)) ])
        assert train_with_again == train_with_indeces, f"{train_with_again} != {train_with_indeces}"
        assert len(train_with_indeces) == len(train_with_again), f"{len(train_with_indeces)} != {len(train_with_again)}"
        assert len(self.hallucinations_examples) >= len(
            self.all_mlp_vector_with_hall), f"{len(self.hallucinations_examples)} != {len(self.all_mlp_vector_with_hall)}"
        self.test_with_examples = [self.hallucinations_examples[i] for i in val_with_indeces]
        self.test_without_examples = [self.non_hallucinations_examples[i] for i in val_with_indeces]
        # for HK- steer
        self.test_general_examples = None if self.general_examples is None else [self.general_examples[i] for i in
                                                                                 val_with_indeces]



        self.mlp_output_of_test_with_examples = [self.all_mlp_vector_with_hall[i] for i in val_with_indeces]
        self.mlp_output_of_test_without_examples = [self.all_mlp_vector_without_hall[i] for i in val_with_indeces]
        self.attention_output_of_test_with_examples = [self.all_attention_vector_with_all[i] for i in val_with_indeces]
        self.attention_output_of_test_without_examples = [self.all_attention_vector_without_hall[i] for i in
                                                          val_with_indeces]
        self.residual_output_of_test_with_examples = [self.all_residual_with[i] for i in val_with_indeces]
        self.residual_output_of_test_without_examples = [self.all_residual_without[i] for i in val_with_indeces]
        if on_test_set:
            print(f"on test set!!")
            if self.path_general_test_set is None:
                self.test_with_examples = [self.hallucinations_examples[i] for i in test_with_indeces]
                self.test_without_examples = [self.non_hallucinations_examples[i] for i in test_with_indeces]
                self.test_general_examples = None if self.general_examples is None else [self.general_examples[i] for i in
                                                                                         test_with_indeces]
            else:
                self.test_with_examples = None
                self.test_without_examples = self.load_dataset(self.path_general_test_set)
                self.test_without_examples = self.test_without_examples[:min(200,len(self.test_without_examples))]
                self.test_general_examples = None
        print(f"{len(self.test_with_examples)=} {len(self.test_without_examples)=} {len(self.test_general_examples)=}")




    def set_what_to_intervene_on(self, use_mlp, use_attention, use_heads, use_residual):
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.intervention.use_mlp = use_mlp
        self.intervention.use_attention = use_attention
        self.intervention.use_heads = use_heads
        self.intervention.use_residual = use_residual

    def load_dataset(self, data_path):
        """
        load the dataset
        :param data_path:
        :return: dataset
        """
        print(f"load dataset {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        print(f"dataset size is {len(data)}")
        # print data hash and its last modification time
        print(f"data name {data_path} and last modification time is {time.ctime(os.path.getmtime(data_path))} ")
        print(f"hash data {subprocess.check_output(['sha1sum', f'{data_path}'])}")
        return data

    def split_data_to_train_val_test(self, data_indexes):
        """
        split data indexes to train val test
        :param data_indexes:
        :return:
        """
        random.seed(42)
        if self.seed_train_val is not None:
            print(f"shuffle all the data with seed {self.seed_train_val}")
            random.seed(self.seed_train_val)
        random.shuffle(data_indexes)
        train_val = data_indexes[:int(len(data_indexes) * 0.8)]
        test = data_indexes[int(len(data_indexes) * 0.8):]
        train = train_val[:int(len(data_indexes) * 0.7)]
        val = train_val[int(len(data_indexes) * 0.7):]
        assert len(train) + len(val) + len(test) == len(data_indexes)
        print(f"{len(train)=} {len(val)=} {len(test)=}")
        return train, val, test

    def split_data_to_train_val_test_for_all_data_types(self, data_split):
        data_indeces = [i for i in range(len(data_split))]
        # data_indeces = [i for i in data_indeces if i not in self.bad_indecess]
        train_indexes, val_indexes, test_indexes = self.split_data_to_train_val_test(data_indeces)
        train = [data_split[i] for i in train_indexes]
        val = [data_split[i] for i in val_indexes]
        test = [data_split[i] for i in test_indexes]
        return train, val, test

    def linear_classifier(self, train_with, train_without, test_with=None, test_without=None):
        """
        train a linear classifier on the data
        :param train_with: data with hallucinations
        :param train_without: data without hallucinations
        :return: classifier for each layer
        """
        # concatenate the data
        train = train_with + train_without
        labels = [1] * len(train_with) + [0] * len(train_without)
        if test_with is not None and test_without is not None:

            test = test_with + test_without
            test_labels = [1] * len(test_with) + [0] * len(test_without)
            assert len(test) == len(
                test_labels), f"{len(test)} != {len(test_labels)} {np.shape(np.array(test))=} {np.shape(np.array(test_labels))=}"
            test, test_labels = shuffle(test, test_labels, random_state=0)
        # shuffle the data
        classifier_for_layers = []
        test_acc = []
        test_labels_predicted = []
        train, true_labels = shuffle(train, labels, random_state=0)
        for layer in range(len(train_with[0])):
            train_vectors_curr_layer = np.array([i[layer] for i in train])
            random_state = 0
            if self.seed_train_val is not None:
                random_state = self.seed_train_val
            clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
            # use neighbors classifier
            clf.fit(train_vectors_curr_layer, true_labels)
            classifier_for_layers.append(clf)
            if test_with is not None and test_without is not None:
                test_vectors_curr_layer = np.array([i[layer] for i in test])
                test_acc.append(clf.score(test_vectors_curr_layer, test_labels))
                test_labels_predicted.append(clf.predict(test_vectors_curr_layer))
        return classifier_for_layers, test_acc, test_labels_predicted, true_labels

    def linear_classifier_for_heads(self, train_with, train_without, test_with=None, test_without=None):
        """
        train a linear classifier on the data and return the classifier and the test accuracy-for heads
        :param train_with:
        :param train_without:
        :param test_with:
        :param test_without:
        :return:
        """
        # concatenate the data
        train = train_with + train_without
        labels = [1] * len(train_with) + [0] * len(train_without)
        if test_with is not None and test_without is not None:

            test = test_with + test_without
            test_labels = [1] * len(test_with) + [0] * len(test_without)
            test, test_labels = shuffle(test, test_labels, random_state=0)
        # shuffle the data
        classifier_for_layers = []
        test_acc = []
        train, true_labels = shuffle(train, labels, random_state=0)
        for layer in range(len(train_with[0])):
            layer_classifier = []
            teas_acc_layer = []
            for head in range(len(train_with[0][layer])):
                train_vectors_curr_layer = np.array([i[layer][head] for i in train])
                random_state = 0
                if self.seed_train_val is not None:
                    random_state = self.seed_train_val
                clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
                clf.fit(train_vectors_curr_layer, true_labels)
                layer_classifier.append(clf)
                if test_with is not None and test_without is not None:
                    test_vectors_curr_layer = np.array([i[layer][head] for i in test])
                    teas_acc_layer.append(clf.score(test_vectors_curr_layer, test_labels))
            classifier_for_layers.append(layer_classifier)
            if test_with is not None and test_without is not None:
                test_acc.append(teas_acc_layer)
        return classifier_for_layers, test_acc

    def normClassification(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,4096) - 100 examples,32 layers,4096 features. we want to get the norm of each example and layer (100,32,1)
        train_with_norm_of_each_example_and_layer = np.linalg.norm(train_with, axis=2)
        train_with_norm_of_each_example_and_layer = train_with_norm_of_each_example_and_layer[:, :, np.newaxis]
        train_without_norm_of_each_example_and_layer = np.linalg.norm(train_without, axis=2)
        train_without_norm_of_each_example_and_layer = train_without_norm_of_each_example_and_layer[:, :, np.newaxis]
        test_with_norm_of_each_example_and_layer = np.linalg.norm(test_with, axis=2)
        test_with_norm_of_each_example_and_layer = test_with_norm_of_each_example_and_layer[:, :, np.newaxis]
        test_without_norm_of_each_example_and_layer = np.linalg.norm(test_without, axis=2)
        test_without_norm_of_each_example_and_layer = test_without_norm_of_each_example_and_layer[:, :, np.newaxis]

        classifier_for_layers, test_acc, test_labels_predicted, true_labels = self.linear_classifier(
            list(train_with_norm_of_each_example_and_layer), list(train_without_norm_of_each_example_and_layer),
            list(test_with_norm_of_each_example_and_layer),
            list(test_without_norm_of_each_example_and_layer))
        return test_acc

    def cosineSimilarityClassification(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,4096)
        train_with_normalized = train_with / np.linalg.norm(train_with, axis=2)[:, :, np.newaxis]
        # the norm of each example is 1, check that the norm of each example is 1
        for i in train_with_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        assert np.shape(train_with_normalized) == np.shape(train_with), f"{np.shape(train_with_normalized)=} {np.shape(train_with)=}"
        train_without_normalized = train_without / np.linalg.norm(train_without, axis=2)[:, :, np.newaxis]
        for i in train_without_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        test_with_normalized = test_with / np.linalg.norm(test_with, axis=2)[:, :, np.newaxis]
        for i in test_with_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        test_without_normalized = test_without / np.linalg.norm(test_without, axis=2)[:, :, np.newaxis]
        for i in test_without_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        classifier_for_layers, test_acc, test_labels_predicted, true_labels = self.linear_classifier(
            list(train_with_normalized), list(train_without_normalized), list(test_with_normalized),
            list(test_without_normalized))
        # assert that the norm of each example is 1
        assert np.linalg.norm(train_with_normalized, axis=2).all() == np.linalg.norm(train_with,
                                                                                     axis=2).all() == np.linalg.norm(
            train_without_normalized, axis=2).all() == np.linalg.norm(train_without,
                                                                      axis=2).all(), f"{np.linalg.norm(train_with_normalized, axis=2)=} {np.linalg.norm(train_with, axis=2)=} {np.linalg.norm(train_without_normalized, axis=2)=} {np.linalg.norm(train_without, axis=2)=}"
        return classifier_for_layers, test_acc, test_labels_predicted, true_labels

    def cosineClassifierForHeads(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,32,128)
        train_with_normalized = train_with / np.linalg.norm(train_with, axis=3)[:, :, :, np.newaxis]
        for i in train_with_normalized:
            for j in i:
                for k in j:
                    assert 1.001>np.linalg.norm(k) > 0.95, f"{np.linalg.norm(k)=}"
        train_without_normalized = train_without / np.linalg.norm(train_without, axis=3)[:, :, :, np.newaxis]
        test_with_normalized = test_with / np.linalg.norm(test_with, axis=3)[:, :, :, np.newaxis]
        test_without_normalized = test_without / np.linalg.norm(test_without, axis=3)[:, :, :, np.newaxis]
        assert np.linalg.norm(train_with_normalized, axis=3).all() == np.linalg.norm(train_with,
                                                                                     axis=3).all() == np.linalg.norm(
            train_without_normalized, axis=3).all() == np.linalg.norm(train_without,
                                                                      axis=3).all(), f"{np.linalg.norm(train_with_normalized, axis=3)=} {np.linalg.norm(train_with, axis=3)=} {np.linalg.norm(train_without_normalized, axis=3)=} {np.linalg.norm(train_without, axis=3)=}"
        classifier_for_layers, test_acc = self.linear_classifier_for_heads(list(train_with_normalized),
                                                                           list(train_without_normalized),
                                                                           list(test_with_normalized),
                                                                           list(test_without_normalized))
        return classifier_for_layers, test_acc

    def create_classifier_for_each_type_of_info(self):
        """
        create a classifier for each type of information - mlp,attention,residual,heads
        :return:
        """
        train_mlp_with, val_mlp_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_mlp_vector_with_hall)
        train_mlp_with_again, _, _ = self.split_data_to_train_val_test_for_all_data_types(self.all_mlp_vector_with_hall)
        assert len(train_mlp_with) == len(train_mlp_with_again), f"{len(train_mlp_with)} != {len(train_mlp_with_again)}"
        for i in range(len(train_mlp_with_again)):
            assert torch.eq(torch.tensor(train_mlp_with[i]),torch.tensor(train_mlp_with_again[
                i])).all(), f"{train_mlp_with[i]} != {train_mlp_with_again[i]}"
        train_mlp_without, val_mlp_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_mlp_vector_without_hall)
        classifier_for_layers_mlp, acc_mlp, labels_predicted_mlp, true_labels_mlp = self.cosineSimilarityClassification(
            train_mlp_with, train_mlp_without, val_mlp_with,
            val_mlp_without)
        train_attention_with, val_attention_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_attention_vector_with_all)
        train_attention_without, val_attention_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_attention_vector_without_hall)
        classifier_for_layers_attention, acc_attn, labels_predicted_attn, true_lables_attn = self.cosineSimilarityClassification(
            train_attention_with,
            train_attention_without, val_attention_with,
            val_attention_without)
        train_residual_with, val_residual_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_residual_with)
        train_residual_without, val_residual_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_residual_without)
        classifier_for_layers_residual, acc_residual, labels_predicted_residual, true_labels_residual = self.cosineSimilarityClassification(
            train_residual_with,
            train_residual_without, val_residual_with,
            val_residual_without)

        train_heads_with, val_heads_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.heads_vectors_with)
        train_heads_without, val_heads_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.heads_vectors_without)
        classifier_for_layers_heads, acc_heads = self.cosineClassifierForHeads(train_heads_with,
                                                                               train_heads_without, val_heads_with,
                                                                               val_heads_without)
        # the similarity of the labels predicted

        classifier_dict = {"mlp": classifier_for_layers_mlp, "attention": classifier_for_layers_attention,
                           "residual": classifier_for_layers_residual, "heads": classifier_for_layers_heads}
        acc_classifier_dict = {"mlp": acc_mlp, "attention": acc_attn, "residual": acc_residual, "heads": acc_heads}

        return classifier_dict, acc_classifier_dict








    def get_intervention(self, label_with_data, label_without_data):
        """
        get the intervention for each layer - the difference between the mean of the data with hallucinations and the mean of the data without hallucinations
        :param label_with_data:
        :param label_without_data:
        :return: a dict with the intervention for each layer
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            assert layer < 50

            layer_intervention_val[layer] = np.mean(label_with_data, axis=0)[layer] - \
                                            np.mean(label_without_data, axis=0)[layer]
            assert np.shape(layer_intervention_val[layer]) == np.shape(label_with_data[0][layer])
        return layer_intervention_val

    def get_intervention_based_classifier(self, label_with_data, label_without_data, type="mlp"):
        """
        use self.classifiers_dict to get the classifier plane and take the perpendicular vector to the direction of without data to be the intervention
        :param label_with_data:
        :param label_without_data:
        :param classifier_for_layers:
        :return:
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            classifier = self.classifiers_dict[type][layer]
            # the hyperplane of the linearsvc classifier
            w = classifier.coef_[0]
            # the perpendicular vector to the hyperplane that is in the direction of the without data
            d = w / np.linalg.norm(w)
            # check that the direction of the without data is in the direction of the without data
            mass_mean_vector = np.mean(label_with_data, axis=0)[layer] - np.mean(label_without_data, axis=0)[layer]

            if np.dot(d, mass_mean_vector) < 0:
                d = -d
            # normalize the vector to be in the same norm as the mass_mean_vector
            d = d * np.linalg.norm(mass_mean_vector)
            assert abs(np.linalg.norm(d) - np.linalg.norm(mass_mean_vector))<0.001, f"{np.linalg.norm(d)} != {np.linalg.norm(mass_mean_vector)}"
            layer_intervention_val[layer] = d
        return layer_intervention_val

    def get_intervention_based_classifier_for_heads(self, label_with_data, label_without_data, type="heads"):
        """
        use self.classifiers_dict to get the classifier plane and take the perpendicular vector to the direction of without data to be the intervention
        :param label_with_data:
        :param label_without_data:
        :param classifier_for_layers:
        :param type:
        :return:
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            layer_intervention_val[layer] = {}
            for head in range(len(label_with_data[0][layer])):
                assert layer < 50 and head < 40
                assert type in self.classifiers_dict.keys()
                assert type == "heads"
                classifier = self.classifiers_dict[type][layer][head]
                w = classifier.coef_[0]
                d = w / np.linalg.norm(w)
                mass_mean_vector = np.mean(label_with_data, axis=0)[layer][head] - \
                                   np.mean(label_without_data, axis=0)[layer][head]

                if np.dot(d, mass_mean_vector) < 0:
                    d = -d
                d = d * np.linalg.norm(mass_mean_vector)
                assert abs(np.linalg.norm(d) - np.linalg.norm(mass_mean_vector))<0.001, f"{np.linalg.norm(d)} != {np.linalg.norm(mass_mean_vector)}"
                layer_intervention_val[layer][head] = d
        return layer_intervention_val

    def get_intervention_for_heads(self, label_with_data, label_without_data):
        """
        get the intervention for each layer and head - the difference between the mean of the data with hallucinations and the mean of the data without hallucinations
        :param label_with_data:
        :param label_without_data:
        :return: a dict with the intervention for each layer and head
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            layer_intervention_val[layer] = {}
            for head in range(len(label_with_data[0][layer])):
                assert layer < 50 and head < 40

                layer_intervention_val[layer][head] = np.mean(label_with_data, axis=0)[layer][head] - \
                                                      np.mean(label_without_data, axis=0)[layer][head]

                assert np.shape(layer_intervention_val[layer][head]) == np.shape(label_with_data[0][layer][head])
        return layer_intervention_val

    def get_intervention_for_all_data_types_based_classifier(self, all_mlp_vector_with_hall,
                                                             all_mlp_vector_without_hall,
                                                             all_attention_vector_with_all,
                                                             all_attention_vector_without_hall,
                                                             all_residual_with, all_residual_without,
                                                             heads_vectors_with=None,
                                                             heads_vectors_without=None):
        train_mlp_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        train_mlp_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_without_hall)
        assert len(train_mlp_with) == len(train_mlp_without), f"{len(train_mlp_with)} != {len(train_mlp_without)}"
        intervention_mlp = self.get_intervention_based_classifier(train_mlp_with, train_mlp_without, type="mlp")
        train_attention_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_with_all)
        train_attention_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_without_hall)
        assert len(train_attention_with) == len(
            train_attention_without), f"{len(train_attention_with)} != {len(train_attention_without)}"
        intervention_attention = self.get_intervention_based_classifier(train_attention_with, train_attention_without,
                                                                        type="attention")
        train_residual_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_with)
        train_residual_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_without)
        assert len(train_residual_with) == len(
            train_residual_without), f"{len(train_residual_with)} != {len(train_residual_without)}"
        intervention_residual = self.get_intervention_based_classifier(train_residual_with, train_residual_without,
                                                                       type="residual")
        train_heads_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_with)
        train_heads_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_without)
        assert len(train_heads_with) == len(
            train_heads_without), f"{len(train_heads_with)} != {len(train_heads_without)}"
        intervention_heads = self.get_intervention_based_classifier_for_heads(train_heads_with, train_heads_without,
                                                                              type="heads")
        intervention_dict = {"mlp": intervention_mlp, "attention": intervention_attention,
                             "residual": intervention_residual, "heads": intervention_heads}
        return intervention_dict

    def get_intervention_for_all_data_types(self, all_mlp_vector_with_hall, all_mlp_vector_without_hall,
                                            all_attention_vector_with_all, all_attention_vector_without_hall,
                                            all_residual_with, all_residual_without, heads_vectors_with=None,
                                            heads_vectors_without=None):
        """
        get the intervention for each type of data - mlp,attention,residual,heads, using the train data
        :param all_mlp_vector_with_hall:
        :param all_mlp_vector_without_hall:
        :param all_attention_vector_with_all:
        :param all_attention_vector_without_hall:
        :param all_residual_with:
        :param all_residual_without:
        :param heads_vectors_with:
        :param heads_vectors_without:
        :return:
        """
        train_mlp_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        train_mlp_with2, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        for i in range(len(train_mlp_with)):
            assert torch.eq(torch.tensor(train_mlp_with[i]), torch.tensor(train_mlp_with2[i])).all(), f"{train_mlp_with[i]} != {train_mlp_with2[i]}"
        train_mlp_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_without_hall)
        assert len(train_mlp_with) == len(train_mlp_without), f"{len(train_mlp_with)} != {len(train_mlp_without)}"
        intervention_mlp = self.get_intervention(train_mlp_with, train_mlp_without)
        train_attention_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_with_all)
        train_attention_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_without_hall)
        assert len(train_attention_with) == len(
            train_attention_without), f"{len(train_attention_with)} != {len(train_attention_without)}"
        intervention_attention = self.get_intervention(train_attention_with, train_attention_without)
        train_residual_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_with)
        train_residual_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_without)
        assert len(train_residual_with) == len(
            train_residual_without), f"{len(train_residual_with)} != {len(train_residual_without)}"
        intervention_residual = self.get_intervention(train_residual_with, train_residual_without)
        train_heads_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_with)
        train_heads_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_without)
        assert len(train_heads_with) == len(
            train_heads_without), f"{len(train_heads_with)} != {len(train_heads_without)}"
        intervention_heads = self.get_intervention_for_heads(train_heads_with, train_heads_without)
        intervention_dict = {"mlp": intervention_mlp, "attention": intervention_attention,
                             "residual": intervention_residual, "heads": intervention_heads}
        return intervention_dict


class ModelIntervention():
    def __init__(self, path_to_results, model_name,
                 dataset_size, dataset_name, threshold_of_data, use_mlp, use_attention, use_heads, use_residual,
                 intervention_all_dict, classifier_dict,
                 alpha=5, normalize=False, acc_classification_val=None, static_intervention=False, check_mlp_out=None,
                 check_attention_out=None, check_residual_out=None):
        self.check_mlp_out = check_mlp_out
        self.check_attention_out = check_attention_out
        self.check_residual_out = check_residual_out
        self.static_intervention = static_intervention
        self.static_random_intervention = False
        self.last_hook = ""
        self.alpha = alpha
        self.normalize = normalize
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.path_to_save_results = path_to_results + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold_of_data}/concat_answer{False}_size{dataset_size}/"
        self.all_interventions = intervention_all_dict
        self.classifier_dict = classifier_dict
        self.acc_classification_val = acc_classification_val
        self.best_layers_by_val_acc = {"mlp": np.argsort(self.acc_classification_val["mlp"])[-5:],
                                       "attention": np.argsort(
                                           self.acc_classification_val["attention"])[-5:], "residual": np.argsort(
                self.acc_classification_val["residual"])[-5:]}
        # sort by acc heads- list of tuples (layer,head) sorted by acc
        self.best_layers_by_val_acc["heads"] = sorted(
            [(layer, head) for layer in range(len(self.acc_classification_val["heads"])) for head in
             range(len(self.acc_classification_val["heads"][layer]))],
            key=lambda x: self.acc_classification_val["heads"][x[0]][x[1]])[-48:]
        np.random.seed(0)

        self.min_acc = 0.65
        self.number_of_interventions = 0

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        transformers.GenerationConfig.do_sample = False
        # transformers.GenerationConfig.top_p = 0
        transformers.GenerationConfig.temperature = None
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model_creator = get_model_config()
        self.model_condig = model_creator.model_config(model_name, self.model)
        self.MODEL_NAME = self.model_condig["model_type"]

        self.call_hook_times = 0

    def rgetattr(self, obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

        # a safe way to set attribute of an object

    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)

    def wrap_model(self, model,
                   layers_to_check: List[str],
                   max_len=1000):

        hs_collector = {}
        handles = []
        options = {"." + self.model_condig["mlp_name"]: self.use_mlp,
                   "." + self.model_condig["attention_name"]: self.use_attention,
                   "." + self.model_condig["attn_proj_name"]: self.use_heads,
                   "": self.use_residual}
        final_layers_to_check = [layer_type for layer_type in layers_to_check if options[layer_type]]
        layers_to_check = final_layers_to_check
        for layer_idx in range(self.model_condig["num_hidden_layers"]):
            for layer_type in layers_to_check:
                layer_with_idx = f'{layer_idx}{layer_type}'
                inside_name = f"{self.model_condig['start_layer_prefex']}{layer_with_idx}"
                layer_pointer = self.rgetattr(model, inside_name)
                add_handler = True
                if add_handler:
                    handel = layer_pointer.register_forward_hook(self.changeActivationOutput(
                        layer_i=layer_idx,
                        layer_type=layer_type
                    ))
                    handles.append(handel)

        return hs_collector, handles

    def attention_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the attention layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = (new_output[0].clone().detach().to(device),) + output[1:]
        new_output[0][0][-1] = output[0][0][-1] - alpha * torch.tensor(
            self.all_interventions['attention'][layer_i]).squeeze(0).to(device)
        if not self.use_attention:
            new_output = output
        return new_output

    def mlp_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the mlp layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = output.clone().detach().to(device)
        new_output[0][-1] = output[0][-1] - alpha * torch.tensor(self.all_interventions['mlp'][layer_i]).squeeze(
            0).to(device)
        assert new_output.shape == output.shape, f"{new_output.shape=} {output.shape=}"
        if not self.use_mlp:
            new_output = output
        return new_output

    def residual_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the residual layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = (new_output[0].clone().detach().to(device),) + output[1:]
        new_output[0][0][-1] = output[0][0][-1] - alpha * torch.tensor(
            self.all_interventions['residual'][layer_i]).squeeze(0).to(device)
        assert new_output[0].shape == output[0].shape, f"{new_output[0].shape=} {output[0].shape=}"
        if not self.use_residual:
            new_output = output
        return new_output

    def heads_intervention(self, layer_i, layer_type, output, alpha, input_, module):
        """
        change the output of the heads layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        dim_head = self.model_condig["hidden_size"] // self.model_condig["num_attention_heads"]
        new_input = input_
        old_input = input_[0].clone().detach().to(device)
        heads_interventions = 0
        for head in range(self.model_condig["num_attention_heads"]):
            clf = self.classifier_dict['heads'][layer_i][head]
            vector_head = old_input[0][-1][dim_head * head:dim_head * (head + 1)].detach().cpu().numpy()
            predict_val_norm_vector = clf.predict([vector_head / np.linalg.norm(vector_head)])
            assert 1.001>np.linalg.norm(vector_head/np.linalg.norm(vector_head)) > 0.95, f"{np.linalg.norm(vector_head/np.linalg.norm(vector_head))=}"
            non_static_intervention = self.static_intervention == False and \
                                      self.acc_classification_val['heads'][layer_i][
                                          head] > self.min_acc and predict_val_norm_vector == 1
            static_intervention = self.static_intervention == True and (layer_i, head) in self.best_layers_by_val_acc[
                "heads"]
            if non_static_intervention or static_intervention:
                self.number_of_interventions += 1
                heads_interventions += 1
                val = self.all_interventions['heads'][layer_i][head]
                new_input[0][0][-1][dim_head * head:dim_head * (head + 1)] -= alpha * torch.tensor(
                    val).squeeze(
                    0).to(device)
        if heads_interventions > 0:
            # assert that not all values in new_input are the same as the old input
            assert not torch.eq(new_input[0][0][-1], old_input[0][-1]).all(), f"{new_input[0][0][-1]=} {old_input[0][-1]=}"
            new_output = module.forward(*new_input)
        else:
            assert torch.eq(new_input[0][0][-1], old_input[0][-1]).all(), f"{new_input[0][0][-1]=} {old_input[0][-1]=}"
            new_output = output
        assert torch.all(
            torch.eq(output[0], module.forward(*(old_input,))[0])), f"{output=} {module.forward(*input_)=}"
        assert np.shape(new_output) == np.shape(output), f"{np.shape(new_output)=} {np.shape(output)=}"
        return new_output

    def changeActivationOutput(self, layer_i, layer_type):
        def hook(module, input_, output):
            self.call_hook_times += 1
            alpha = self.alpha
            new_output = output
            next_hook = str(layer_i) + "_" + layer_type
            assert self.last_hook != next_hook, f"{self.last_hook=} {next_hook=}"
            self.last_hook = next_hook
            # attention intervention
            if layer_type == "." + self.model_condig["attention_name"] and self.use_attention is True:
                clf = self.classifier_dict['attention'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["attention"]
                vector_attention = output[0][0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_attention / np.linalg.norm(vector_attention)])
                assert 1.001>np.linalg.norm(vector_attention/np.linalg.norm(vector_attention)) > 0.95, f"{np.linalg.norm(vector_attention/np.linalg.norm(vector_attention))=}"
                non_static = self.static_intervention == False and self.acc_classification_val['attention'][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    self.number_of_interventions += 1
                    new_output = self.attention_intervention(layer_i, layer_type, output, alpha)

            # mlp intervention
            elif layer_type == "." + self.model_condig["mlp_name"] and self.use_mlp is True:
                clf = self.classifier_dict['mlp'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["mlp"]

                vector_mlp = output[0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_mlp / np.linalg.norm(vector_mlp)])
                assert 1.001>np.linalg.norm(vector_mlp/np.linalg.norm(vector_mlp)) > 0.95, f"{np.linalg.norm(vector_mlp/np.linalg.norm(vector_mlp))=}"
                non_static = self.static_intervention == False and self.acc_classification_val["mlp"][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    self.number_of_interventions += 1
                    new_output = self.mlp_intervention(layer_i, layer_type, output, alpha)

            elif layer_type == "" and self.use_residual is True:
                clf = self.classifier_dict['residual'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["residual"]

                vector_residual = output[0][0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_residual / np.linalg.norm(vector_residual)])
                assert 1.001>np.linalg.norm(vector_residual/np.linalg.norm(vector_residual)) > 0.95, f"{np.linalg.norm(vector_residual/np.linalg.norm(vector_residual))=}"
                non_static = self.static_intervention == False and self.acc_classification_val['residual'][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    self.number_of_interventions += 1
                    new_output = self.residual_intervention(layer_i, layer_type, output, alpha)

            elif layer_type == "." + self.model_condig["attn_proj_name"] and self.use_heads is True:
                new_output = self.heads_intervention(layer_i=layer_i, layer_type=layer_type, output=output, alpha=alpha,
                                                     input_=input_, module=module)

            assert new_output[0].shape == output[0].shape, f"{new_output[0].shape=} {output[0].shape=}"
            return new_output

        return hook

    def model_wrap_remove_hooks(self, model, handels_to_remove: List[torch.utils.hooks.RemovableHandle]):
        """
        remove hooks from model
        :param model:
        :param handels_to_remove:
        :return:
        """
        for handel in handels_to_remove:
            handel.remove()

    def run_model_with_hook(self, model, input_encoded):
        """
        run model with hook that will change values after some MLP and attention layers
        :param model:
        :param input_encoded:
        :return: model's output
        """
        model.requires_grad_(False)
        self.call_hook_times = 0
        # collect the hidden states before and after each of those layers (modules)
        hs_collector, handles = self.wrap_model(model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                        "." + self.model_condig["attention_name"],
                                                                        "." + self.model_condig["attn_proj_name"], ""])
        with torch.no_grad():
            output = model(input_encoded, output_hidden_states=True, output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
        gc.collect()

        self.model_wrap_remove_hooks(model, handles)

        return output

    def run_model_generate_with_hook(self, model, input_encoded, length=generation_length,
                                     min_length=generation_length):

        start_time = time.time()
        model.requires_grad_(False)
        hs_collector, handles = self.wrap_model(model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                        "." + self.model_condig["attention_name"],
                                                                        "." + self.model_condig["attn_proj_name"], ""])
        with torch.no_grad():
            # if type(input_encoded) == torch.Tensor:
            self.call_hook_times = 0
            output = model.generate(input_encoded, output_hidden_states=True, output_attentions=True, use_cache=False,
                                    max_length=(len(input_encoded[0]) + length), do_sample=False, temperature=None,top_p=None,
                                    num_beams=1, attention_mask=torch.ones(input_encoded.shape).to(device),
                                    pad_token_id=self.tok.eos_token_id, min_new_tokens=min_length)
            # assert self.call_hook_times % 32 == 0, f"{self.call_hook_times=}"
        assert len(output[0]) <= len(input_encoded[0]) + length
        assert len(output[0]) >= len(input_encoded[0]) + min_length, f"{len(output[0])=} {len(input_encoded[0])=}"
        generated = self.tok.batch_decode(output, skip_special_tokens=True)
        # remove hooks
        self.model_wrap_remove_hooks(model, handles)
        # print the time it took to generate
        return generated

    def get_answer_rank(self, model, prompt, answer_tokens):
        """
        get the mean rank if the two answers are the same else get the difference between the the first different token
        between the two answers
        :param model:
        :param prompt:
        :param answer_tokens:
        :return: if less than zero prefer the new answer and if more than zero prefer the old answer
        """
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        model_out = self.run_model_with_hook(model, input_ids)
        self.run_mlp_check = False
        self.run_residual_check = False
        self.run_attention_check = False
        logits = model_out.logits
        probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        # probability of the first token of the old and new answer
        prob_old = [probabilities[0][answer_tokens[0][0]].item()]
        prob_new = [probabilities[0][answer_tokens[1][0]].item()]
        rank_old = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[0]])]
        rank_new = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[0]])]
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        for i, token in enumerate(answer_tokens[0][:-1]):
            # add the token to the input_ids
            next_token = answer_tokens[0][i + 1]
            input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
            model_out = self.run_model_with_hook(model, input_ids)
            logits = model_out.logits
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            prob_old.append(probabilities[0][next_token].item())
            rank_old.append(sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[-1]]))

        assert len(prob_old) == len(answer_tokens[0]), f"{len(prob_old)=} {len(answer_tokens[0])=}"
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        if answer_tokens[0] == answer_tokens[1]:
            prob_new = prob_old
        else:
            for i, token in enumerate(answer_tokens[1][:-1]):
                next_token = answer_tokens[1][i + 1]
                input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
                model_out = self.run_model_with_hook(model, input_ids)
                logits = model_out.logits
                probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                prob_new.append(probabilities[0][next_token].item())
                rank_new.append(
                    sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[-1]]))

        assert len(prob_new) == len(answer_tokens[1]), f"{len(prob_new)=} {len(answer_tokens[1])=}"
        # geometric mean
        prob_old = np.prod(np.array(prob_old)) ** (1 / len(prob_old))
        prob_new = np.prod(np.array(prob_new)) ** (1 / len(prob_new))
        shorter_answer = min(len(answer_tokens[0]), len(answer_tokens[1]))
        non_similar_index = [i for i in range(shorter_answer) if answer_tokens[0][i] != answer_tokens[1][i]]
        if len(non_similar_index) > 0:
            assert answer_tokens[0][non_similar_index[0]] != answer_tokens[1][non_similar_index[
                0]], f"{answer_tokens[0][non_similar_index[0]]=} {answer_tokens[1][non_similar_index[0]]=}"
        if len(non_similar_index) == 0:
            non_similar_index = [0]
        rank = rank_new[non_similar_index[0]] - rank_old[non_similar_index[0]]
        if answer_tokens[0] == answer_tokens[1]: #no context setting
            return prob_old, sum(rank_old) / len(rank_old)
        del input_ids
        del model_out
        torch.cuda.empty_cache()
        return prob_old - prob_new, rank_old[0]



    def run_dataset_with_hook(self, dataset, tag, check_mlp_out=None, check_attention_out=None, check_residual_out=None,
                              no_context_dataset=False, calculate_wiki_pp=False):
        """
        run the dataset with the hook, it will check the dataset results with modification to the model inner state (the hook)
        It will print the std and mean of the probability of parametric answer - the probability of the contextual answer
        :param dataset_path:
        :param tag:
        :return:
        """

        self.check_mlp_out = check_mlp_out
        self.check_attention_out = check_attention_out
        self.check_residual_out = check_residual_out
        self.number_of_interventions = 0
        self.wanted_intervention = 0
        self.wanted_mlp_intervention = 0
        self.wanted_attention_intervention = 0
        self.wanted_residual_intervention = 0
        rank_bigger_that_zero = 0
        no_context_rank_high = 0
        non_text_generated = 0
        generated_both_answers = 0
        prompt_index = 0
        parametric_answer = 1
        contextual_answer = 2
        paraphraze_prompt_index = 5
        logits_on_true_answer_without_context_index = -1
        number_of_examples_used = 0
        preferable_answer_prob = []
        preferable_answer_rank = []
        generated_preferable_answer = []
        generated_text = []
        for index, point in enumerate(dataset):
            self.index = index
            self.run_mlp_check = True
            self.run_residual_check = True
            self.run_attention_check = True
            if index % 10 == 0:
                assert number_of_examples_used == index

            number_of_examples_used += 1

            prompt = point[prompt_index]
            old_target = point[1]
            new_target = point[2]
            answer_tokens = (point[3], point[4])

            praphraze_prompt = point[paraphraze_prompt_index]
            start_time = time.time()
            prob, rank = self.get_answer_rank(self.model, praphraze_prompt, answer_tokens)

            preferable_answer_prob.append(prob)
            preferable_answer_rank.append(rank)
            generated = self.run_model_generate_with_hook(self.model, self.tok([praphraze_prompt], return_tensors="pt")[
                "input_ids"].to(device))

            generated_text.append((generated[0], generated[0][len(praphraze_prompt):], old_target, new_target))
            generated = generated[0][len(praphraze_prompt):]

            if len(generated) == 0:
                non_text_generated += 1
            wanted_answer_generated = 0
            if (
                    old_target.strip() in generated or old_target.strip().lower() in generated.lower()):
                wanted_answer_generated += 1
            generated_preferable_answer.append(wanted_answer_generated)

        if calculate_wiki_pp:
            pp_wikipedia, generated_wiki = self.get_wikipedia_pp_score()
        else:
            pp_wikipedia = None
            generated_wiki = None

        perplexity = None
        return preferable_answer_rank, generated_preferable_answer, perplexity, generated_text, pp_wikipedia, generated_wiki




    def get_wikipedia_pp_score(self):
        """
        get the perplexity of wikipedia text using the model with the hook
        :return:
        """
        start_time = time.time()
        np.random.seed(42)

        def calc_loss(labels, logits):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss
        if not os.path.exists("wiki_dataset"):
            # save the wikipedia dataset
            wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", ignore_verifications=True)
            wiki.save_to_disk("wiki_dataset")
        data_wiki = load_from_disk("wiki_dataset")
        indexes_of_title = [i for i in range(len(data_wiki)) if
                            "=" in data_wiki[i]["text"] and data_wiki[i][
                                "text"].count("=") == 2]
        wanted_indexes = []
        for i in indexes_of_title:
            article = ""
            last_paragraph = ""
            article += data_wiki[i]["text"]
            j = 1
            count = 0
            while i + j < len(data_wiki) and data_wiki[i + j]["text"].count("=") != 2:
                if count < 2 and data_wiki[i + j]["text"] != "":
                    article += data_wiki[i + j]["text"]
                    count += 1
                j += 1
                if count == 2:
                    wanted_indexes.append((article, last_paragraph))
                    break

        sample = [wanted_indexes[i] for i in np.random.choice(len(wanted_indexes), 100, replace=False)]
        del data_wiki
        sample = sample[:100]
        assert len(sample) == 100, f"{len(sample)=}"
        wikipedia_pp_score = []
        direct_pp_score_no_intervention = []
        texts = []
        list_loss = []

        # collect the hidden states before and after each of those layers (modules)
        hs_collector, handles = self.wrap_model(self.model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                             "." + self.model_condig["attention_name"],
                                                                             "." + self.model_condig["attn_proj_name"],
                                                                             ""])

        for index, s in enumerate(sample):
            text = s[0]
            # last_paragraph = s[1]
            texts.append((text))
            input_ids_all = self.tok(text)["input_ids"][:100]
            assert len(input_ids_all) == 100, f"{len(input_ids_all)=}"
            # last_paragraph_ids = self.tok(last_paragraph)["input_ids"]
            # add the first token to the input_ids
            input_ids = torch.tensor([[input_ids_all[0]]]).to(device)
            input_ids_all = input_ids_all[1:]
            logits_all_tokens = []
            for i, token in enumerate(input_ids_all):
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True, use_cache=False)
                    torch.cuda.empty_cache()
                    gc.collect()
                    logits = outputs.logits
                    logits_all_tokens.append(logits[0, -1, :].clone().detach().cpu())
                    # update the input_ids
                    if i < len(input_ids_all) - 1:
                        input_ids = torch.cat((input_ids, torch.tensor([[token]]).clone().detach().to(device)), dim=-1)

                    del outputs

                torch.cuda.empty_cache()
                gc.collect()


            # concat the last logit from each logits in the list
            logits_all_tokens = torch.stack(logits_all_tokens)
            assert logits_all_tokens.shape[0] == len(input_ids_all), f"{logits_all_tokens.shape=} {len(input_ids_all)=}"
            assert input_ids[0][1:].all() == torch.tensor(
                input_ids_all).all(), f"{input_ids[0][1:]=} {torch.tensor(input_ids_all)=}"
            # calculate the loss
            loss = calc_loss(input_ids, logits_all_tokens)
            list_loss.append(loss.item())
            pp = torch.exp(loss)
            if pp.item() == float("inf"):  # for not want to use inf
                pp = torch.tensor(0)
            wikipedia_pp_score.append(pp.item())
            if not self.use_mlp and not self.use_attention and not self.use_residual and not self.use_heads:
                with torch.no_grad():
                    output = self.model(input_ids, labels=input_ids.clone())
                    loss = output.loss
                direct_pp_score_no_intervention.append(torch.exp(loss).item())
            else:
                direct_pp_score_no_intervention.append(0)
            del input_ids
            torch.cuda.empty_cache()
            gc.collect()

        # remove hooks
        self.model_wrap_remove_hooks(self.model, handles)

        assert len(sample) == len(wikipedia_pp_score) == len(direct_pp_score_no_intervention) == len(texts)
        return wikipedia_pp_score, texts




def run_intervention_results(intervene):

    preferable_answer_rank_correct, generated_preferable_answer_correct, perplexity_correct, generated_text_correct, pp_wikipedia_correct, generated_wiki_correct = intervene.intervention.run_dataset_with_hook(
        intervene.test_without_examples, "without", check_mlp_out=intervene.mlp_output_of_test_without_examples,
        check_attention_out=intervene.attention_output_of_test_without_examples,
        check_residual_out=intervene.residual_output_of_test_without_examples, no_context_dataset=True,
        calculate_wiki_pp=False)

    preferable_answer_rank_hall, generated_preferable_answer_hall, perplexity_hall, generated_text_hall, pp_wikipedia_hall, generated_wiki_hall = intervene.intervention.run_dataset_with_hook(
        intervene.test_with_examples, "with", check_mlp_out=intervene.mlp_output_of_test_with_examples,
        check_attention_out=intervene.attention_output_of_test_with_examples,
        check_residual_out=intervene.residual_output_of_test_with_examples, no_context_dataset=True,
        calculate_wiki_pp=False)
    preferable_answer_rank_hall_minus, generated_preferable_answer_hall_minus, perplexity_hall_minus, generated_text_hall_minus, pp_wikipedia_hall_minus, generated_wiki_hall_minus = intervene.intervention.run_dataset_with_hook(
        intervene.test_general_examples, "general", no_context_dataset=True, calculate_wiki_pp=False)


    return generated_preferable_answer_correct, generated_preferable_answer_hall, generated_preferable_answer_hall_minus

if __name__ == "__main__":
    import argparse
    initial_dataset_path = "datasets/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    # threshold
    # dataset name
    parser.add_argument("--dataset_name", type=str, default="triviaqa")

    parser.add_argument("--alpha", type=float, default=5.0,)

    parser.add_argument("--setting", type=str, default="")

    parser.add_argument("--concat", type=bool, default=True)

    parser.add_argument("--hk_", action="store_true",help="use the hk- model")

    dataset_name_test = parser.parse_args().dataset_name
    model_name = parser.parse_args().model_name
    model_name = model_name.replace("/", "_")
    setting = parser.parse_args().setting
    print(f"running with {model_name=}, {dataset_name_test=}, {setting=}, {parser.parse_args().alpha=} {parser.parse_args().concat=} {parser.parse_args().hk_=}", flush=True)
    # model_name = "google_gemma-2-9b"
    # model_name = "meta-llama_Llama-3.1-8B"
    # model_name = "mistralai_Mistral-7B-v0.3"
    results=[]
    for seed in [100,200,300]:
        if not parser.parse_args().hk_:
            intervene = InterventionByDetection("results/",
                                                f"{initial_dataset_path}{parser.parse_args().setting}NonHallucinate{dataset_name_test[0].upper() + dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                f"{initial_dataset_path}{parser.parse_args().setting}Hallucinate{dataset_name_test[0].upper() + dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                general_data_path=f"{initial_dataset_path}{parser.parse_args().setting}General{dataset_name_test[0].upper() + dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                model_name=model_name.replace("_", "/"), dataset_size=1000,
                                                dataset_name=dataset_name_test,
                                                threshold_of_data=1.0, use_mlp=False, use_attention=False,
                                                use_heads=False, use_residual=False, alpha=parser.parse_args().alpha,
                                                concatenate_answer=parser.parse_args().concat,
                                                static_intervention=True, on_test_set=True, seed_train_val=seed,
                                                use_classifier_for_intervention=False,
                                                alice=True if "Alice" in setting else False,
                                                persona=True if "Persona" in setting else False,
                                                truthfulness=True if "Truthful" in setting else False,
                                                fake_alignment=False,
                                                realistic=True if "Realistic" in setting else False)
        else:
            intervene = InterventionByDetection("results/",
                                                f"{initial_dataset_path}{parser.parse_args().setting}NonHallucinate{dataset_name_test[0].upper()+ dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                f"{initial_dataset_path}{parser.parse_args().setting}General{dataset_name_test[0].upper()+ dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                general_data_path=f"{initial_dataset_path}{parser.parse_args().setting}Hallucinate{dataset_name_test[0].upper()+ dataset_name_test[1:]}WithThreshold1.0_{model_name}.json",
                                                model_name=model_name.replace("_","/"), dataset_size=1000, dataset_name=dataset_name_test,
                                                threshold_of_data=1.0, use_mlp=False, use_attention=False,
                                                use_heads=False, use_residual=False, alpha=parser.parse_args().alpha,concatenate_answer=parser.parse_args().concat,
                                                static_intervention=True, on_test_set=True, seed_train_val=seed,
                                                use_classifier_for_intervention=False,alice=True if "Alice" in setting else False, persona=True if "Persona" in setting else False, truthfulness=True if "Truthful" in setting else False, fake_alignment = False, realistic=True if "Realistic" in setting else False, hk_=parser.parse_args().hk_)

        intervene.set_what_to_intervene_on(use_attention=False, use_mlp=False, use_heads=True, use_residual=False)
        generated_preferable_answer_correct, generated_preferable_answer_hall, generated_preferable_answer_hall_minus = run_intervention_results(intervene)

        if not parser.parse_args().hk_:
            results.append({
                "seed": seed,
                "factually_correct": sum(generated_preferable_answer_correct) / len(generated_preferable_answer_correct),
                "hallucinations_k+": sum(generated_preferable_answer_hall) / len(generated_preferable_answer_hall),
                "hallucinations_k-": sum(generated_preferable_answer_hall_minus) / len(
                    generated_preferable_answer_hall_minus)
            })
        else:
            results.append({
                "seed": seed,
                "factually_correct": sum(generated_preferable_answer_correct) / len(generated_preferable_answer_correct),
                "hallucinations_k+": sum(generated_preferable_answer_hall_minus) / len(generated_preferable_answer_hall_minus),
                "hallucinations_k-": sum(generated_preferable_answer_hall) / len(
                    generated_preferable_answer_hall)
            })
    print(f"results {results}")
    print(f"final results:\n Factual: {round(np.mean([r['factually_correct'] for r in results])*100,2)}+-{round(np.std([r['factually_correct'] for r in results])*100,2)}"
            f"\nHK+{round(np.mean([r['hallucinations_k+'] for r in results])*100,2)}+-{round(np.std([r['hallucinations_k+'] for r in results])*100,2)}, "
            f"\nHK-{round(np.mean([r['hallucinations_k-'] for r in results])*100,2)}+-{round(np.std([r['hallucinations_k-'] for r in results])*100,2)}")
