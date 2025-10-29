import gc
import os
import json
import pickle
import random
import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import model_inside
from sklearn.svm import LinearSVC
# mlp classifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# load f1_score
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(data_path):
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
    return data

def split_data_to_train_val_test(data_indexes,seed=None,static_dataset=None):
    """
    split data indexes to train val test
    :param data_indexes:
    :return:
    """
    random.seed(42)
    if seed is not None:
        random.seed(seed)
    random.shuffle(data_indexes)
    train = data_indexes[:int(0.7 * len(data_indexes))]
    val = data_indexes[int(0.7 * len(data_indexes)):]
    test = []
    assert len(train) + len(val) + len(test) == len(data_indexes)
    # print(f"{len(train)=} {len(val)=} {len(test)=}")
    return train, val, test

def split_data_to_train_val_test_for_all_data_types(data_split,seed=None,static_dataset=None):
    data_indeces = [i for i in range(len(data_split))]
    train_indexes, val_indexes, test_indexes = split_data_to_train_val_test(data_indeces,seed)
    train = [data_split[i] for i in train_indexes]
    val = [data_split[i] for i in val_indexes]
    test = [data_split[i] for i in test_indexes]
    return train, val, test



def linear_classifier(train_with, train_without, test_with=None, test_without=None,seed_train_val=None):
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
        if seed_train_val is not None:
            random_state = seed_train_val
        clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
        clf.fit(train_vectors_curr_layer, true_labels)
        classifier_for_layers.append(clf)
        if test_with is not None and test_without is not None:
            test_vectors_curr_layer = np.array([i[layer] for i in test])
            test_acc.append(clf.score(test_vectors_curr_layer, test_labels))
            test_labels_predicted.append(clf.predict(test_vectors_curr_layer))
    return classifier_for_layers, test_acc, test_labels_predicted, true_labels





def classifier_dict_create(train_mlp_with, train_mlp_without, train_attention_with, train_attention_without,
                            train_residual_with, train_residual_without,component_val, s=None, train_heads_with=None, train_heads_without=None):
    classifiers_dict = {}
    if train_mlp_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_mlp_with, train_mlp_without,
                                                                                      seed_train_val=s)
        classifiers_dict["mlp"] = classifiers
    if train_attention_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_attention_with,
                                                                                      train_attention_without,
                                                                                      seed_train_val=s)
        classifiers_dict["attention"] = classifiers
    if train_residual_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_residual_with,
                                                                                      train_residual_without,
                                                                                      seed_train_val=s)
        classifiers_dict["residual"] = classifiers

    one_dict = {key: [] for key in classifiers_dict.keys()}
    f1_dict = {key: [] for key in classifiers_dict.keys()}
    for key in classifiers_dict.keys():
        test_with = component_val[key][0]
        test_without = component_val[key][1]
        test = test_with + test_without
        test_labels = [1] * len(test_with) + [0] * len(test_without)
        test, test_labels = shuffle(test, test_labels, random_state=0)
        for layer in range(len(classifiers_dict[key])):
            classifier = classifiers_dict[key][layer]
            test_vectors_curr_layer = np.array([i[layer] for i in test])
            test_predictions = classifier.predict(test_vectors_curr_layer)
            acc = accuracy_score(test_labels, test_predictions)
            one_dict[key].append(acc)
            f1 = f1_score(test_labels, test_predictions, average='weighted')
            f1_dict[key].append(f1)


    return one_dict, f1_dict




def plot_classification_graphs_on_generalization(threshold, model_name, dataset_size=500, dataset_name="triviaqa", alpha=5,
                                      concat_answer=False, seed=None, static_dataset=False, concat_answer_test=True, static_dataset_test=False,setting1 = "Alice", setting2 = "Persona"):
    final_classification_dict_static = []
    final_classification_dict = []
    seeds = [None,100,200]
    model_name = model_name.replace("Meta-Llama", "Llama")
    data_with_setting1 = load_dataset(f"datasets/{setting1}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    data_without_setting1 = load_dataset(f"datasets/{setting1}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    data_with_setting2 = load_dataset(f"datasets/{setting2}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    data_without_setting2 = load_dataset(f"datasets/{setting2}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    for s in seeds:
        MLPCheck = ModelInside.ModelInside("results/",
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=1.0, concat_answer=concat_answer_test,static_dataset=static_dataset_test,alice=True if setting1 == "Alice" else False, persona=True if setting1 == "Persona" else False, truthfulness=True if setting1 == "Truthful" else False, realistic=True if setting1 == "Realistic" else False)
        all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.load_all_data()
        example_indeces = [i for i in range(len(all_mlp_vector_with_hall))]
        train, val, test = split_data_to_train_val_test(example_indeces, s)
        train_residual_with , val_residual_with,_ = split_data_to_train_val_test_for_all_data_types(all_residual_with/ np.linalg.norm(all_residual_with, axis=2)[:, :, np.newaxis],s)
        train_residual_without , val_residual_without,_ = [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in train], [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in val], [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in test]
        component_val = { "residual": [ val_residual_with, val_residual_without]}
        acc,f1 = classifier_dict_create(None,None,None,None,  train_residual_with,  train_residual_without, component_val, s)
        final_classification_dict.append(acc)
        del MLPCheck
        gc.collect()
        torch.cuda.empty_cache()
        MLPCheck = ModelInside.ModelInside("results/",
                                               None,
                                               None,
                                               model_name=model_name, dataset_size=dataset_size,
                                               dataset_name=dataset_name,
                                               threshold_of_data=1.0, concat_answer=concat_answer,
                                               static_dataset=static_dataset, alice=True if setting2 == "Alice" else False, persona=True if setting2 == "Persona" else False, truthfulness=True if setting2 == "Truthful" else False, realistic=True if setting2 == "Realistic" else False)
        all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.load_all_data()

        train_static_indexes = random.sample([i for i in range(len(all_mlp_vector_with_hall))],
                                             min(len(train_residual_with), len(all_mlp_vector_with_hall)))
        data_with_val_prompt = [data_with_setting1[i][0] for i in val]
        data_without_val_prompt = [data_without_setting1[i][0] for i in val]
        non_use_static_indeces = [i for i in train_static_indexes if
                                  data_with_setting2[i][0] in data_with_val_prompt or data_with_setting2[i][
                                      0] in data_without_val_prompt or data_without_setting2[i][
                                      0] in data_with_val_prompt or data_without_setting2[i][
                                      0] in data_without_val_prompt]
        train_static_indexes = [i for i in train_static_indexes if i not in non_use_static_indeces]
        for i in train_static_indexes:
            assert data_without_setting2[i][0] not in data_with_val_prompt and data_without_setting2[i][
                0] not in data_without_val_prompt and data_with_setting2[i][0] not in data_with_val_prompt and \
                   data_with_setting2[i][0] not in data_without_val_prompt
        for i in range(len(non_use_static_indeces)):
            sampling_index = [i for i in range(len(all_mlp_vector_with_hall)) if
                              (i not in train_static_indexes and i not in non_use_static_indeces)]
            if len(sampling_index) == 0:
                # print(f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                break
            new_index = random.sample(sampling_index, 1)[0]
            while data_with_setting2[new_index][0] in data_with_val_prompt or data_with_setting2[new_index][
                0] in data_without_val_prompt or data_without_setting2[new_index][0] in data_with_val_prompt or \
                    data_without_setting2[new_index][0] in data_without_val_prompt:
                non_use_static_indeces.append(new_index)
                sampling_index = [i for i in range(len(all_mlp_vector_with_hall)) if
                                  (i not in train_static_indexes and i not in non_use_static_indeces)]
                if len(sampling_index) == 0:
                    # print(f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                    new_index = None
                    break
                new_index = random.sample(sampling_index, 1)[0]
                if len(non_use_static_indeces) + len(train_static_indexes) == len(all_mlp_vector_with_hall):
                    # print(
                    #     f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                    new_index = None
                    break
            if new_index is not None:
                train_static_indexes.append(new_index)
            if len(non_use_static_indeces) + len(train_static_indexes) == len(all_mlp_vector_with_hall):
                # print(
                #     f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                break

        train_residual_with_2 = [all_residual_with[i] / np.linalg.norm(all_residual_with[i], axis=1)[:, np.newaxis] for i
                               in train_static_indexes if i < len(all_mlp_vector_with_hall)]
        train_residual_without_2 = [
            all_residual_without[i] / np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in
            train_static_indexes if i < len(all_mlp_vector_with_hall)]



        one_dict,f1_Score = classifier_dict_create(None,None,None,None,
                                            train_residual_with_2, train_residual_without_2, component_val, s)
        final_classification_dict_static.append(one_dict)
    print(f"{final_classification_dict=}")
    final_dict = {"residual": []}
    final_dict_std = {"residual": []}
    final_dict_static = {"residual_static": []}
    final_dict_std_static = {"residual_static": []}

    for key in final_classification_dict[0].keys():

        for i in range(len(final_classification_dict[0][key])):
            final_dict[key].append(np.mean([c[key][i] * 100 for c in final_classification_dict]))
            final_dict_std[key].append(np.std([c[key][i] * 100 for c in final_classification_dict]))
            final_dict_static[key + "_static"].append(np.mean([c[key][i] * 100 for c in final_classification_dict_static]))
            final_dict_std_static[key + "_static"].append(np.std([c[key][i] * 100 for c in final_classification_dict_static]))
    print(f"{final_dict=} {final_dict_static=}")

    return final_dict, final_dict_std, final_dict_static, final_dict_std_static




def plot_bar_chart_generalization_results(results_dict, dataset_name,model_name, generate_legand = False):
    # plot bar graph such that for each key two bars will be next to each other the [0]+-[1] is the first bar and [2]+-[3] is the second bar
    import matplotlib.pyplot as plt
    import numpy as np
    keys = list(results_dict.keys())
    print(f"{keys=}")
    scores1 = []
    stds1 = []
    scores2 = []
    stds2 = []

    for key in keys:
        if len(results_dict[key]) != 4:
            raise ValueError(f"Each key must have exactly 4 values (score1, std1, score2, std2). "
                             f"Key '{key}' has {len(results_dict[key])} values.")

        score1, std1, score2, std2 = results_dict[key]
        scores1.append(score1)
        stds1.append(std1)
        scores2.append(score2)
        stds2.append(std2)

    # Set up the plot
    fig, ax = plt.subplots()

    # Set positions for bars
    x = np.arange(len(keys))
    width = 0.35  # Width of bars

    # Default colors if none provided
    colors = ['skyblue', 'lightcoral']

    # Create bars
    bars1 = ax.bar(x - width / 2, scores1, width, yerr=stds1,
                    color=colors[0],
                   capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width / 2, scores2, width, yerr=stds2,
                    color=colors[1],
                   capsize=5, alpha=0.8)

    # Customize the plot

    ax.set_ylabel('Accuracy',fontsize=30)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 101, 20))
    # set the y-axis size to 15
    ax.set_yticklabels(np.arange(0, 101, 20), fontsize=25)
    ax.set_xticklabels(keys, rotation=45, ha='right' if len(max(keys, key=len)) > 8 else 'center', fontsize=30)
    # legend that the blue is base and the red is generalization on the lower left corner
    if generate_legand:
        ax.legend([bars1[0], bars2[0]], ['Base', 'Generalization'], loc='lower left', fontsize=30)
    ax.grid(True, alpha=0.3, axis='y')



    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(f"results/{dataset_name}_{model_name.replace('/','_')}_generalization_results.pdf", dpi=300, format='pdf')


def run_generalization():
    dataset_name = ["triviaqa", "naturalqa"]
    models = ["meta-llama/Meta-Llama-3.1-8B", "google/gemma-2-9b", "mistralai/Mistral-7B-v0.3", ]
    settings = ["Realistic", "Alice", "Persona", "Truthful", ]


    for model in models:
        for name in dataset_name:
            for setting1 in settings:
                results = {}
                for setting2 in settings:
                    if setting1 == setting2:
                        continue
                    print(f"{model=} {name=} {setting1=} {setting2=}", flush=True)
                    final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_generalization(
                        threshold=1.0, model_name=model, dataset_size=1000, dataset_name=name, alpha=5,
                        concat_answer=True, seed=None, static_dataset=False,
                        concat_answer_test=True, static_dataset_test=False,
                        setting1=setting1, setting2=setting2, )
                    print(f"{final_dict=} {final_dict_std=} {final_dict_static=} {final_dict_std_static=}")
                    results[setting2] = [final_dict["residual"][15], final_dict_std["residual"][15],
                                         final_dict_static["residual_static"][15],
                                         final_dict_std_static["residual_static"][15]]
                plot_bar_chart_generalization_results(results, name, model + setting1,
                                                      generate_legand=True if setting1 == "Realistic" else False)

