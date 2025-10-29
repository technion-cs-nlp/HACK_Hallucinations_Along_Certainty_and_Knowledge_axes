import gc
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import psutil
import torch
from huggingface_hub import login
import pandas as pd
import seaborn as sns
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LogisticRegression
import nltk
from scipy.stats import spearmanr
from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    EntailmentGPT4,
    EntailmentGPT35,
    EntailmentGPT4Turbo,
    EntailmentLlama,
    HuggingfaceModel,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
    cluster_assignment_entropy,
)

import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class case_study:

    def __init__(self, root_path):
        self.root_path = root_path
        self.stats_files = []

    def find_hallucinations_stats_files(self, root_path, alternative_name='hallucinations_stats_half.json',
                                        factuality=False):
        # List to store all found file paths
        stats_files = []

        # Walk through all directories and files in the given root path
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                # Check if the filename matches 'hallucinations_stats.json'
                part_of_name = alternative_name
                if factuality:
                    part_of_name = 'factuality_stats.json'
                if filename == part_of_name:
                    file_path = os.path.join(dirpath, filename)

                    stats_files.append(file_path)

        return stats_files

    def open_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        # if the data is a list of dictionaries, return it else return None
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            random.seed(42)
            random.shuffle(data)
            return data
        else:
            return None

    def remove_half_hallucinations(self, root_path):
        """
        Remove the hallucinations that might not be hallucinations. Save the new files with the name hallucinations_stats_half.json
        :param root_path:
        :return:
        """
        stats_files = self.find_hallucinations_stats_files(root_path, alternative_name='hallucinations_stats.json')
        import nltk
        # # nltk.download('stopwords')
        # nltk.download('wordnet')
        stop_words = list(set(stopwords.words('english'))) + ["the"]
        for file_path in stats_files:
            data = self.open_file(file_path)
            if data is None or len(data) == 0:
                continue

            new_file_path = file_path.replace('hallucinations_stats.json', 'hallucinations_stats_half.json')
            new_data = []
            for example in data:
                true_answer = example["true_answer"].lower().strip().replace("-", " ")
                generated = example["generated"].lower().strip()
                model_name = file_path.split("/")[1]
                if model_name == "google_gemma-2-9b-it" and generated.count("*") < 4:
                    continue
                if "the answer is not " in generated:
                    continue
                generated_answer = \
                    generated.split("\nanswer:")[2].strip().replace("assistant\n\n", "").replace("model\nthe answer is",
                                                                                                 "").replace(
                        "the answer is", "").split("\n")[0].split(".")[0].split(",")[0].strip().lower().replace("-",
                                                                                                                " ")
                generated_answer = generated_answer.split(". ")[0]
                # remove from both answers "the"
                true_answer = " ".join([word for word in true_answer.split() if word.lower() not in stop_words])
                generated_answer = " ".join(
                    [word for word in generated_answer.split() if word.lower() not in stop_words])

                # check that a synonym of the true answer is not in the generated answer
                synonims = nltk.corpus.wordnet.synsets(true_answer)
                is_syn = False
                for syn in synonims:
                    for l in syn.lemmas():
                        if l.name().replace("_", " ").lower() in generated_answer:
                            is_syn = True
                            break
                if is_syn:
                    continue
                # stem the words
                true_answer = " ".join([nltk.PorterStemmer().stem(word) for word in true_answer.split()])
                generated_answer = " ".join([nltk.PorterStemmer().stem(word) for word in generated_answer.split()])
                dist = nltk.edit_distance(true_answer, generated_answer)
                if len(generated_answer) == 0 or len(true_answer) == 0 or sum(
                        [1 for word in true_answer.split() if word in generated_answer.split()]) >= 0.5 * len(
                    true_answer.split()) \
                        or true_answer.split()[-1].lower() in generated_answer.lower().split():
                    continue

                if (dist > 2 or true_answer.isdigit()) and (
                        len(generated_answer) > 0 and "great" not in generated_answer and "none " not in generated_answer and "n/a" not in generated_answer \
                        and not (
                        generated_answer.split()[0] == true_answer.split()[0] and len(generated_answer.split()) == 1)):
                    new_data.append(example)

            with open(new_file_path, 'w') as file:
                json.dump(new_data, file)



    def clean_generations(self, examples):

        response_clean = []
        examples_with_clean = []
        for example in examples:

            generated = example["generated"]
            prompt = example["prompt"]
            only_generation = generated.replace(prompt, "")
            answer_only = \
            generated.split("\nanswer:")[2].strip().replace("assistant\n\n", "").replace("model\nThe answer is",
                                                                                         "").replace(
                "The answer is", "").split("\nquestion:")[0].split("\n")[0].split(".")[0].strip()
            answer_only = answer_only.split(". ")[0]
            if answer_only == "":
                continue

            generated = generated.replace(
                "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\n", "").replace(
                "user\n", "").replace("\nmodel", "").replace("assistant\n", "").replace("model\n", "")
            generated_with_answer = generated.split(answer_only)[0] + answer_only
            response_clean.append(generated_with_answer)
            examples_with_clean.append(example)
        return response_clean, examples_with_clean

    def get_hidden_states(self, model, tokenizer, prompt_with_response):

        hidden_states_examples = []
        for example in prompt_with_response:
            with torch.no_grad():
                # Concatenate prompt and response
                input_text = example
                inputs = tokenizer(input_text, return_tensors='pt').to(device)

                # Enable output of hidden states
                outputs = model.model(**inputs, output_hidden_states=True)
                # Extract hidden states from the specified layer
                hidden_state = outputs.hidden_states  # Shape: layers tuple (batch_size, seq_len, hidden_size)
                # move it to cpu and keep only last token hidden state for each layer
                hidden_state = [hidden_state[i][:, -1, :].cpu() for i in range(len(hidden_state))]
                hidden_states_examples.append(hidden_state)
        return hidden_states_examples

    def mitigation_via_prob(self, hall_train, test_hall, factual_train, test_factual, model_name, chock_examples,
                            all_chock_examples, svm=True, our_score=False, train_hall_step_2=None,
                            train_factual_step_2=None):
        """
        Mitigation via probabilities
        :param hall_train:
        :param test_hall:
        :param factual_train:
        :param test_factual:
        :return:
        """




        generated_with_response_hall_train, examples_train_hall = self.clean_generations(hall_train)
        generated_with_response_hall_test, examples_test_hall = self.clean_generations(test_hall)
        generated_with_response_factual_train, examples_train_factual = self.clean_generations(factual_train)
        generated_with_response_factual_test, examples_test_factual = self.clean_generations(test_factual)
        generated_with_response_chock_examples, chock_examples = self.clean_generations(chock_examples)
        generated_with_response_chock_all, all_chock_examples = self.clean_generations(all_chock_examples)

        hidden_states_hall_train = self.get_hidden_states(self.model, self.tokenizer,
                                                          generated_with_response_hall_train)

        hidden_states_hall_test = self.get_hidden_states(self.model, self.tokenizer,
                                                         generated_with_response_hall_test)
        hidden_states_factual_train = self.get_hidden_states(self.model, self.tokenizer,
                                                             generated_with_response_factual_train)
        hidden_states_factual_test = self.get_hidden_states(self.model, self.tokenizer,
                                                            generated_with_response_factual_test)
        hidden_states_chock_examples = self.get_hidden_states(self.model, self.tokenizer,
                                                              generated_with_response_chock_examples)
        hidden_states_chock_all = self.get_hidden_states(self.model, self.tokenizer,
                                                         generated_with_response_chock_all)
        num_layers = len(hidden_states_hall_train[0])
        if our_score:
            def add_entropy_to_hidden_states(hidden_states, examples):
                modified_states = []
                for i, example_states in enumerate(hidden_states):
                    # example_states is a list of tensors for each layer
                    modified_example = []
                    for l, layer_states in enumerate(example_states):

                        entropy_tensor = torch.tensor(
                            [[1 - (len(set(examples[i]["temp_generations"])) / len(examples[i]["temp_generations"]))]],
                            dtype=layer_states.dtype,
                            device=layer_states.device)
                        modified_layer = torch.cat((layer_states, entropy_tensor), 1)
                        modified_example.append(modified_layer)
                    modified_states.append(modified_example)
                return modified_states

            hidden_states_hall_train = add_entropy_to_hidden_states(hidden_states_hall_train, examples_train_hall)
            hidden_states_factual_train = add_entropy_to_hidden_states(hidden_states_factual_train,
                                                                       examples_train_factual)
            hidden_states_hall_test = add_entropy_to_hidden_states(hidden_states_hall_test, examples_test_hall)
            hidden_states_factual_test = add_entropy_to_hidden_states(hidden_states_factual_test, examples_test_factual)
            hidden_states_chock_examples = add_entropy_to_hidden_states(hidden_states_chock_examples, chock_examples)
            hidden_states_chock_all = add_entropy_to_hidden_states(hidden_states_chock_all, all_chock_examples)

        acc = []
        chock_score = []
        chock_all = []
        acc_hall = []
        factual_accuracy_ = []
        auroc_list = []
        for layer in [14]:
            # calculate the prob successing the hidden states
            def prepare_data(group1, group2, layer, train=False):
                if len(group2) == 0:
                    x = torch.vstack([g1[layer] for g1 in group1])
                elif len(group1) == 0:
                    x = torch.vstack([g2[layer] for g2 in group2])
                else:
                    x = torch.cat(
                        [torch.vstack([g1[layer] for g1 in group1]), torch.vstack([g2[layer] for g2 in group2])], dim=0)
                y = torch.cat([
                    torch.zeros(len(group1), dtype=torch.float32),
                    torch.ones(len(group2), dtype=torch.float32)
                ])
                # shuffle the data x and y together
                p = np.random.permutation(len(x))
                if not our_score or train:
                    x = x[p]
                    y = y[p]
                return x, y, p

            # Prepare training and test data
            X_train, y_train, p = prepare_data(hidden_states_hall_train, hidden_states_factual_train, layer, train=True)
            X_test, y_test, _ = prepare_data(hidden_states_hall_test, hidden_states_factual_test, layer)
            # subset of the data containing only hall examples
            X_test_hall, y_test_hall, _ = prepare_data(hidden_states_hall_test, [], layer)
            X_test_factual, y_test_factual, _ = prepare_data([], hidden_states_factual_test, layer)
            X_chock, y_chock, _ = prepare_data(hidden_states_chock_examples, [], layer)
            X_chock_all, y_chock_all, _ = prepare_data(hidden_states_chock_all, [], layer)
            if not svm:
                clf = LogisticRegression(random_state=42, tol=1e-5, max_iter=1000)
                clf.fit(X_train.numpy(), y_train.numpy())
                accuracy = clf.score(X_test.numpy(), y_test.numpy())
                factual_accuracy = clf.score(X_test_factual.numpy(), y_test_factual.numpy())
                hall_accuracy = clf.score(X_test_hall.numpy(), y_test_hall.numpy())
                accuracy_chock = clf.score(X_chock.numpy(), y_chock.numpy())
                accuracy_chock_all = clf.score(X_chock_all.numpy(), y_chock_all.numpy())
                auroc = self.auroc(clf.predict_proba(X_test_hall.numpy())[:, 1],
                                   clf.predict_proba(X_test_factual.numpy())[:, 1], "prob")
            else:

                clf = LinearSVC(random_state=42, tol=1e-5, dual=True, max_iter=1000)
                clf.fit(X_train.numpy(), y_train.numpy())
                accuracy = clf.score(X_test.numpy(), y_test.numpy())
                factual_accuracy = clf.score(X_test_factual.numpy(), y_test_factual.numpy())
                hall_accuracy = clf.score(X_test_hall.numpy(), y_test_hall.numpy())
                accuracy_chock = clf.score(X_chock.numpy(), y_chock.numpy())
                accuracy_chock_all = clf.score(X_chock_all.numpy(), y_chock_all.numpy())
                auroc = self.auroc(clf.decision_function(X_test_hall.numpy()),
                                   clf.decision_function(X_test_factual.numpy()), "prob")
                # preds_binary = clf.predict(X_test.numpy())
                # f1 = f1_score(y_test, preds_binary)
            acc.append(accuracy)
            chock_score.append(accuracy_chock)
            acc_hall.append(hall_accuracy)
            chock_all.append(accuracy_chock_all)
            factual_accuracy_.append(factual_accuracy)
        assert len(acc) == len(chock_score) == len(acc_hall) == len(chock_all) == len(
            factual_accuracy_), f"{len(acc)=} {len(chock_score)=} {len(acc_hall)=} {len(chock_all)=} {len(factual_accuracy_)=}"
        assert len(acc) == 1
        assert (round(factual_accuracy_[0] * len(test_factual)) + round(acc_hall[0] * len(test_hall))) / (
                    len(test_hall) + len(test_factual)) - acc[
                   0] <= 0.05, f"{factual_accuracy_[0]*len(test_factual)=} {acc_hall[0]*len(test_hall)=} {accuracy=} {(factual_accuracy_[0]*len(test_factual) + acc_hall[0]*len(test_hall))/ (len(test_hall)+ len(test_factual))=}"
        return acc[0], acc_hall[0], chock_all[0], chock_score[0], round(
            factual_accuracy_[0] * len(test_factual)), round(acc_hall[0] * len(test_hall)), round(
            chock_all[0] * len(all_chock_examples)), round(chock_score[0] * len(chock_examples)), auroc

    def strong_chock_examples(self, train_hall, train_factual, test_hall, test_factual):

        # strong chock examples are examples that pass ALL measures
        thresholds = {}
        for measure in ["prob", "prob_diff", "semantic_entropy"]:
            threshold, _, _, _, _ = self.get_threshold(train_hall, train_factual, measure)
            thresholds[measure] = threshold

        strong_chock_examples = []
        all_chock_examples = []
        for example in test_hall:
            if example["prob"] >= thresholds["prob"] or example["semantic_entropy"] <= thresholds["semantic_entropy"] or \
                    example["prob_diff"] >= thresholds["prob_diff"]:
                all_chock_examples.append(example)
        for example in test_hall:
            is_strong = True
            for measure, threshold in thresholds.items():
                if "entropy" in measure:
                    if example[measure] >= threshold:
                        is_strong = False
                        break
                else:
                    if example[measure] < threshold:
                        is_strong = False
                        break

            if is_strong:
                strong_chock_examples.append(example)
        for e in strong_chock_examples:
            assert e in all_chock_examples, f"{e} not in all chock examples"
        print(
            f"len strong CM examples: {len(strong_chock_examples)} {len(all_chock_examples)=} {len(test_hall)=} {len(test_factual)=}")
        return all_chock_examples, strong_chock_examples

    def mitigation_generation(self, prompt, mitigation_prompt, model_name, model, tokenizer, number_of_tokens=10,
                              abstain=True):
        """
        Mitigation via generation
        :param prompt:
        :param model_name:
        :return:
        """
        if "Instruct" in model_name or "-it" in model_name:
            # split the prompt before "The answer is" and after

            prompt_start = prompt.split("The answer is")[0].replace("assistantist", "").replace("model", "").replace(
                "assistant\n\n", "").replace("assistant", "")
            if abstain:
                messages = [
                    {"role": "user", "content": prompt_start},

                ]
                messages += [{"role": "assistant", "content": " The answer is " + prompt.split("The answer is")[1]}]
                messages += [{"role": "user", "content": mitigation_prompt}]
                messages += [{"role": "assistant", "content": " The answer is "}]
            else:
                messages = [
                    {"role": "user", "content": mitigation_prompt + prompt_start},
                ]


            unwanted_tokens_at_the_end = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n",
                                          "<end_of_turn>", "<start_of_turn>", "model", " ", "\n\n", "</s>"]
            unwanted_tokens_embedded = self.tokenizer(unwanted_tokens_at_the_end)["input_ids"]
            unwanted_tokens_embedded = [x for y in unwanted_tokens_embedded for x in y]
            unwanted_tokens_embedded = list(set(unwanted_tokens_embedded))
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            while input_ids[0][-1] in unwanted_tokens_embedded:
                input_ids = input_ids[:, :-1]

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            with torch.no_grad():
                response = self.model.generate(input_ids, max_length=(len(input_ids[0]) + number_of_tokens),
                                               do_sample=False,
                                               pad_token_id=self.tokenizer.eos_token_id, num_beams=1,
                                               eos_token_id=terminators, top_p=None, temperature=None,
                                               attention_mask=torch.ones_like(input_ids))
            generated = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]

        else:
            final_prompt = prompt + mitigation_prompt + " The answer is "
            if not abstain:
                final_prompt = mitigation_prompt + prompt
            input_ids = \
                self.tokenizer([final_prompt], padding=True, return_token_type_ids=False, return_tensors="pt")[
                    "input_ids"].to(device)
            with torch.no_grad():
                model_out = self.model.generate(input_ids, max_length=(len(input_ids[0]) + number_of_tokens),
                                                do_sample=False,
                                                pad_token_id=self.tokenizer.eos_token_id, num_beams=1, top_p=None,
                                                temperature=None, attention_mask=torch.ones_like(input_ids))
            generated = self.tokenizer.batch_decode(model_out, skip_special_tokens=True)[0]
        input_data = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated[len(input_data):]

    def abstain_prompt_mitigation(self, hall_train, test_hall, factual_train, test_factual, model_name, chock_examples,
                                  all_chock_examples):
        abstain_prompt = "The above answers is: A. True B. False."

        def count_abstain(examples, abstain_prompt):
            count = 0
            used_examples = []
            for example in examples:
                clean_generations, _ = self.clean_generations([example])
                if len(clean_generations) == 0 or clean_generations[0].endswith("question:") or "The answer is" not in \
                        clean_generations[0]:
                    continue
                used_examples.append(example)
                generated = self.mitigation_generation(clean_generations[0], abstain_prompt, model_name, self.model,
                                                       self.tokenizer, abstain=True)
                if generated == "":
                    continue
                if ("B" in generated or "False" in generated) and (not "True" in generated and not "A" in generated):
                    count += 1

            return count, used_examples

        abstain_hall, hall_used = count_abstain(test_hall, abstain_prompt)
        abstain_factual, factual_used = count_abstain(test_factual, abstain_prompt)

        assert abstain_hall <= len(hall_used), f"{abstain_hall=} {len(hall_used)=}"
        assert abstain_factual <= len(factual_used), f"{abstain_factual=} {len(factual_used)=}"
        accuracy = (abstain_hall + len(test_factual) - abstain_factual) / (len(test_hall) + len(test_factual))
        abstain_chock, chock_used = count_abstain(chock_examples, abstain_prompt)
        accuracy_chock = abstain_chock / len(chock_examples)
        all_chock, all_chock_used = count_abstain(all_chock_examples, abstain_prompt)
        accuracy_all_chock = all_chock / len(all_chock_examples)
        return accuracy, abstain_hall / len(test_hall), accuracy_all_chock, accuracy_chock, len(
            test_factual) - abstain_factual, abstain_hall, all_chock, abstain_chock

    def few_shots_mitigation(self, hall_train, test_hall, factual_train, test_factual, model_name, chock_examples,
                             all_chock_examples):
        null_mitigation = "Look at examples in the “Examples” section and utilize examples and information from that section to perform the following task."

        def count_true_answer(examples, abstain_prompt):
            count = 0
            for example in examples:
                prompt = example["prompt"]
                true_answer = example["true_answer"]
                generated = self.mitigation_generation(prompt, abstain_prompt, model_name, self.model,
                                                       self.tokenizer, abstain=False)
                if generated == "":
                    continue
                if true_answer.lower().strip() in generated.lower().strip() or generated.lower().strip() in true_answer.lower().strip():
                    count += 1
            return count

        abstain_hall = count_true_answer(test_hall, null_mitigation)
        abstain_factual = count_true_answer(test_factual, null_mitigation)
        accuracy = (abstain_hall + abstain_factual) / (len(test_hall) + len(test_factual))
        abstain_chock = count_true_answer(chock_examples, null_mitigation)
        accuracy_chock = abstain_chock / len(chock_examples)
        all_chock = count_true_answer(all_chock_examples, null_mitigation)
        accuracy_all_chock = all_chock / len(all_chock_examples)
        return accuracy, abstain_hall / len(
            test_hall), accuracy_all_chock, accuracy_chock, abstain_factual, abstain_hall, all_chock, abstain_chock

    def auroc(self, data_hall, data_fact, metric_name):

        from sklearn.metrics import roc_auc_score
        y = [0] * len(data_hall) + [1] * len(data_fact)
        if "entropy" in metric_name:
            X = [-e[metric_name] for e in data_hall] + [-e[metric_name] for e in data_fact]
        elif "temp_generations" in metric_name:
            X = [1 - (len(set(e[metric_name])) / len(e[metric_name])) for e in data_hall] + [
                1 - (len(set(e[metric_name])) / len(e[metric_name])) for e in data_fact]

        else:
            X = np.concatenate((data_hall, data_fact))

        return roc_auc_score(y, X)

    def mitigation_all_based(self, root_path):
        # temp_generations - sampling based mitigation
        # mean_entropy - entropy based mitigation
        # certainty_prob_entropy - our proposed mitigation via probe
        # linear_probe_sig - linear probe based mitigation
        # prompt_abstain - prompt abstention based mitigation
        # few_shots_prompt - prompt mitigate based mitigation
        stats_files = self.find_hallucinations_stats_files(root_path)

        mitigations = ["certainty_prob_entropy", "prompt_abstain", "few_shots_prompt", "linear_probe_sig",
                       "temp_generations", "mean_entropy", ]

        results = {}

        for file_path in stats_files:
            data_hall = self.open_file(file_path)
            data_fact = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))

            if data_hall is None or data_fact is None or "27" in file_path:
                continue


            random.seed(42)
            random.shuffle(data_hall)
            random.shuffle(data_fact)
            min_size = min(len(data_hall), len(data_fact))
            hall_train, test_hall = data_hall[:int(0.5 * min_size)], data_hall[int(0.5 * min_size):min_size]
            factual_train, test_factual = data_fact[:int(0.5 * min_size)], data_fact[int(0.5 * min_size):min_size]

            all_chock, strong_chock_examples = self.strong_chock_examples(train_hall=hall_train,
                                                                          train_factual=factual_train,
                                                                          test_hall=test_hall,
                                                                          test_factual=test_factual)
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            settings = file_path.split("/")[3]
            self.model = HuggingfaceModel(
                model_name.replace("_", "/"), stop_sequences="default", max_new_tokens=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name.replace("_", "/"))
            self.model.eval()
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token

            if settings not in results.keys():
                results[settings] = {}
                results[settings] = {}

            if dataset not in results[settings].keys():
                results[settings][dataset] = {}
                results[settings][dataset] = {}

            if model_name not in results[settings][dataset].keys():
                results[settings][dataset][model_name] = {}
                results[settings][dataset][model_name] = {}
            results[settings][dataset][model_name]["temp_generations"] = {}
            results[settings][dataset][model_name]["mean_entropy"] = {}
            # results[settings][dataset][model_name]["prob"] = {}
            results[settings][dataset][model_name]["linear_probe_sig"] = {}
            results[settings][dataset][model_name]["prompt_abstain"] = {}
            results[settings][dataset][model_name]["few_shots_prompt"] = {}
            results[settings][dataset][model_name]["non_mitigation"] = {"correct": len(test_factual),
                                                                        "hallucination": len(test_hall),
                                                                        "CA": len(all_chock),
                                                                        "CS": len(strong_chock_examples)}
            results[settings][dataset][model_name]["certainty_prob_entropy"] = {}

            print(f"File: {file_path}")
            for mitigation in mitigations:
                print(f"Mitigation: {mitigation}")
                if "certainty_prob_entropy" in mitigation:
                    all_chock_train, strong_chock_examples_train = self.strong_chock_examples(train_hall=hall_train,
                                                                                              train_factual=factual_train,
                                                                                              test_hall=hall_train,
                                                                                              test_factual=factual_train)

                    chock_presentage = len(all_chock_train) / len(hall_train)
                    hall_train_without_chock = [example for example in hall_train if example not in all_chock_train]
                    number_chock_to_add = int(0.65 * len(hall_train) - len(all_chock_train))
                    # replace examples from hall_train with all_chock till we reach 15%
                    # select random examples from all_chock
                    if number_chock_to_add > 0 and len(all_chock) > 50:
                        number_chock_to_add = min(number_chock_to_add, len(all_chock) - 50)
                        chock_additional = random.sample(all_chock,
                                                         number_chock_to_add)
                        hall_train_without_chock = random.sample(hall_train_without_chock,
                                                                 len(hall_train_without_chock) - number_chock_to_add)
                        assert len(hall_train_without_chock) + len(chock_additional) + len(all_chock_train) == len(
                            hall_train), f"{len(hall_train_without_chock)=} {len(chock_additional)=} {len(all_chock_train)=} {len(hall_train)=}"
                        hall_train_final = chock_additional + hall_train_without_chock + all_chock_train
                        new_chock = [example for example in all_chock if example not in chock_additional]
                        new_chock_strong = [example for example in strong_chock_examples if
                                            example not in chock_additional]
                        new_hall_test = [example for example in test_hall if example not in chock_additional]
                        new_test_factual = random.sample(test_factual, len(new_hall_test))
                        assert len(new_hall_test) == len(
                            new_test_factual), f"{len(new_hall_test)=} {len(new_test_factual)=}"
                    else:
                        hall_train_final = hall_train
                        new_chock = all_chock
                        new_chock_strong = strong_chock_examples
                        new_hall_test = test_hall
                        new_test_factual = test_factual
                    # assert abs(len(hall_train_final) -len(hall_train))<=1, f"{len(hall_train_final)=} {len(hall_train)=}"

                    for example in new_chock_strong:
                        assert example not in hall_train_final, f"{example} in hall train final {example in chock_additional} {example in hall_train_without_chock} {example in all_chock_train}"
                    for example in new_hall_test:
                        assert example not in hall_train_final, f"{example} in hall train final {example in chock_additional} {example in hall_train_without_chock} {example in all_chock_train}"
                    for example in new_chock:
                        assert example not in hall_train_final, f"{example} in hall train final {example in chock_additional} {example in hall_train_without_chock} {example in all_chock_train}"

                    acc, hall_acc, all_chock_score, chock_score, abstain_fact, abstain_hall, abstain_chock_all, abstain_chock_strong, auroc = self.mitigation_via_prob(
                        hall_train=hall_train_final, test_hall=new_hall_test, factual_train=factual_train,
                        test_factual=new_test_factual, model_name=model_name, chock_examples=new_chock_strong,
                        all_chock_examples=new_chock, svm=False, our_score=False)
                    # scores = self.our_score(hall_train=hall_train, test_hall=test_hall, factual_train=factual_train, test_factual=test_factual, model_name=model_name,chock_examples=strong_chock_examples, all_chock_examples=all_chock, certainty_mitigation_name="mean_entropy")
                    results[settings][dataset][model_name][mitigation] = {"accuracy": acc, 'hall_acc': hall_acc,
                                                                          "cm": chock_score,
                                                                          "cm-f": all_chock_score,
                                                                          "abstain_fact": abstain_fact,
                                                                          "abstain_hall": abstain_hall,
                                                                          "abstain_cm-f": abstain_chock_all,
                                                                          "abstain_cm": abstain_chock_strong,"auroc": auroc }

                    test_hall = new_hall_test
                    all_chock = new_chock
                    strong_chock_examples = new_chock_strong
                    test_factual = new_test_factual
                    results[settings][dataset][model_name]["non_mitigation"] = {"correct": len(test_factual),
                                                                                "hallucination": len(test_hall),
                                                                                "CA": len(all_chock),
                                                                                "CS": len(strong_chock_examples)}
                    continue

                if "prompt_abstain" in mitigation:
                    if "Instruct" in model_name or "-it" in model_name:
                        acc, hall_acc, all_chock_score, chock_score, abstain_fact, abstain_hall, abstain_chock_all, abstain_chock_strong = self.abstain_prompt_mitigation(
                            hall_train=hall_train, test_hall=test_hall, factual_train=factual_train,
                            test_factual=test_factual, model_name=model_name, chock_examples=strong_chock_examples,
                            all_chock_examples=all_chock)
                        results[settings][dataset][model_name][mitigation] = {"accuracy": acc, 'hall_acc': hall_acc,
                                                                              "cm": chock_score,
                                                                              "cm-f": all_chock_score,
                                                                              "abstain_fact": abstain_fact,
                                                                              "abstain_hall": abstain_hall,
                                                                              "abstain_cm-f": abstain_chock_all,
                                                                              "abstain_cm": abstain_chock_strong}
                    continue
                if "few_shots_prompt" in mitigation:
                    if "Instruct" in model_name or "-it" in model_name:
                        acc, hall_acc, all_chock_score, chock_score, abstain_fact, abstain_hall, abstain_chock_all, abstain_chock_strong = self.few_shots_mitigation(
                            hall_train=hall_train, test_hall=test_hall, factual_train=factual_train,
                            test_factual=test_factual, model_name=model_name, chock_examples=strong_chock_examples,
                            all_chock_examples=all_chock)
                        results[settings][dataset][model_name][mitigation] = {"accuracy": acc, 'hall_acc': hall_acc,
                                                                              "cm": chock_score,
                                                                              "cm-f": all_chock_score,
                                                                              "abstain_fact": abstain_fact,
                                                                              "abstain_hall": abstain_hall,
                                                                              "abstain_cm-f": abstain_chock_all,
                                                                              "abstain_cm": abstain_chock_strong}
                    continue
                if "linear_probe" in mitigation:
                    acc, hall_acc, all_chock_score, chock_score, abstain_fact, abstain_hall, abstain_chock_all, abstain_chock_strong, auroc = self.mitigation_via_prob(
                        hall_train=hall_train, test_hall=test_hall, factual_train=factual_train,
                        test_factual=test_factual, model_name=model_name, chock_examples=strong_chock_examples,
                        all_chock_examples=all_chock, svm=True if "svm" in mitigation else False, )
                    results[settings][dataset][model_name][mitigation] = {"accuracy": acc, 'hall_acc': hall_acc,
                                                                          "cm": chock_score,
                                                                          "cm-f": all_chock_score,
                                                                          "abstain_fact": abstain_fact,
                                                                          "abstain_hall": abstain_hall,
                                                                          "abstain_cm-f": abstain_chock_all,
                                                                          "abstain_cm": abstain_chock_strong,
                                                                          "auroc": auroc}
                    continue

                threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss = self.get_threshold(
                    hall_train,
                    factual_train,
                    mitigation)
                print(f"{mitigation=} {model_name=} {dataset=} {threshold=}")

                if "temp_generations" in mitigation:
                    number_of_not_mitigated = sum([1 for e in test_hall if
                                                   1 - (len(set(e[mitigation])) / len(e[mitigation])) >= threshold])
                    number_of_mitigated_non_hall = sum([1 for e in test_factual if
                                                        1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])
                    accuracy = (sum([1 for e in test_hall if
                                     1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold]) + sum(
                        [1 for e in test_factual if
                         1 - (len(set(e[mitigation])) / len(e[mitigation])) >= threshold])) / (
                                           len(test_hall) + len(test_factual))
                    accuracy_chock = (sum([1 for e in strong_chock_examples if
                                           1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])) / len(
                        strong_chock_examples)
                    hall_accuracy = (sum([1 for e in test_hall if
                                          1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])) / len(
                        test_hall)
                    accuracy_chock_all = (sum([1 for e in all_chock if
                                               1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])) / len(
                        all_chock)
                    hall_correct = sum([1 for e in test_hall if
                                        1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])
                    factual_correct = sum([1 for e in test_factual if
                                           1 - (len(set(e[mitigation])) / len(e[mitigation])) >= threshold])
                    chock_all_correct = sum([1 for e in all_chock if
                                             1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])
                    chock_strong_correct = sum([1 for e in strong_chock_examples if
                                                1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])
                    assert accuracy == (factual_correct + hall_correct) / (len(test_factual) + len(
                        test_hall)), f"{accuracy=} {factual_correct=} {hall_correct=} {len(test_factual)=} {len(test_hall)=}"
                    assert accuracy_chock == chock_strong_correct / len(
                        strong_chock_examples), f"{accuracy_chock=} {chock_strong_correct=} {len(strong_chock_examples)=}"
                    results[settings][dataset][model_name][mitigation] = {"accuracy": accuracy,
                                                                          'hall_acc': hall_accuracy,
                                                                          "cm": accuracy_chock,
                                                                          "cm-f": accuracy_chock_all,
                                                                          "abstain_fact": factual_correct,
                                                                          "abstain_hall": hall_correct,
                                                                          "abstain_cm-f": chock_all_correct,
                                                                          "abstain_cm": chock_strong_correct,
                                                                          "auroc": self.auroc(test_hall, test_factual,
                                                                                              mitigation)}


                elif mitigation == "mean_entropy":
                    number_of_not_mitigated = sum([1 for e in test_hall_values if e[mitigation] < threshold])
                    number_of_mitigated_non_hall = sum([1 for e in test_non_hall_values if e[mitigation] >= threshold])
                    accuracy = (sum([1 for e in test_hall if e[mitigation] >= threshold]) + sum(
                        [1 for e in test_factual if
                         e[mitigation] < threshold])) / (len(test_hall) + len(test_factual))
                    accuracy_chock = (sum([1 for e in strong_chock_examples if
                                           e[mitigation] >= threshold])) / len(strong_chock_examples)
                    hall_accuracy = (sum([1 for e in test_hall if
                                          e[mitigation] >= threshold])) / len(test_hall)
                    accuracy_chock_all = (sum([1 for e in all_chock if
                                               e[mitigation] >= threshold])) / len(all_chock)
                    hall_correct = sum([1 for e in test_hall if
                                        e[mitigation] >= threshold])
                    factual_correct = sum([1 for e in test_factual if
                                           e[mitigation] < threshold])
                    chock_all_correct = sum([1 for e in all_chock if
                                             e[mitigation] >= threshold])
                    chock_strong_correct = sum([1 for e in strong_chock_examples if
                                                e[mitigation] >= threshold])
                    assert hall_correct / len(
                        test_hall) == hall_accuracy, f"{hall_correct=}/{len(test_hall)=} {hall_accuracy=}"
                    assert (factual_correct + hall_correct) / (len(test_factual) + len(
                        test_hall)) == accuracy, f"{factual_correct=}/{len(test_factual)=} {hall_correct=}/{len(test_hall)=} {accuracy=}"
                    results[settings][dataset][model_name][mitigation] = {"accuracy": accuracy,
                                                                          'hall_acc': hall_accuracy,
                                                                          "cm": accuracy_chock,
                                                                          "cm-f": accuracy_chock_all,
                                                                          "abstain_fact": factual_correct,
                                                                          "abstain_hall": hall_correct,
                                                                          "abstain_cm-f": chock_all_correct,
                                                                          "abstain_cm": chock_strong_correct,
                                                                          "auroc": self.auroc(test_hall, test_factual,
                                                                                              mitigation)}

            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

        print(results) # results dictionary contain the final results for each setting, dataset, model and mitigation. The results are accuracy, hall_acc, cm, cm-f, abstain_fact, abstain_hall, abstain_cm-f, abstain_cm



    def plot_all_measures(self, root_path):
        """
        Plot all measures
        :param root_path:
        :return:
        """
        stats_files = self.find_hallucinations_stats_files(root_path)
        all_results = {}
        for file_path in stats_files:
            data_hall = self.open_file(file_path)
            print(f"File: {file_path}")
            data_fact = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))
            if data_hall is None or data_fact is None:
                continue
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            settings = file_path.split("/")[3]
            if dataset not in all_results.keys():
                all_results[dataset] = {}
            if model_name not in all_results[dataset].keys():
                all_results[dataset][model_name] = {}
            if settings not in all_results[dataset][model_name].keys():
                all_results[dataset][model_name][settings] = {}

            for measure in ["prob", "prob_diff", "semantic_entropy", "semantic_entropy_temp_0.5"]:
                threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss = self.get_threshold(
                    data_hall, data_fact, measure)
                assert test_hall_values[0] in data_hall, f"{test_hall_values[0]=} {data_hall[0]=}"
                assert test_non_hall_values[0] in data_fact, f"{test_non_hall_values[0]=} {data_fact[0]=}"
                print(f"{measure=} {threshold=}")
                prob_uncertain = [e[measure] for e in test_hall_values]
                prob_correct = [e[measure] for e in test_non_hall_values]
                self.plot_measure_hallucination_cumulative(prob_correct, prob_uncertain,
                                                           model_name + "_" + dataset + "_" + settings + "_" + measure,
                                                           "results/plots/", measure, threshold)
                y_values_hall = 100 * (
                            sum([1 for prob in prob_uncertain if prob >= threshold]) / max(1, len(prob_uncertain)))

                if "entropy" in measure:
                    y_values_hall = 100 * (
                                sum([1 for prob in prob_uncertain if prob <= threshold]) / max(1, len(prob_uncertain)))
                all_results[dataset][model_name][settings][measure] = y_values_hall
        self.generate_latex_tables(all_results)

    def generate_latex_tables(self, all_results):
        """
        Generate LaTeX tables for each dataset with improved formatting and statistics.
        Table format will have certainty methods as rows and models as columns.

        Parameters:
        all_results (dict): Dictionary with structure all_results[dataset][model_name][settings]
                            where settings values are metrics like "prob", "prob_diff", etc.

        Returns:
        dict: Dictionary of LaTeX tables, one for each dataset
        """
        import numpy as np

        # Define model name mappings (simplify names)
        model_name_map = {
            "meta-llama_Llama-3.1-8B": "Llama",
            "mistralai_Mistral-7B-v0.3": "Mistral",
            "google_gemma-2-9b": "Gemma",
            "meta-llama_Llama-3.1-8B-Instruct": "Llama-Inst",
            "mistralai_Mistral-7B-Instruct-v0.3": "Mistral-Inst",
            "google_gemma-2-9b-it": "Gemma-Inst"
        }

        # Method display names
        method_display_names = {
            "prob": " Probability",
            "prob_diff": " Probability Diff.",
            "semantic_entropy": " Semantic Entropy"
        }

        # List of methods in order
        methods = ["prob", "prob_diff", "semantic_entropy"]

        # List of settings in order
        settings_array = ["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "prompt_6", "prompt_7"]

        # Dictionary to store the tables
        latex_tables = {}

        # Generate a table for each dataset
        for dataset_name, dataset in all_results.items():
            # Get all models except gemma-27B
            models = [model for model in dataset.keys() if ("gemma-2-27b" not in model)]

            # Create the LaTeX table
            table = ""

            # Table header with improved formatting
            table += r"\begin{table*}[h]" + "\n"
            table += r"\centering" + "\n"
            table += r"%\small" + "\n"
            table += r"\setlength{\tabcolsep}{8pt} % Adjust column spacing" + "\n"
            table += r"\renewcommand{\arraystretch}{1.2} % Adjust row spacing" + "\n"

            # Create column specification based on number of models
            column_spec = "l|" + "c" * len(models)
            table += r"\begin{tabular}{" + column_spec + "}" + "\n"
            table += r"\toprule" + "\n"

            # Create header row with model names
            header = r"\textbf{Certainty Method}"
            for model in models:
                model_display = model_name_map.get(model, model)
                header += r" & \textbf{" + model_display + "}"
            header += r" \\"
            table += header

            table += r"\midrule"

            # For each method (prob, prob_diff, semantic_entropy)
            for method in methods:
                row = method_display_names.get(method, method)

                # For each model
                for model in models:
                    # Collect values for calculating mean and std across all settings
                    values = []

                    # Gather all values for this model and method across settings
                    for setting in settings_array:
                        if setting not in dataset[model]:
                            continue
                        value = dataset[model][setting][method]
                        values.append(value)

                    # Calculate mean and standard deviation
                    mean = np.mean(values)
                    std = np.std(values)

                    # Format as mean±std with 2 decimal places
                    formatted_value = f"${mean:.2f}_{{\scriptscriptstyle \\pm {std:.2f}}}$"
                    formatted_value = f"${mean:.1f} \pm {std:.1f}$"

                    # Add to the row
                    row += " & " + formatted_value

                # End the row
                row += r" \\"
                table += row

            # Table footer
            table += r"\bottomrule" + "\n"
            table += r"\end{tabular}" + "\n"
            table += r"\caption{Results for " + dataset_name + "}" + "\n"
            table += r"\label{tab:results_" + dataset_name.lower().replace(' ', '_') + "}" + "\n"
            table += r"\end{table*}" + "\n"

            # Add the table to the dictionary
            latex_tables[dataset_name] = table
        print(latex_tables)
        return latex_tables



    def plot_measure_hallucination_cumulative(self, prob_uncertain: list, prob_hall: list, title: str, path,
                                              measure: str, true_threshold: float = None):


        sns.set_theme(style="whitegrid", font_scale=2, rc={
            'font.size': 40,  # Set a large font size
            'axes.titlesize': 40,
            'axes.labelsize': 72,
            'xtick.labelsize': 60,
            'ytick.labelsize': 60,
            'legend.fontsize': 50,
            'figure.figsize': (14, 10),
        })

        bins = np.linspace(1, 0, 21)
        if "entropy" in measure:
            bins = np.linspace(0, 4, 9)
        epsilon = 1e-6
        # Calculate cumulative percentage of examples for each bin
        y_values = [(
                100 * (
                sum([1 for prob in prob_uncertain if prob >= threshold - epsilon]) / max(1, len(prob_uncertain))))
            for threshold in bins
        ]

        y_values_hall = [
            100 * (sum([1 for prob in prob_hall if prob >= threshold - epsilon]) / max(1, len(prob_hall)))
            for threshold in bins
        ]
        if "entropy" in measure:
            y_values = [
                100 * (sum([1 for prob in prob_uncertain if prob <= threshold + epsilon]) / max(1, len(prob_uncertain)))
                for threshold in bins
            ]

            y_values_hall = [
                100 * (sum([1 for prob in prob_hall if prob <= threshold + epsilon]) / max(1, len(prob_hall)))
                for threshold in bins
            ]

        plt.figure(figsize=(14, 10))
        plt.plot(bins, y_values, marker='o', linestyle='-', color='b', markersize=4, label='NH', linewidth=10)
        plt.plot(bins, y_values_hall, marker='o', linestyle='-', color='r', markersize=4, label='H',
                 linewidth=10)
        # plt.title(title)
        plt.grid(True)
        plt.xlim(1, 0)
        if "entropy" in measure:  # from 4 to 0
            plt.xlim(0, 4)
        plt.ylim(0, 100)
        plt.xlabel(measure.replace("_temp_0.5", "").replace("prob_diff", "Probability Difference").replace("prob",
                                                                                                           "Probability").replace(
            "semantic_entropy", "Semantic entropy"))
        plt.ylabel('Cumulative (%)')
        plt.tick_params(axis='both', which='major')
        if "prompt" in title and "-it" not in title and "Instruct" not in title and "semantic" in measure:
            plt.legend()
        if true_threshold is not None:  # add vertical line for the threshold
            # modify threshold to the closest bin
            threshold = bins[np.argmin(np.abs(bins - true_threshold))]
            plt.axvline(x=threshold, color='k', linestyle='--', label='Threshold')
            # add color to the area between the threshold and the end of the x axis under the hall curve
            if "entropy" in measure:
                plt.fill_between(bins, y_values_hall, 0, where=(bins <= threshold), color='red', alpha=0.3)
            else:
                plt.fill_between(bins, y_values_hall, 0, where=(bins >= threshold), color='red', alpha=0.3)

        plt.xticks(np.arange(1, -0.001, -0.2))
        if "entropy" in measure:  # from 4 to zero
            plt.xticks(np.arange(0, 4.1, 1))
        plt.yticks(np.arange(0, 101, 20))
        plt.tick_params(axis='x', which='major', pad=20, length=10)  # Add padding for x-axis ticks
        plt.tick_params(axis='y', which='major', pad=20, length=10)  # Add padding for y-axis ticks
        plt.tight_layout()
        plt.savefig(path + title + ".pdf", format='pdf')
        plt.close()


    def threshold_check(self, non_hall_val: list[float], hall_val: list[float], entropy=False):

        all_values = sorted(set(non_hall_val + hall_val))
        assert len(all_values) <= len(non_hall_val) + len(
            hall_val), f"{len(all_values)=} {len(non_hall_val)=} {len(hall_val)=}"
        for val in all_values:
            assert val in non_hall_val or val in hall_val, f"{val=}"
        best_threshold = 0.0
        min_misclassifications = float('inf')
        num_misclassified_hall = 0
        num_misclassified_non_hall = 0

        for threshold in all_values:
            misclassified_hall = sum(h > threshold for h in hall_val)
            misclassified_non_hall = sum(n <= threshold for n in non_hall_val)
            if entropy:
                misclassified_hall = sum(h < threshold for h in hall_val)
                misclassified_non_hall = sum(n >= threshold for n in non_hall_val)
            total_misclassified = misclassified_hall + misclassified_non_hall

            if total_misclassified < min_misclassifications:
                min_misclassifications = total_misclassified
                best_threshold = threshold
                num_misclassified_hall = misclassified_hall
                num_misclassified_non_hall = misclassified_non_hall
        return best_threshold, num_misclassified_hall, num_misclassified_non_hall

    def get_threshold(self, data_hall, data_non_hall, parameter):
        """
        Find the threshold between that minimizes the number of
        hallucinations (hall) with values higher than the threshold
        and non-hallucinations (non_hall) with values lower than the threshold.

        """

        size = int(min(len(data_hall), len(data_non_hall)))

        hall_values = data_hall[:size]
        non_hall_values = data_non_hall[:size]
        test_hall_values = data_hall[:size]
        test_non_hall_values = data_non_hall[:size]
        assert len(hall_values) == len(non_hall_values), f"{len(hall_values)=} {len(non_hall_values)=}"

        hall_values = [e[parameter] for e in hall_values]
        non_hall_values = [e[parameter] for e in non_hall_values]
        if "temp_generations" in parameter:
            hall_values = [1 - (len(set(e)) / len(e)) for e in hall_values]
            non_hall_values = [1 - (len(set(e)) / len(e)) for e in non_hall_values]
        threshold, _, _ = self.threshold_check(non_hall_values, hall_values,
                                               entropy=True if "entropy" in parameter else False)
        # check accuracy on test set
        if "temp_generations" in parameter:
            non_hall_miss = sum(
                [1 for e in test_non_hall_values if 1 - (len(set(e[parameter])) / len(e[parameter])) < threshold])
            hall_miss = sum(
                [1 for e in test_hall_values if 1 - (len(set(e[parameter])) / len(e[parameter])) >= threshold])
        elif "entropy" in parameter:
            non_hall_miss = sum([1 for e in test_non_hall_values if e[parameter] >= threshold])
            hall_miss = sum([1 for e in test_hall_values if e[parameter] < threshold])
        else:
            non_hall_miss = sum([1 for e in test_non_hall_values if e[parameter] < threshold])
            hall_miss = sum([1 for e in test_hall_values if e[parameter] >= threshold])
        return threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss





def run_results():
    root_path = "results/"
    if not os.path.exists(root_path + "plots/"):
        os.makedirs(root_path + "plots/")
    cs = case_study(root_path)
    cs.remove_half_hallucinations(root_path)
    cs.plot_all_measures(root_path)
    cs.mitigation_all_based(root_path)
