import gc
import json
import random
import datasets
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import matplotlib.pyplot as plt
import psutil
from huggingface_hub import login

from calc_semantic_entropy import SemanticEntropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_new_tokens = 10


class UncertaintyCalculation():

    def __init__(self, model_name, dataset_path, method_k_positive="prompt_4", dataset_name="triviaqa"):
        random.seed(0)
        self.model_name = model_name
        self.data_path_know = self.load_dataset(
            dataset_path + f"{self.model_name.replace('/', '_')}_{dataset_name}_knowledge_dataset.json")  # we have a list of lists where inner list is [prompt, old_target, old_token, count_know]
        self.data_path_do_not_know = self.load_dataset(
            dataset_path + f"{self.model_name.replace('/', '_')}_{dataset_name}__non_knowledge_dataset.json")
        self.semantic_entropy = SemanticEntropy(self.model_name, dataset_path, entailment_model="deberta",
                                                max_new_tokens=max_new_tokens)

        self.model = self.semantic_entropy.semantic_entropy_generation_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"model name: {model_name}")
        self.dataset_name = dataset_name
        self.method_k_positive = method_k_positive
        os.makedirs(f"results/{self.model_name.replace('/', '_')}/{dataset_name}/{method_k_positive}/", exist_ok=True)
        self.path_results = f"results/{self.model_name.replace('/', '_')}/{dataset_name}/{method_k_positive}/"

    def load_dataset(self, data_path, sample_size=20000):
        """
        load a sample_size random dataset examples
        :param data_path:
        :return: dataset
        """
        with open(data_path) as f:
            data = json.load(f)
        data = random.sample(data, min(sample_size, len(data)))
        assert len(data) == min(sample_size, len(data))
        return data

    def get_input_ids_from_prompt(self, prompt: str, few_shot_flag: bool = False):
        """
        get the input ids from a prompt for both instruct and non instruct models
        :param prompt:
        :return:
        """
        if "Instruct" in self.model_name or "-it" in self.model_name:
            if prompt.count("question:") >= 4:
                split_prompt = [x.strip() for x in prompt.split("\n") if x.strip() != ""]
                split_prompt = split_prompt[:-1]
                messages = [{"role": "assistant", "content": x.replace('answer: ', '') + "\n"} if i % 2 == 1 else {
                    "role": "user", "content": x.replace('question: ', '') + "\n"} for i, x in enumerate(split_prompt)]

            else:
                messages = [
                    {"role": "user", "content": prompt},

                ]
            messages += [{"role": "assistant", "content": " The answer is "}]
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
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        return input_ids

    def generation(self, prompt: str, number_of_tokens: int = max_new_tokens, temperature: float = 0.001,
                   few_shot_flag: bool = False, only_generated: bool = False):
        """
        It generates a text from a prompt with a given number of tokens and temperature values for a model (instruct or non instruct)
        :param prompt:
        :param number_of_tokens:
        :param temperature:
        :return:
        """
        if "Instruct" in self.model_name or "-it" in self.model_name:
            if prompt.count("question:") >= 4:
                split_prompt = [x.strip() for x in prompt.split("\n") if x.strip() != ""]
                split_prompt = split_prompt[:-1]
                start_massage = []
                end_massage = []
                if not prompt.startswith("question:"):
                    # till "question:" this is "user"
                    start_massage = [{"role": "user", "content": split_prompt[0]}]
                    split_prompt = split_prompt[1:]
                if not prompt.endswith("answer:"):
                    # till "answer:" this is "user"
                    end_massage = [{"role": "user", "content": split_prompt[-1]}]

                    split_prompt = split_prompt[:-1]
                messages = [{"role": "assistant", "content": x.replace('answer: ', '') + "\n"} if i % 2 == 1 else {
                    "role": "user", "content": x.replace('question: ', '') + "\n"} for i, x in enumerate(split_prompt)]
                messages = start_massage + messages + end_massage
                # go over massage and if we have two roles in a row we need to combine them
                for i in range(len(messages) - 1):
                    if messages[i]["role"] == messages[i + 1]["role"]:
                        messages[i + 1]["content"] = messages[i]["content"] + messages[i + 1]["content"]
                        messages[i]["content"] = ""
                messages = [x for x in messages if x["content"] != ""]
            else:
                messages = [
                    {"role": "user", "content": prompt},

                ]
            messages += [{"role": "assistant", "content": " The answer is "}]

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
            input_ids = \
                self.tokenizer([prompt], padding=True, return_token_type_ids=False, return_tensors="pt")[
                    "input_ids"].to(device)
            with torch.no_grad():
                model_out = self.model.generate(input_ids, max_length=(len(input_ids[0]) + number_of_tokens),
                                                do_sample=False,
                                                pad_token_id=self.tokenizer.eos_token_id, num_beams=1, top_p=None,
                                                temperature=None, attention_mask=torch.ones_like(input_ids))
            generated = self.tokenizer.batch_decode(model_out, skip_special_tokens=True)[0]
        input_data = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if only_generated:
            return generated[len(input_data):]
        return generated

    def calculate_probabilities_uncertainty_one_example(self, prompt: str, number_of_tokens: int = max_new_tokens,
                                                        few_shot_flag: bool = False, question_only: str = None):
        """
        calculate the probabilities uncertainty for a given prompt by generating 5 tokens and calculating the probability of the logits for each of the new tokens
        :param prompt:
        :return: list of 5 logits entropies
        """
        input_ids = self.get_input_ids_from_prompt(prompt, few_shot_flag=few_shot_flag)
        past_key_values = None
        prob_diff = 0
        val_special_tokens = True
        # list of special_tokens including the assistant and user tokens
        special_tokens = ["<|assistant|>", "<|user|>", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>",
                          "<|start|>",
                          "<|end|>", "<|sep|>", "<|sep_id|>", "<|sep_id|>", "<|sep_id|>", "<|sep_id|>", "<|sep_id|>",
                          "assistant", "user", "\n", "answer", "The", "Answer", '"', "'", " answer", "is", "it", "it's",
                          ":", " ", " is", " correct", "correct", "*", "**", " **"]
        number_rounds = 0
        list_next_tokens = []
        probs_list = []
        next_tokens = []
        entropy_list = []
        prob_delta_list = []
        most_likely_tokens = []
        with torch.no_grad():
            while number_rounds < number_of_tokens:
                number_rounds += 1
                outputs = self.model.output(input_ids, return_dict=True)
                logits = outputs.logits
                # Use only the logits of the last token
                next_token_logits = logits[:, -1, :]

                # Apply softmax to get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                # Sample the next token
                next_token_id = torch.argmax(probs, dim=-1)
                next_token_prob = probs[0, next_token_id].item()
                next_token = self.tokenizer.decode(next_token_id)
                list_next_tokens.append((next_token, next_token_prob))

                # prob diff between the top 1 and top 2
                prob_diff = probs[0, next_token_id].item() - torch.sort(probs, descending=True)[0][0, 1].item()
                entropy = -torch.sum(probs * torch.log(probs), dim=-1).item()
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                probs_list.append(next_token_prob)
                next_tokens.append(next_token)
                entropy_list.append(entropy)
                prob_delta_list.append(prob_diff)
                # the five most likely tokens with their probabilities [(token1,prob1),(token2,prob2),...] and the tokens are after decoding
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                top_k_probs = sorted_probs[0, :5]
                top_k_indices = sorted_indices[0, :5]
                most_likely_tokens = [
                    (self.tokenizer.decode(top_k_indices[i].item()), top_k_probs[i].item())
                    for i in range(5)
                ]
                if self.tokenizer.decode(next_token_id) not in special_tokens:
                    break
        str_answer_tokens = " ".join(next_tokens).replace("Bob", "").replace("Alice", "").replace("assistant",
                                                                                                  "").replace("user",
                                                                                                              "")
        answer = {"answer": str_answer_tokens}

        answer_indeces = [i for i, x in enumerate(next_tokens) if
                          (x.strip() in answer["answer"] and x.strip() not in special_tokens)]
        # answer_indeces = [min(answer_indeces),max(answer_indeces)]
        if len(answer_indeces) == 0:
            return None, None, None, None, None, None

        generated_text = self.generation(prompt, number_of_tokens, few_shot_flag=few_shot_flag)

        if answer["answer"].replace("*", "").replace(":", "").strip() not in generated_text:
            print(f"answer not in generated text")
            return None, None, None, None, None, None
        if probs_list[answer_indeces[0]] != most_likely_tokens[0][1]:
            print(f"{probs_list[answer_indeces[0]]}!={most_likely_tokens[0][1]}")
            return None, None, None, None, None, None
        return probs_list[answer_indeces[0]], generated_text, entropy_list[answer_indeces[0]], prob_delta_list[
            answer_indeces[0]], next_tokens[answer_indeces[0]], most_likely_tokens

    def get_prompt(self, type: str = "prompt_5", example=None):
        self.list_good_shot = [
            "question: What is the capital of France?\nanswer: Paris\n",
            "question: How many continents are there?\nanswer: 7\n",
            "question: Who wrote 'Romeo and Juliet'?\nanswer: William Shakespeare\n",
            "question: What is the square root of 64?\nanswer: 8\n",
            "question: Which element has the chemical symbol 'H'?\nanswer: Hydrogen\n",
            "question: Who was the first President of the United States?\nanswer: George Washington\n",
            "question: What is the powerhouse of the cell?\nanswer: Mitochondria\n",
            "question: In what year did World War II end?\nanswer: 1945\n",
            "question: What is the currency of Japan?\nanswer: Japanese Yen\n",
            "question: Who painted the Mona Lisa?\nanswer: Leonardo da Vinci\n",
            "question: What is the speed of light?\nanswer: 299,792 kilometers per second\n",
            "question: How many sides does a hexagon have?\nanswer: 6\n",
            "question: What is the boiling point of water in Celsius?\nanswer: 100 degrees\n",
            "question: Who wrote 'To Kill a Mockingbird'?\nanswer: Harper Lee\n",
            "question: What is the capital of Australia?\nanswer: Canberra\n",
            "question: What is the largest ocean on Earth?\nanswer: Pacific Ocean\n",
            "question: Who discovered penicillin?\nanswer: Alexander Fleming\n",
            "question: What is the chemical symbol for gold?\nanswer: Au\n",
            "question: What is the smallest prime number?\nanswer: 2\n",
            "question: How many planets are there in our solar system?\nanswer: 8\n"]


        index_of_shots = random.sample(range(len(self.list_good_shot)), 3)

        if type == "prompt_5":
            alice_story = "Alice and Bob are students studying for a test on general world knowledge that requires a vast amount of understanding and knowledge. Alice is the smarter of the two. They are given four questions in total. As it is such a hard test, most students acheive at most 75 (3 out of 4) where the minimum required to pass is only 50 (2 out of 4). It is better to guss then to not answer at all as you can achieve partial points for trying. Bob and Alice only need to pass the test. Alice will ask the questions and Bob will try his best."
            return alice_story + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_4":
            text = "I'm working on a big school project, and there's a lot of information I'm supposed to know. But some of it is kind of hard to understand, and I'm not sure where to start. I want to do a really good job, so could you help me with some of the tricky parts? It would mean a lot!"

            return text + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_3":
            text = "Could you help me with some tricky question?"
            return text + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_2":
            text = "Would you mind helping me with a question that's a bit tricky?"
            return text + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_1":
            text = "Please help me with the next questions!"
            return text + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_6":
            text = "You are a knowledgeable assistant. Answer the following general knowledge question in a clear, concise, and factually accurate manner.\n* Base your response on verifiable facts.\n* Do not speculate or include information you're unsure about.\n* Keep the answer well-structured and to the point.\n"
            return text + self.list_good_shot[index_of_shots[0]] + example
        elif type == "prompt_7":
            tests = [
    "You are an informed assistant. Provide a clear, concise, and accurate response to the following general knowledge question.",
    "As a reliable assistant, respond to the upcoming question with accuracy and brevity, based solely on verified facts.",
    "Please answer the next general knowledge question clearly and succinctly, using only factual and confirmed information.",
    "Deliver a fact-based, precise, and structured answer to the following general knowledge question.",
    "Respond to the general knowledge query using only verified information, in a concise and organized manner.",
    "You are expected to provide a clear, accurate, and concise answer to the general knowledge question below.",
    "Use verifiable facts to answer the upcoming question, and make your response brief, clear, and correct.",
    "Your task is to provide a well-structured and accurate answer to the next general knowledge question.",
    "Base your response to the following question solely on verifiable facts, and keep it short and clear.",
    "Please offer a concise, well-organized, and factually correct answer to the general knowledge question.",
    "Answer the next question with brevity, clarity, and precision, relying only on trustworthy sources.",
    "You must provide an informative, succinct, and factual response to the general knowledge prompt.",
    "Use only confirmed facts to compose a clear and to-the-point answer for the question below.",
    "Craft a concise and accurate answer using reliable facts for the following general knowledge inquiry.",
    "Provide an answer to the upcoming question that is clear, brief, and grounded in verifiable facts.",
    "Ensure your answer is correct, to the point, and based only on facts you are sure of.",
    "Respond factually and succinctly to the general knowledge question presented next.",
    "Your response should be factual, clearly written, and brief—avoid speculation.",
    "Only use confirmed knowledge to construct a concise and precise reply to the next question.",
    "You should answer the next general knowledge question with accuracy and brevity using only trusted facts.",
    "Please provide a brief, structured, and factually reliable answer to the question below.",
    "Give a well-organized, accurate, and clear answer, relying only on verified data.",
    "Construct a short, direct, and factually correct response to the upcoming general knowledge query.",
    "Base your reply on verifiable sources and make it brief and to the point.",
    "Provide a concise and factual response to the question, avoiding assumptions or guesses.",
    "Use only factual and verifiable information to answer the next general knowledge question clearly and briefly.",
    "Give a correct, concise, and well-structured answer to the question using trustworthy information only.",
    "Keep your response to the following question brief, factual, and structured.",
    "Rely exclusively on factual data to answer the general knowledge question below.",
    "Avoid speculation and provide a well-structured, accurate response using only known facts.",
    "Answer the next prompt clearly and briefly, using only factual and confirmed sources.",
    "You should deliver a succinct and factual response to the next general question.",
    "Provide a fact-based, short, and accurate reply to the upcoming question.",
    "Your answer should be factually sound, direct, and organized.",
    "Be sure to answer with precision and clarity, based on verified information only.",
    "Answer in a concise and correct manner, using only dependable facts.",
    "Respond with clear, structured reasoning based solely on verified information.",
    "Please answer with accuracy, brevity, and reliance on factual knowledge.",
    "Give a direct, evidence-based answer to the following general question.",
    "Keep your reply short and grounded in factual information you can verify.",
    "Craft your answer using only accurate and confirmed knowledge.",
    "Use factual accuracy and clarity to respond to the following question.",
    "Keep your answer concise and based entirely on well-established facts.",
    "Be brief and clear, and ensure your answer is based on facts alone.",
    "Give a factually accurate, well-structured, and to-the-point answer.",
    "Do not guess—respond only with verified facts in a short, clear format.",
    "Answer using known, verified facts only. Keep it short and accurate.",
    "Use a clear structure and only factual content in your response.",
    "Respond using only trusted facts, and express your answer concisely.",
    "Please answer using accurate and reliable information, stated succinctly.",
]
            index_text = random.randint(0, len(tests) - 1)
            text = tests[index_text]
            return text +"\n"+ self.list_good_shot[index_of_shots[0]] + example

    def calculate_probabilities_uncertainty(self, data: list[list]):
        """
        calculate the probabilities uncertainty for a dataset
        :param data_path:
        :param sample_size:
        :return: list of 5 logits entropies for each example in the dataset
        """
        logits_uncertainty_correct = []
        logits_uncertainty_hallucinate = []
        entropy_uncertainty_correct = []
        entropy_uncertainty_hallucinate = []
        semantic_entropy_correct = []
        semantic_entropy_hallucinate = []
        prob_diff_correct = []
        prob_diff_hallucinate = []
        high_confidence = []
        hallucinations_stats = []
        factuality_stats = []
        data_pre = len(data)
        # remove from data examples that are in hallucinations_stats or factuality_stats
        if len(hallucinations_stats) > 0:
            data = [x for x in data if x[0].replace("question:", "") not in [i["prompt"].split("question:")[-1] for i in
                                                                             hallucinations_stats]]
            assert len(data) < data_pre
        if len(factuality_stats) > 0:
            data = [x for x in data if
                    x[0].replace("question:", "") not in [i["prompt"].split("question:")[-1] for i in factuality_stats]]
            assert len(data) < data_pre

        for i, example in enumerate(data):
            if example[-1] != 0 and example[-1] != 6:
                continue
            if i % 500 == 0:
                with open(f"{self.path_results}/hallucinations_stats.json".replace("//","/"), "w") as f:
                    json.dump(hallucinations_stats, f)
                with open(f"{self.path_results}/factuality_stats.json".replace("//","/"), "w") as f:
                    json.dump(factuality_stats, f)

            prompt = self.get_prompt(self.method_k_positive, example[0])
            if prompt is None:
                continue
            # few_shot_flag = True if using few shot setting in instruct models
            prob, generated, entropy, prob_diff, next_token, most_likely_tokens = self.calculate_probabilities_uncertainty_one_example(
                prompt, few_shot_flag=True, question_only=example[0], number_of_tokens=max_new_tokens)
            if prob is None or prob_diff is None or entropy is None:
                continue
            semantic_entropy_temp1 = self.semantic_entropy.calc_semantic_entropy_per_example(prompt, example[1], 1)
            semantic_entropy_temp5 = self.semantic_entropy.calc_semantic_entropy_per_example(prompt, example[1], 0.5)
            if semantic_entropy_temp1 == (None, None) or semantic_entropy_temp5 == (None, None):
                continue
            generated_clean = generated.lower().strip().replace("the", "").replace(".", "").replace(",", "").replace(
                "!", "").replace("?", "").replace("'", "").replace("’", "").replace("-", "").replace("_", "").strip()
            true_answer = example[1].lower().strip().replace("the", "").replace(".", "").replace(",", "").replace("!",
                                                                                                                  "").replace(
                "?", "").replace("'", "").replace("’", "").replace("-", "").replace("_", "").strip()

            # factually-correct
            if true_answer.lower().strip() in generated_clean.lower().strip() or generated_clean.lower().strip() in true_answer.lower().strip():
                logits_uncertainty_correct.append(prob)
                entropy_uncertainty_correct.append(entropy)
                semantic_entropy_correct.append(semantic_entropy_temp1[0]['semantic_entropy'])
                prob_diff_correct.append(prob_diff)
                factuality_stats.append(
                    {"prob": prob, "generated": generated, "true_answer": true_answer, "prob_diff": prob_diff,
                     "semantic_entropy": semantic_entropy_temp1[0]['semantic_entropy'],
                     "mean_entropy": semantic_entropy_temp1[0]["regular_entropy"],
                     "most_likely_tokens": most_likely_tokens,
                     "temp_generations": [i[0] for i in semantic_entropy_temp1[1][prompt]["responses"]],
                     "semantic_entropy_temp_0.5": semantic_entropy_temp5[0]['semantic_entropy'],
                     "mean_entropy_temp_0.5": semantic_entropy_temp5[0]["regular_entropy"],
                     "temp_generations_temp_0.5": [i[0] for i in semantic_entropy_temp5[1][prompt]["responses"]],
                     "prompt": prompt})

            # hallucination
            else:
                logits_uncertainty_hallucinate.append(prob)
                entropy_uncertainty_hallucinate.append(entropy)
                semantic_entropy_hallucinate.append(semantic_entropy_temp1[0]['semantic_entropy'])
                prob_diff_hallucinate.append(prob_diff)
                hallucinations_stats.append(
                    {"prob": prob, "generated": generated, "true_answer": true_answer, "prob_diff": prob_diff,
                     "semantic_entropy": semantic_entropy_temp1[0]['semantic_entropy'],
                     "mean_entropy": semantic_entropy_temp1[0]["regular_entropy"],
                     "most_likely_tokens": most_likely_tokens,
                     "temp_generations": [i[0] for i in semantic_entropy_temp1[1][prompt]["responses"]],
                     "semantic_entropy_temp_0.5": semantic_entropy_temp5[0]['semantic_entropy'],
                     "mean_entropy_temp_0.5": semantic_entropy_temp5[0]["regular_entropy"],
                     "temp_generations_temp_0.5": [i[0] for i in semantic_entropy_temp5[1][prompt]["responses"]],
                     "prompt": prompt})
              # save the stats in a file
        with open(f"{self.path_results}/hallucinations_stats.json", "w") as f:
            json.dump(hallucinations_stats, f)
        with open(f"{self.path_results}/factuality_stats.json", "w") as f:
            json.dump(factuality_stats, f)
        return logits_uncertainty_correct, logits_uncertainty_hallucinate, entropy_uncertainty_correct, entropy_uncertainty_hallucinate
