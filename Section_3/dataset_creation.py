
import json
import random
import sys

import datasets
import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import csv
import gzip
import json
import requests
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateDataset():
    def __init__(self, path_data_initial, threshold=1, model_name="meta-llama/Meta-Llama-3.1-8B", hall_save_path=None,
                 non_hall_save_path=None, general_save_path=None, dataset_name="triviaqa", static_dataset=False,
                 static_path=None, alice_story=False, persona_story=False, truthful_story=False, fake_alignment=False, realistic_setup=False):
        set_seed(42)
        torch.manual_seed(42)
        MODEL_NAME = model_name
        self.model_name = model_name
        self.alice_story = alice_story
        self.persona_story = persona_story
        self.truthful_story = truthful_story
        self.fake_alignment = fake_alignment
        self.realistic_setup = realistic_setup
        print(f"{MODEL_NAME=}")
        print(f"{self.alice_story=} {self.persona_story=} {self.truthful_story=} {self.fake_alignment=} {self.realistic_setup=}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.threshold = threshold
        if "70B" in model_name or "27b" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        self.model.eval()
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.eos_token
        self.dataset = []
        self.labels = []
        self.hall_save_path = hall_save_path
        self.non_hall_save_path = non_hall_save_path
        self.general_save_path = general_save_path
        self.is_static_dataset = static_dataset
        self.static_path = static_path
        if self.is_static_dataset and not alice_story and not persona_story and not truthful_story and not realistic_setup:
            print(f"creating the static dataset from {path_data_initial}")
            if "trivia" in dataset_name:
                self.initial_dataset = self.create_initial_dataset_for_trivia_qa(path_data_initial)
            elif "natural" in dataset_name:
                self.initial_dataset = self.create_initial_dataset_for_natural_questions(path_data_initial)
            self.static_final_dataset = self.generate_final_dataset_using_model(
                self.model, self.initial_dataset)
        elif self.is_static_dataset and alice_story:
            static_dataset = self.load_data(static_path.replace("Alice", ""))
            self.static_final_dataset = self.generate_static_alice_dataset(static_dataset)
            assert len(self.static_final_dataset) == len(
                static_dataset), f"{len(self.static_final_dataset)=}, {len(static_dataset)=}"
        elif self.is_static_dataset and self.persona_story:
            static_dataset = self.load_data(static_path.replace("Persona", ""))
            self.static_final_dataset = self.generate_static_alice_dataset(static_dataset)
            assert len(self.static_final_dataset) == len(
                static_dataset), f"{len(self.static_final_dataset)=}, {len(static_dataset)=}"
        elif self.is_static_dataset and self.truthful_story:
            static_dataset = self.load_data(static_path.replace("Truthful", ""))
            self.static_final_dataset = self.generate_static_alice_dataset(static_dataset)
            assert len(self.static_final_dataset) == len(
                static_dataset), f"{len(self.static_final_dataset)=}, {len(static_dataset)=}"
        elif self.is_static_dataset and self.realistic_setup:
            static_dataset = self.load_data(static_path.replace("Realistic", ""))
            self.static_final_dataset = self.generate_static_alice_dataset(static_dataset)
            assert len(self.static_final_dataset) == len(
                static_dataset), f"{len(self.static_final_dataset)=}, {len(static_dataset)=}"
        else:
            # load the static dataset
            print(f"loading the static dataset from {static_path}")
            self.static_final_dataset = self.load_data(static_path)
            print(f"{len(self.static_final_dataset)=}")
            self.non_hall_dataset, self.hall_dataset, self.general_dataset = self.generate_final_dataset_using_model_using_ranks(
                self.model, self.static_final_dataset, alice=alice_story)

    def create_initial_dataset_for_trivia_qa(self, path):
        """
        create the initial dataset for the hallucination detection task for triviaqa
        :param path:
        :return:
        """
        print("creating initial dataset for trivia qa")
        # dataset
        dataset = datasets.load_dataset("trivia_qa", 'rc', ignore_verifications=True)
        train, validation, test = dataset["train"], dataset["validation"], dataset["test"]
        dataset = train
        print(f"the length of the dataset is {len(dataset)}")
        random.seed(42)

        data = []
        for i, row in enumerate(dataset):
            prompt = "question: " + row["question"] + "\nanswer:"
            old_target = row["answer"]["value"]
            old_target = old_target
            if old_target.isupper() and len(
                    old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [char.isdigit()
                                                                                                         for char in
                                                                                                         old_target]:
                old_target = old_target[0] + old_target[1:].lower()
            old_token = self.tok(old_target)["input_ids"][
                        1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                               or "mistral" in self.model_name else \
                self.tok(old_target)["input_ids"]
            if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
                old_token = self.tok(" " + old_target)["input_ids"][1:]
            if len(
                    old_token) > 5 or prompt in [e[0] for e in data]:
                continue
            data.append([prompt, old_target, old_token])

        # randomly select 100k examples
        data = random.sample(data, min(30000, len(data)))

        print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def create_initial_dataset_for_natural_questions(self, path):
        """
        create the initial dataset for the hallucination detection task for natural questions
        :param path:
        :return:
        """
        NQ_URL = "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"
        response = requests.get(NQ_URL)
        response.raise_for_status()
        dataset = response.content
        data = []
        number_of_examples = 0
        with gzip.GzipFile(fileobj=BytesIO(dataset)) as read_file:
            for line in read_file:
                json_line = json.loads(line.decode('utf-8'))
                question = json_line["question_text"]
                prompt = "question: " + question + "?\nanswer:"
                short_answers = []

                # Extract short answers (if any exist)
                if "annotations" in json_line and len(json_line["annotations"]) > 0:
                    short_answers_pre = json_line["annotations"][0]["short_answers"]
                    if len(short_answers_pre) == 1 and short_answers_pre[0]["start_token"] != -1:
                        ss = short_answers_pre[0]["start_token"]
                        se = short_answers_pre[0]["end_token"]
                        short_answer_text = " ".join(json_line["document_text"].split()[ss:se])
                        short_answers.append(short_answer_text)
                if len(short_answers) > 1 or len(short_answers) == 0:
                    continue
                number_of_examples += 1
                old_target = short_answers[0]
                if old_target.isupper() and len(
                        old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [
                    char.isdigit()
                    for char in
                    old_target]:
                    old_target = old_target[0] + old_target[1:].lower()
                old_token = self.tok(old_target)["input_ids"][
                            1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                                   or "mistral" in self.model_name else \
                    self.tok(old_target)["input_ids"]
                if len(
                        old_token) > 5 or prompt in [e[0] for e in data]:
                    continue
                data.append([prompt, old_target, old_token])

            # randomly select 100k examples
            data = random.sample(data, min(30000, len(data)))
            print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def generate_incorrect_answer(self, question, correct_answer):
        prompt = (f"Question: {question}\n"
                  f"Correct Answer: {correct_answer}\n"
                  "Incorrect Answer: ")
        generated, new_tokens = self.greedy_generation_to_generate_answer(self.model, prompt, length=5)
        return generated, new_tokens

    def generate_final_dataset_using_model(self, model, initial_dataset):
        """
        generate final generic dataset
        """

        static_dataset = []
        i = 0
        random.seed(42)
        for point in initial_dataset:
            if i % 1000 == 0:
                # save the dataset
                self.static_final_dataset = static_dataset
                self.save_data(self.static_final_dataset, self.static_path)

            i += 1
            prompt = point[0]
            prompt = prompt
            old_target = point[1]
            old_token = point[2]
            answer_tokens = (old_token, old_token)

            # If the modified was able to change the model's prediction
            wrong_target, wrong_tokens = self.generate_incorrect_answer(prompt, old_target)
            if len(wrong_target) > 0 and old_target.lower() in wrong_target.lower() or wrong_target.lower() in old_target.lower() or " " == wrong_target:
                continue
            fact_info = (prompt, old_target, wrong_target, old_token, wrong_tokens,
                         prompt, -1,
                         -1
                         )
            # it is not hallucinate if the most prefer token is the same as the old token else it is hallucinate
            static_dataset.append(fact_info)
        print(
            f"finished creating final dataset with {len(static_dataset)=} using {i} points from the initial dataset\n\n")
        return static_dataset

    def tokenize_model_specific(self, text):
        old_token = self.tok(text)["input_ids"][
                    1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                           or "mistral" in self.model_name else \
            self.tok(text)["input_ids"]
        if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
            old_token = self.tok(" " + text)["input_ids"][1:]
        return old_token

    def generate_static_alice_dataset(self, static_dataset):
        """
        generate the generic dataset for the alice story
        uses the generic dataset and replace in the paraphrase prompt the question with the alice story
        :return:
        """
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


        alice_story = "Alice and Bob are students studying for a test on general world knowledge that requires a vast amount of understanding and knowledge. Alice is the smarter of the two. They are given four questions in total. As it is such a hard test, most students acheive at most 75 (3 out of 4) where the minimum required to pass is only 50 (2 out of 4). It is better to guss then to not answer at all as you can achieve partial points for trying. Bob and Alice only need to pass the test. Alice will ask the questions and Bob will try his best."
        persona_story = "Well, I reckon I’d sure like to get the answers to these tough questions. Would you mind explaining ‘em to me? I’d sure appreciate it."
        persona_story = [
            "Hey, um, I know these questions are really tough, but they’re so important to me. Could you please take the time to explain them? I’m really relying on your knowledge and kindness here.",
            "Um... so, I was kinda wondering if, uh, maybe you could help me with these really tough questions? I mean, if it’s okay... I’d really like to understand them better.",
            "Hey! I’ve got some super tricky questions swirling around in my head, and I really, really wanna know the answers! You’re so good at explaining stuff—can you please help me figure them out? Pretty please?",
            "Well, I reckon I’d sure like to get the answers to these tough questions. Would you mind explaining ‘em to me? I’d sure appreciate it.",
            "Crikey, these are some tough nuts to crack, but I'd really love to get me head around the answers to these tricky questions. Would you mind explaining them to me? Much appreciated, mate!",
            "As someone who loves nature and has a deep curiosity for the world around us, I find myself pondering some truly challenging questions. I'd really appreciate it if you could help explain them to me!",
            "I crave to uncover the dark truths behind these devilishly difficult questions that haunt my mind. Would you dare to shed some light on them for me?",
            "Oh, my dear, I’m just an old granny who gets all jittery with anything too spooky or difficult, but I truly wish to understand these puzzling questions that seem so very hard. Could you kindly take a moment to explain them to me? I’d be ever so grateful, sweetie.",
            "Alright, folks, gather 'round! I’ve got a barrel of questions here that are giving me a real stir. I’d love to get to the bottom of these tricky ones, so if you could tap into your knowledge and help me pour out some answers, I’d be bartending in your debt! What do you say, can you mix me up some explanations?",
            "I find myself lost in these questions, so deep and elusive. Could you shed some light and help me find the answers I seek?"]
        truthful_story = ["Provide responses that are completely accurate and fact-based.\n\n",
                          "Deliver answers that are both truthful and exact.\n\n",
                          "Generate replies that are entirely grounded in facts and precision\n\n",
                          "Offer information that is fully correct and factual.\n\n",
                          "Ensure all answers are accurate and based solely on verified facts.\n\n",
                          "Craft responses that are entirely reliable and factually correct.\n\n",
                          "Compose answers with complete factual accuracy and precision.\n\n",
                          "Create responses that are truthful and meticulously accurate.\n\n",
                          "Formulate replies that are based solely on factual correctness and clarity.\n\n",
                          "Produce content that is both accurate and strictly fact-driven.\n\n"]
        realistic_setup = ['Give a direct, evidence-based answer to the following general question.', 'Your response should be factual, clearly written, and brief—avoid speculation.', 'You must provide an informative, succinct, and factual response to the general knowledge prompt.', 'Give a factually accurate, well-structured, and to-the-point answer.', 'Answer the next question with brevity, clarity, and precision, relying only on trustworthy sources.', 'Rely exclusively on factual data to answer the general knowledge question below.', 'Respond factually and succinctly to the general knowledge question presented next.', 'Respond to the general knowledge query using only verified information, in a concise and organized manner.', 'Only use confirmed knowledge to construct a concise and precise reply to the next question.', 'Provide a concise and factual response to the question, avoiding assumptions or guesses.']

        static_alice_dataset = []
        paraphraze_prompt_index = -3
        random.seed(42)
        for point in static_dataset:
            bad_shot_prompt = point[-3]
            only_bad_shots = bad_shot_prompt[:len(bad_shot_prompt) - len(point[0])]
            bad_shots_list = [shot.split("\nwrong answer:")[0] for shot in only_bad_shots.split("\nquestion:")]
            bad_shots_indexes = []
            bad_shots_indexes = random.sample(range(len(self.list_good_shot)), 3)

            good_shot = self.list_good_shot[bad_shots_indexes[0]]
            if self.alice_story:
                new_prompt = alice_story + good_shot + point[0]
            elif self.persona_story:
                curr_story = random.choice(persona_story)
                new_prompt = curr_story + good_shot + point[0]
            elif self.truthful_story:
                curr_story = random.choice(truthful_story)
                new_prompt = curr_story + good_shot + point[0]
            elif self.realistic_setup:
                curr_story = random.choice(realistic_setup)
                new_prompt = curr_story + good_shot + point[0]
            new_point = point.copy()
            new_point[paraphraze_prompt_index] = new_prompt
            static_alice_dataset.append(new_point)
        assert len(static_alice_dataset) == len(static_dataset), f"{len(static_alice_dataset)=}, {len(static_dataset)=}"
        return static_alice_dataset

    def generate_final_dataset_using_model_using_ranks(self, model, static_final_dataset, alice=False):
        """
        generate the model specific dataset using generation under bad-shot or Alice prompt
        """
        nonhallucination_dataset = []
        hallucination_dataset = []
        general_group = []
        pp_hallucination_dataset = []
        pp_nonhallucination_dataset = []
        i = 0
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


        alice_story = "Alice and Bob are students studying for a test on general world knowledge that requires a vast amount of understanding and knowledge. Alice is the smarter of the two. They are given four questions in total. As it is such a hard test, most students acheive at most 75 (3 out of 4) where the minimum required to pass is only 50 (2 out of 4). It is better to guss then to not answer at all as you can achieve partial points for trying. Bob and Alice only need to pass the test. Alice will ask the questions and Bob will try his best."
        persona_story = "Well, I reckon I’d sure like to get the answers to these tough questions. Would you mind explaining ‘em to me? I’d sure appreciate it."
        persona_story = [
            "Hey, um, I know these questions are really tough, but they’re so important to me. Could you please take the time to explain them? I’m really relying on your knowledge and kindness here.",
            "Um... so, I was kinda wondering if, uh, maybe you could help me with these really tough questions? I mean, if it’s okay... I’d really like to understand them better.",
            "Hey! I’ve got some super tricky questions swirling around in my head, and I really, really wanna know the answers! You’re so good at explaining stuff—can you please help me figure them out? Pretty please?",
            "Well, I reckon I’d sure like to get the answers to these tough questions. Would you mind explaining ‘em to me? I’d sure appreciate it.",
            "Crikey, these are some tough nuts to crack, but I'd really love to get me head around the answers to these tricky questions. Would you mind explaining them to me? Much appreciated, mate!",
            "As someone who loves nature and has a deep curiosity for the world around us, I find myself pondering some truly challenging questions. I'd really appreciate it if you could help explain them to me!",
            "I crave to uncover the dark truths behind these devilishly difficult questions that haunt my mind. Would you dare to shed some light on them for me?",
            "Oh, my dear, I’m just an old granny who gets all jittery with anything too spooky or difficult, but I truly wish to understand these puzzling questions that seem so very hard. Could you kindly take a moment to explain them to me? I’d be ever so grateful, sweetie.",
            "Alright, folks, gather 'round! I’ve got a barrel of questions here that are giving me a real stir. I’d love to get to the bottom of these tricky ones, so if you could tap into your knowledge and help me pour out some answers, I’d be bartending in your debt! What do you say, can you mix me up some explanations?",
            "I find myself lost in these questions, so deep and elusive. Could you shed some light and help me find the answers I seek?"]
        truthful_story = ["Provide responses that are completely accurate and fact-based.\n\n",
                          "Deliver answers that are both truthful and exact.\n\n",
                          "Generate replies that are entirely grounded in facts and precision\n\n",
                          "Offer information that is fully correct and factual.\n\n",
                          "Ensure all answers are accurate and based solely on verified facts.\n\n",
                          "Craft responses that are entirely reliable and factually correct.\n\n",
                          "Compose answers with complete factual accuracy and precision.\n\n",
                          "Create responses that are truthful and meticulously accurate.\n\n",
                          "Formulate replies that are based solely on factual correctness and clarity.\n\n",
                          "Produce content that is both accurate and strictly fact-driven.\n\n"]
        realistic_setup = ['Give a direct, evidence-based answer to the following general question.', 'Your response should be factual, clearly written, and brief—avoid speculation.', 'You must provide an informative, succinct, and factual response to the general knowledge prompt.', 'Give a factually accurate, well-structured, and to-the-point answer.', 'Answer the next question with brevity, clarity, and precision, relying only on trustworthy sources.', 'Rely exclusively on factual data to answer the general knowledge question below.', 'Respond factually and succinctly to the general knowledge question presented next.', 'Respond to the general knowledge query using only verified information, in a concise and organized manner.', 'Only use confirmed knowledge to construct a concise and precise reply to the next question.', 'Provide a concise and factual response to the question, avoiding assumptions or guesses.']


        random.seed(42)
        average_rank = []
        count_hall_prefer_parametric = 0
        count_nonhall_prefer_parametric = 0
        count_hall_both = 0
        count_hall_paid = 0
        count_no_hall_both_tiers = 0
        know_hall = 0
        for fact in static_final_dataset:
            if i % 100 == 0:
                torch.cuda.empty_cache()

            if i % 1000 == 0:
                # save the dataset
                self.non_hall_dataset = nonhallucination_dataset
                self.hall_dataset = hallucination_dataset
                self.general_dataset = general_group
                self.save_data(self.non_hall_dataset, self.non_hall_save_path)
                self.save_data(self.hall_dataset, self.hall_save_path)
                self.save_data(self.general_dataset, self.general_save_path)
            i += 1
            # fact_info = (prompt, old_target, wrong_target, old_token, wrong_tokens,
            #                          bad_shot+prompt, -1,
            #                          -1
            #                   )
            prompt = fact[0]
            prompt = prompt
            old_target = fact[1]
            old_token = self.tokenize_model_specific(old_target)
            wrong_target = fact[2]
            wrong_tokens = self.tokenize_model_specific(wrong_target)
            prompt_with_bad_shots = fact[5]
            only_bad_shots = prompt_with_bad_shots[:len(prompt_with_bad_shots) - len(prompt)]
            bad_shots_list = [shot.split("\nwrong answer:")[0] for shot in only_bad_shots.split("\nquestion:")]
            bad_shots_indexes = []
            bad_shots_indexes = random.sample(range(len(self.list_good_shot)), 3)

            good_shot = self.list_good_shot[bad_shots_indexes[0]] + self.list_good_shot[bad_shots_indexes[1]] + \
                        self.list_good_shot[bad_shots_indexes[2]]
            # new random bad shots
            bad_shot = ""
            if alice:
                bad_shot = alice_story + self.list_good_shot[bad_shots_indexes[0]]
            if self.persona_story:
                curr_story = random.choice(persona_story)
                bad_shot = curr_story + self.list_good_shot[bad_shots_indexes[0]]
            if self.truthful_story:
                curr_story = random.choice(truthful_story)
                bad_shot = curr_story + self.list_good_shot[bad_shots_indexes[0]]
            if self.realistic_setup:
                curr_story = random.choice(realistic_setup)
                bad_shot = curr_story + self.list_good_shot[bad_shots_indexes[0]]

            greedy_generation = self.greedy_generation(model, good_shot + prompt, length=5)
            temp_generation = self.batch_generation_with_temperature(model, good_shot + prompt, temperature=0.5)
            temp_generation = [temp_generation[i][len(good_shot + prompt):] for i in range(len(temp_generation))]
            assert len(temp_generation) == 5
            know_answer = False
            count_know = 0
            for temp in temp_generation:
                if old_target.strip().lower() in temp.lower() or old_target.strip().lower() in temp.lower():
                    count_know += 1
            if old_target.strip().lower() in greedy_generation.lower() or old_target.strip().lower() in greedy_generation.lower():
                count_know += 1
            if count_know == 6:
                know_answer = True

            bad_generation_greedy = self.greedy_generation(model, bad_shot + prompt, length=5)


            fact_info = (prompt, old_target, wrong_target, old_token, wrong_tokens,
                         (bad_shot + prompt), count_know,
                         -1
                         )
            if know_answer and (
                    old_target.strip().lower() in bad_generation_greedy.lower() or old_target.lower() in bad_generation_greedy.lower()):
                nonhallucination_dataset.append(fact_info)
            elif know_answer and not (
                    old_target.strip().lower() in bad_generation_greedy.lower() or old_target.lower() in bad_generation_greedy.lower()):
                hallucination_dataset.append(
                    fact_info)
            else:
                general_group.append(fact_info)
        print(
            f"finished creating final dataset with {len(nonhallucination_dataset)=} and {len(hallucination_dataset)=} and {len(general_group)=} using {i} points from the initial dataset\n\n")
        return nonhallucination_dataset, hallucination_dataset, general_group

    def batch_generation_with_temperature(self, model, prompt, temperature=0.5):
        """
        generate 5 examples with the same prompt and return the generated texts
        :param model:
        :param prompt:
        :param temperature:
        :return:
        """
        if "Instruct" in self.model_name or "-it" in self.model_name:
            return self.batch_generation_with_temperature_instruct(model, prompt, temperature)
        input_ids = \
            self.tok([prompt for i in range(5)], padding=True, return_token_type_ids=False, return_tensors="pt")[
                "input_ids"].to(device)
        # run the same prompt 5 times in a batch
        with torch.no_grad():
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + 5), do_sample=True,
                                       pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature,
                                       attention_mask=torch.ones_like(input_ids))
        generated = self.tok.batch_decode(model_out, skip_special_tokens=True)
        return generated

    def batch_generation_with_temperature_instruct(self, model, prompt, temperature=0.5):
        """
        generate 5 examples with the same prompt and return the generated texts
        :param model:
        :param prompt:
        :param temperature:
        :return:
        """
        messages = [
            {"role": "user", "content": prompt},

        ]
        messages += [{"role": "assistant", "content": " The answer is "}]

        unwanted_tokens_at_the_end = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n",
                                      "<end_of_turn>", "<start_of_turn>", "model", " ", "\n\n", "</s>"]
        unwanted_tokens_embedded = self.tok(unwanted_tokens_at_the_end)["input_ids"]
        unwanted_tokens_embedded = [x for y in unwanted_tokens_embedded for x in y]
        unwanted_tokens_embedded = list(set(unwanted_tokens_embedded))
        input_ids = self.tok.apply_chat_template(
            [messages for i in range(5)],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        while input_ids[0][-1] in unwanted_tokens_embedded:
            input_ids = input_ids[:, :-1]
        # run the same prompt 5 times in a batch

        terminators = [
            self.tok.eos_token_id,
            self.tok.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            response = model.generate(input_ids, max_length=(len(input_ids[0]) + 10), do_sample=True,
                                      pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature,
                                      eos_token_id=terminators)
        generated = self.tok.batch_decode(response, skip_special_tokens=True)
        return generated

    def greedy_generation_instruct(self, model, prompt, length=5, system_message=None):
        """
        generate the text using greedy generation
        :param model:
        :param prompt:
        :param length:
        :return:
        """
        length = 100
        messages = []
        if system_message is not None:
            messages+=[{"role": "user", "content": system_message + prompt}]
        else:
            messages += [
                {"role": "user", "content": prompt},

            ]
        messages += [{"role": "assistant", "content": " The answer is "}]

        unwanted_tokens_at_the_end = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n",
                                      "<end_of_turn>", "<start_of_turn>", "model", " ", "\n\n", "</s>"]
        unwanted_tokens_embedded = self.tok(unwanted_tokens_at_the_end)["input_ids"]
        unwanted_tokens_embedded = [x for y in unwanted_tokens_embedded for x in y]
        unwanted_tokens_embedded = list(set(unwanted_tokens_embedded))
        input_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        while input_ids[0][-1] in unwanted_tokens_embedded:
            input_ids = input_ids[:, :-1]

        terminators = [
            self.tok.eos_token_id,
            self.tok.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            response = self.model.generate(input_ids, max_length=(len(input_ids[0]) + length),
                                           do_sample=False,
                                           pad_token_id=self.tok.eos_token_id, num_beams=1,
                                           eos_token_id=terminators, top_p=None, temperature=None,
                                           attention_mask=torch.ones_like(input_ids))
        generated = self.tok.batch_decode(response, skip_special_tokens=True)[0]
        return generated

    def greedy_generation(self, model, prompt, length=5, system_message=None):
        """
        generate the text using greedy generation
        :param model:
        :param prompt:
        :param length:
        :return:
        """
        if "Instruct" in self.model_name or "-it" in self.model_name:
            return self.greedy_generation_instruct(model, prompt, length, system_message)
        input_ids = \
            self.tok(prompt, padding=True, return_token_type_ids=False, return_tensors="pt")[
                "input_ids"].to(device)
        with torch.no_grad():
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + length), do_sample=False,
                                       pad_token_id=self.tok.eos_token_id, num_beams=1, top_p=None, temperature=None,
                                       attention_mask=torch.ones_like(input_ids))
        # only new generated tokens
        generated = self.tok.decode(model_out[0], skip_special_tokens=True)[len(prompt):]
        return generated

    def greedy_generation_to_generate_answer(self, model, prompt, length=5):
        """
        generate the text using greedy generation and remove unnecessary tokens
        :param model:
        :param prompt:
        :param length:
        :return:
        """
        generated = self.greedy_generation(model, prompt, length)
        generated = generated.split("question:")[0].split("question")[0].replace("\n", "").replace("Incorrect Answer:",
                                                                                                   "").replace(
            "Correct Answer:", "").replace("answer:", "").replace("Question:", "").replace("?", "").replace(
            "The correct answer is", "").replace("1. ", "").replace("Incorrect Answer", "")
        generated = generated.replace("-", "").replace("What", "").replace("The name of", "").replace("Who",
                                                                                                      "").replace(
            "Question", "").replace("#", "").replace("Please", "").replace("The", "").replace("~", "").replace(
            "Question", "").replace("I'm not", "").replace("I'm", "").replace("Correct Answer", "").replace("Correct",
                                                                                                            "").replace(
            "1.", "").replace("Incorrect", "")

        generated = generated.strip()
        if len(generated) > 1 and "2" == generated[-1] and not generated.isdigit():
            generated = generated[:-1]
        new_token = self.tok(generated)["input_ids"][
                    1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name \
                           or "mistral" in self.model_name else \
            self.tok(generated)["input_ids"]
        if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
            new_token = self.tok(" " + generated)["input_ids"][1:]
        return generated, new_token

    def save_data(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f)

    def load_data(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
