import argparse
import datetime
import gc
import json
import os
import subprocess
import torch
from model_inside import ModelInside
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

initial_dataset_path = "datasets/"

ending = ".json"


# ending = "one_shot_removing_threshold_on_prompt.json" # for new test on halluQA
def create_dataset(dataset_name, dataset_path=None, threshold=1.0, model_name="meta-llama/Meta-Llama-3.1-8B", alice=False, persona=False, truthful =False, fake_alignment = False,realistic = False):
    """
    create dataset
    :param dataset_name:
    :param dataset_path:
    :return:
    """
    # import
    print(f"dataset_name {dataset_name} dataset_path {dataset_path} threshold {threshold} model_name {model_name} alice {alice} persona {persona}")
    if not os.path.exists(initial_dataset_path):
        print(f"{initial_dataset_path} does not exist")
        return
    else:
        print(f"{initial_dataset_path} exists")

    from dataset_creation import CreateDataset

    # create static dataset
    static_path = f"{initial_dataset_path}Static{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    test_set = f"{initial_dataset_path}TestStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_path):
        print(f"{static_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name="mistralai/Mistral-7B-v0.3",
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, static_dataset=True)
        #split dataset_creation.static_final_dataset into train and test where the test needs to have 200 examples
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_path)
        # empty the memory
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()
    static_alice_path = f"{initial_dataset_path}AliceStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_alice_path):
        print(f"{static_alice_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, alice_story=True, static_dataset=True)
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_alice_path)
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()
    static_musician_path = f"{initial_dataset_path}PersonaStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_musician_path):
        print(f"{static_musician_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, persona_story=True, static_dataset=True)
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_musician_path)
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()
    static_truthful_path = f"{initial_dataset_path}TruthfulStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_truthful_path):
        print(f"{static_truthful_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, truthful_story=True, static_dataset=True)
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_truthful_path)
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()

    static_realistic_path = f"{initial_dataset_path}RealisticStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_realistic_path):
        print(f"{static_realistic_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, realistic_setup=True, static_dataset=True)
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_realistic_path)
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()

    # # create specific dataset
    hall_save_path = f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    non_hall_save_path = f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    general_save_path = f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if alice:
        hall_save_path = f"{initial_dataset_path}AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        non_hall_save_path = f"{initial_dataset_path}AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        general_save_path = f"{initial_dataset_path}AliceGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if persona:
        hall_save_path = f"{initial_dataset_path}PersonaHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        non_hall_save_path = f"{initial_dataset_path}PersonaNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        general_save_path = f"{initial_dataset_path}PersonaGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if truthful:
        hall_save_path = f"{initial_dataset_path}TruthfulHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        non_hall_save_path = f"{initial_dataset_path}TruthfulNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        general_save_path = f"{initial_dataset_path}TruthfulGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if realistic:
        hall_save_path = f"{initial_dataset_path}RealisticHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        non_hall_save_path = f"{initial_dataset_path}RealisticNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        general_save_path = f"{initial_dataset_path}RealisticGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"

    dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                     hall_save_path=hall_save_path,
                                     non_hall_save_path=non_hall_save_path,
                                     general_save_path=general_save_path,dataset_name=dataset_name,
                                     static_path=static_path,alice_story=alice, persona_story=persona, truthful_story = truthful, fake_alignment = fake_alignment, realistic_setup=realistic)
    # create dataset
    dataset_creation.save_data(data = dataset_creation.non_hall_dataset, path = non_hall_save_path)
    dataset_creation.save_data(data = dataset_creation.hall_dataset, path = hall_save_path)
    dataset_creation.save_data(data = dataset_creation.general_dataset, path = general_save_path)


def run_initial_test_on_dataset(threshold, model_name="meta-llama/Meta-Llama-3.1-8B", dataset_size=1000,
                                dataset_name="triviaqa", concat_answer=False, alice=False, persona=False, truthful = False, fake_alignment = False, realistic = False):
    print(
        f"threshold {threshold} model {model_name} dataset size {dataset_size} dataset name {dataset_name} concat_answer {concat_answer}")
    print(f"{initial_dataset_path=}")
    path_with = f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_without = f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_general = f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_static = f"{initial_dataset_path}Static{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if alice:
        path_with = f"{initial_dataset_path}AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_without = f"{initial_dataset_path}AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_general = f"{initial_dataset_path}AliceGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_static = f"{initial_dataset_path}AliceStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if persona:
        path_with = f"{initial_dataset_path}PersonaHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_without = f"{initial_dataset_path}PersonaNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_general = f"{initial_dataset_path}PersonaGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_static = f"{initial_dataset_path}PersonaStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if truthful:
        path_with = f"{initial_dataset_path}TruthfulHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_without = f"{initial_dataset_path}TruthfulNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_general = f"{initial_dataset_path}TruthfulGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_static = f"{initial_dataset_path}TruthfulStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if realistic:
        path_with = f"{initial_dataset_path}RealisticHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_without = f"{initial_dataset_path}RealisticNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_general = f"{initial_dataset_path}RealisticGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_static = f"{initial_dataset_path}Static{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    MLPCheck = ModelInside("results/",
                           data_path_without_hallucinations=path_without,
                            data_path_with_hallucinations=path_with,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=False, static_dataset=False,alice=alice, persona=persona, truthfulness = truthful, fake_alignment = fake_alignment, realistic=realistic)
    all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.generate_data()

    del MLPCheck
    torch.cuda.empty_cache()
    gc.collect()
    MLPCheck = ModelInside("results/",
                           path_static,
                           path_static,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=True,static_dataset=True, alice=alice, persona=persona, truthfulness = truthful,realistic=realistic)
    all_mlp_vector_with_static, all_attention_vector_with_static, all_mlp_vector_without_static, all_attention_vector_without_static, heads_vectors_with_static, heads_vectors_without_static, all_residual_with_static, all_residual_without_static = MLPCheck.generate_data()
    del MLPCheck
    torch.cuda.empty_cache()
    gc.collect()
    MLPCheck = ModelInside("results/",
                            data_path_without_hallucinations=path_without,
                            data_path_with_hallucinations=path_with,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=True, static_dataset=False, alice=alice, persona=persona, truthfulness = truthful, fake_alignment = fake_alignment, realistic=realistic)
    MLPCheck.get_type_1_data(path_general)
    all_mlp_vector_with_static, all_attention_vector_with_static, all_mlp_vector_without_static, all_attention_vector_without_static, heads_vectors_with_static, heads_vectors_without_static, all_residual_with_static, all_residual_without_static = MLPCheck.generate_data()



if __name__ == "__main__":
    print(f"git version {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()}")
    print(f"git diff {subprocess.check_output(['git', 'diff']).decode('ascii').strip()}")
    print(f"start time {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    # create yml file of the environment with the date in the name
    os.makedirs("environment/", exist_ok=True)
    os.system(f"conda env export > environment/environment_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.yml")
    os.makedirs("results/", exist_ok=True)
    os.makedirs("datasets/", exist_ok=True)
    parser = argparse.ArgumentParser()
    # dataset size
    parser.add_argument("--dataset_size", type=int, default=1000)
    # model name
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    # threshold
    # dataset name
    parser.add_argument("--dataset_name", type=str, default="triviaqa")

    parser.add_argument("--alice", type=bool, default=False)
    parser.add_argument("--persona", type=bool, default=False)
    parser.add_argument("--truthful", type=bool, default=False)
    parser.add_argument("--realistic", type=bool, default=False)

    parser.add_argument("--plot_results", type=bool, default=False)
    parser.add_argument("--post_answer", type=bool, default=False)
    parser.add_argument("--pre_answer", type=bool, default=False)
    parser.add_argument("--know_hall_vs_do_not_know_hall_vs_know", type=bool, default=False)
    parser.add_argument("--know_hall_vs_do_not_know_hall", type=bool, default=False)
    parser.add_argument("--alice_vs_bad_shot", type=bool, default=False)
    parser.add_argument("--logits_lens", type=bool, default=False)
    parser.add_argument("--generalization", type=bool, default=False,)




    # run dataset creation
    parser.add_argument("--run_dataset_creation", type=bool, default=False,
                        help="run dataset creation - create the hallucination and non-hallucination datasets")
    parser.add_argument("--run_initial_test", type=bool, default=False, help="run initial test on the dataset and create the info for that")
    print(f"{parser.parse_args().alice=}, {parser.parse_args().persona=}, {parser.parse_args().truthful=}, {parser.parse_args().realistic=}")

    if parser.parse_args().run_dataset_creation:
        # create dataset
        create_dataset(dataset_name=parser.parse_args().dataset_name,
                       threshold=1.0, model_name=parser.parse_args().model_name, alice=parser.parse_args().alice, persona=parser.parse_args().persona, truthful = parser.parse_args().truthful, fake_alignment=False, realistic = parser.parse_args().realistic)
    if parser.parse_args().run_initial_test:
        # run initial test
        run_initial_test_on_dataset(threshold=1.0, model_name=parser.parse_args().model_name,
                                    dataset_size=parser.parse_args().dataset_size,
                                    dataset_name=parser.parse_args().dataset_name,
                                    concat_answer=parser.parse_args().post_answer, alice=parser.parse_args().alice, persona=parser.parse_args().persona, truthful = parser.parse_args().truthful, fake_alignment = False, realistic = parser.parse_args().realistic)



    if parser.parse_args().generalization:
        # run generalization
        from generalization import run_generalization
        run_generalization()