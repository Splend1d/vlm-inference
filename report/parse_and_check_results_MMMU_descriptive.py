import os
import logging
import json
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

eval_name = "MMMU_local"
#"MMMU_descriptive"
exp_name = "inference_Qwen2-VL-72B-Instruct_cot_summurize"
#"Qwen2-VL-72B-Instruct"
#"QWQ-32B-Preview"
dataset_name = f"Splend1dchan/{eval_name}"

output_content = {}

n_corrects = defaultdict(int)
n_total = defaultdict(int)
all_accuracies = []

if dataset_name == f"Splend1dchan/{eval_name}":
    configs = get_dataset_config_names(f"data/{eval_name}")
    for config in configs:
        
        logging.info("Processing "+str(config))
        result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{config}"
        if not os.path.exists(result_dir):
            continue
        
        ds = load_dataset(f"data/{eval_name}",config, split = "validation")
        
        for data in ds:
            str_id = data["id"]
            # logging.info("Running "+str(str_id))
            file_path = os.path.join(result_dir, f"{str_id}.json")
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                selected_answer = data["final_answer"]
            except:
                selected_answer = "NA"
                os.remove(file_path)
            
            try:
                label = data["label"]
            except:
                label = ""
            

            if selected_answer in label:
                n_corrects[config] += 1
            else:
                pass
            if selected_answer != "NA":
                n_total[config] += 1

            # Extract the last text value from the JSON
        
        if n_total[config]:
            print(f"MMMU_descriptive-{config} Accuracy:", n_corrects[config]/n_total[config], n_corrects[config], n_total[config])
        
        if n_total[config] == 30:
            all_accuracies.append(n_corrects[config]/n_total[config])

print("Macro avg:", sum(all_accuracies)/len(all_accuracies))# "identical # samples in each category, easy sum"
            
            
