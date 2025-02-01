import os
import logging
import json

from datasets import load_dataset, get_dataset_config_names

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

exp_name = "inference_qwenvl7b_cot_summurize"
dataset_name = "MMMU/MMMU"

output_content = {}
if dataset_name == "MMMU/MMMU":
    configs = get_dataset_config_names("MMMU/MMMU")
    for config in configs:
        
        logging.info("Processing "+str(config))
        result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{config}"
        os.makedirs(result_dir, exist_ok = True)
        
        ds = load_dataset("MMMU/MMMU",config, split = "validation")
        
        for data in ds:
            str_id = data["id"]
            # logging.info("Running "+str(str_id))
            file_path = os.path.join(result_dir, f"{str_id}.json")
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                selected_answer = data[-1]['content'][0]['text']
            except:
                selected_answer = "NA"
            
            output_content[str_id] = selected_answer

            # Extract the last text value from the JSON
        
        
    with open(os.path.join(f"results/{exp_name}/{dataset_name.split('/')[-1]}","results.json"), 'w') as file:
        json.dump(output_content, file, indent=4)
            
            
            
