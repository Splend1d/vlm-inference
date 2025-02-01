import os
import logging
import json
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names

import utils_MMLU

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

exp_names = ["inference_Qwen2-1.5B-Instruct_cot_summurize", "inference_Qwen2-VL-2B-Instruct_cot_summarize", \
    "inference_Qwen2-7B-Instruct_cot_summurize", "inference_Qwen2-VL-7B-Instruct_cot_summarize", \
    "inference_deepseek-moe-16b-chat_cot_summurize", "inference_deepseek-vl2-small_cot_summurize"
]
dataset_name = "Idavidrein/gpqa"


for exp_name in exp_names:
    n_corrects = defaultdict(int)
    n_total = defaultdict(int)
    if dataset_name == "Idavidrein/gpqa":
        configs = ["gpqa_diamond"]
        for config in configs:
            
            logging.info("Processing "+str(config))
            result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{config}"
            os.makedirs(result_dir, exist_ok = True)
            
            
            for fname in os.listdir(result_dir):
                file_path = os.path.join(result_dir, fname)
                
                # logging.info("Running "+str(str_id))
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    selected_answer = data['final_answer']
                    
                except:
                    selected_answer = "NAN"
                
                label = data['label']
                
                if label in selected_answer:
                    n_corrects[config] += 1
                else:
                    pass
                n_total[config] += 1

    
    
            print(f"＊＊＊ {exp_name}＊＊＊")
            if n_total[config] == 0:
                print("Not evaluated yet")
            else:
                print("Avg:", n_corrects[config]/ n_total[config])
            print("=" *  30 + "\n")

        
        

            
            
            
