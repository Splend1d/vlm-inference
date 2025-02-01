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
    "inference_deepseek-moe-16b-chat_cot_summurize", "inference_deepseek-vl2-small_cot_summurize", \
    "inference_deepseek-llm-7b-chat_cot_summurize", "inference_deepseek-vl-7b-chat_cot_summurize"
]
dataset_name = "cais/mmlu"


for exp_name in exp_names:
    n_corrects = defaultdict(int)
    n_total = defaultdict(int)
    if dataset_name == "cais/mmlu":
        configs = utils_MMLU.DATASET_CONFIG[dataset_name]["subsets"]
        for config in configs:
            
            logging.info("Processing "+str(config))
            result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{config}"
            os.makedirs(result_dir, exist_ok = True)
            
            
            for fname in os.listdir(result_dir):
                file_path = os.path.join(result_dir, fname)
                cat = fname[:-17]
                # logging.info("Running "+str(str_id))
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    selected_answer = data['final_answer']
                    
                except:
                    selected_answer = "NAN"
                
                label = data['label']
                
                if label in selected_answer:
                    n_corrects[cat] += 1
                else:
                    pass
                n_total[cat] += 1

    main_category_accuracies = defaultdict(list)
    for cat in n_total:
        
        main_category_accuracies[utils_MMLU.subcategories_to_main_categories[cat]].append(n_corrects[cat]/n_total[cat])
        #print(cat, n_corrects[cat]/n_total[cat])
    
    # Define column widths
    col1_width = 35  # Category
    col2_width = 16  # # Subcategories
    col3_width = 30  # Accuracy (Mean ± Std)

    # Print table header
    print(f"＊＊＊ {exp_name}＊＊＊")
    print("\n" + "=" * (col1_width + col2_width + col3_width + 6))
    print(f"| {'Category'.ljust(col1_width)} | {'# Subcategories'.ljust(col2_width)} | {'Accuracy (Mean ± Std)'.ljust(col3_width)} |")
    print("|" + "-" * (col1_width) + "|" + "-" * (col2_width) + "|" + "-" * (col3_width) + "|")

    # Print table rows
    main_cat_accs = []
    for main_cat, acc_list in main_category_accuracies.items():
        mean_acc = sum(acc_list) / len(acc_list)
        main_cat_accs.append(mean_acc)
        print(f"| {main_cat.ljust(col1_width)} | {str(len(acc_list)).ljust(col2_width)} | {f'{mean_acc * 100:.2f}% '.ljust(col3_width)} |")

    # Print table footer
    
    print("|" + "-" * (col1_width) + "|" + "-" * (col2_width) + "|" + "-" * (col3_width) + "|")
    
    print("Avg:", sum(main_cat_accs) / len(main_cat_accs))

    print("=" * (col1_width + col2_width + col3_width + 6) + "\n")

        
        

            
            
            
