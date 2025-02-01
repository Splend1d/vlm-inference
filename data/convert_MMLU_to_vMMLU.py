from datasets import load_dataset, Dataset, DatasetDict
from datasets import get_dataset_config_names, load_dataset
import random

# Load MMLU dataset (all subjects)
mmlu = load_dataset("cais/mmlu", "all")["test"]

# Load MMMU dataset (all available configs)
subjects = get_dataset_config_names("MMMU/MMMU")
mmmu_images = []
for subject in subjects:
    mmmu = load_dataset("MMMU/MMMU", subject)  # Load all available configurations

    mmmu_images.extend(list(zip(mmmu["test"]["image_1"],mmmu["test"]["img_type"])))  # Adjust split if necessary
    print("done",subject)
    #break

# Function to add random image to each sample
def add_random_image(sample):
    sample["image"], sample["img_type"] = random.choice(mmmu_images)
    return sample

# Apply function to all samples
vMMLU = mmlu.map(add_random_image)

dataset_dict = DatasetDict({"test": vMMLU})
# Save the modified dataset to Hugging Face DatasetDict with a random subset name
dataset_name = f"Splend1dchan/MMLU_visual_noise"
dataset_dict.push_to_hub(dataset_name, "random_noise")
