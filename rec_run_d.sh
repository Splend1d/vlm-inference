while true
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 data/generate_MMMU_to_descriptive.py
done
