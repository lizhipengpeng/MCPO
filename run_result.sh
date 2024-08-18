llama3_path_list=( "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3_lora_70B-instruct-epoch3-trainandval-1000-awq" "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3-70B-instruct-epoch3-trainandval-4000-awq" "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3-70B-instruct-epoch3-trainandval-6000-awq" "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3-70B-instruct-epoch3-trainandval-10000-awq" "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3_lora_70B-instruct-epoch3-20000-awq" "/cbd-lizhipeng/yiying_model/ecellm_all_samples_awq")
qwen2_path_list=( "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-2000" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-4000" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-6000" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-10000" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-20000" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct-all")
phi3_path_list=( "/cbd-lizhipeng/zhewei_model/phi128" "/cbd-lizhipeng/zhewei_model/phi128-2000" "/cbd-lizhipeng/zhewei_model/phi128-1000" "/cbd-lizhipeng/zhewei_model/phi128-4000" "/cbd-lizhipeng/zhewei_model/phi128-6000" "/cbd-lizhipeng/zhewei_model/phi128-6100" "/cbd-lizhipeng/zhewei_model/phi128-20000" "/cbd-lizhipeng/zhewei_model/phi128-6100" "/cbd-lizhipeng/zhewei_model/phi128-10000")

test_dataset=( "/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide.json" "/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test.json")

instructions=( "self_cot_all" "self_cot_mcp" "little_model_cot" "little_model_cot_all" "generation_prompt" "similarity" "all")

base_model=( "/cbd-lizhipeng/llama-3-70b-instruct-awq" "/cbd-lizhipeng/yiying_model/qwen2-7b/Qwen2-7B-Instruct" "/cbd-lizhipeng/zhewei_model/phi128")



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/yiying_model/ecellm_all_samples_awq" --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide.json --instruction self_cot_mcp

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/llama-3-70b-instruct-awq" --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test.json --instruction self_cot_mcp
# mergekit-yaml /cbd-lizhipeng/mergekit/examples/gradient-slerp.yml /cbd-lizhipeng/merge_model/llama3-70b-130000_and_20000 --cuda
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path /cbd-lizhipeng/merge_model/llama3-70b-130000_and_20000 --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test.json --instruction None
# mergekit-yaml /cbd-lizhipeng/mergekit/examples/gradient-slerp-phi.yml /cbd-lizhipeng/merge_model/phi-3b-2000_and_6000 --cuda
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path /cbd-lizhipeng/yiying_model/ecellm_all_samples_awq --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide_test_RKQ_en.json --instruction similarity
# CUDA_VISIBLE_DEVICES=0,1,2,3 python local_evaluation.py --model_path /cbd-lizhipeng/merge_model/qwen2-7b-20000_and_130000 --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide.json --instruction None
# CUDA_VISIBLE_DEVICES=0,1,2,3 python local_evaluation.py --model_path /cbd-lizhipeng/merge_model/qwen2-7b-raw_and_20000 --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test.json --instruction None

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/yiying_model/ecellm_all_samples_awq" --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test_small_cot.json --instruction all

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/llama-3-70b-instruct-awq" --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ecellm_test_small_cot.json --instruction all


# bash /cbd-lizhipeng/mergekit/merge.sh


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/llama-3-70b-instruct-awq" --data_path /cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide_small_cot.json

# for model_path in ${llama3_path_list[@]}
# do
# echo $model_path
# if [[ $model_path =~ "Qwen" ]]
# then
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python local_evaluation.py --model_path $model_path --data_path "/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide.json"
# else
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path $model_path --data_path "/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/data/ec-guide.json"
# fi
# done


# for instruction in ${instructions[@]}
# do
# for dataset in ${test_dataset[@]}
# do
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python local_evaluation.py --model_path "/cbd-lizhipeng/llama-3-70b-instruct-awq" --data_path $dataset --instruction $instruction
# done
# done