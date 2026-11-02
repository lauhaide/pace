#!/usr/bin/bash

export PYTHONPATH=$PYTHONPATH:/mnt/data/algomo/codebase/


declare -A data=(
    ["hotpotqa"]="hotpotqa-test-orlhf")

QWEN_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

declare -A model_paths=(
    ["qwen"]="${QWEN_MODEL_PATH}"
)

declare -A model_prompt_type=(
    ["qwen"]="none"
)

prompt_types=("dataass")

methods_names=("datasel")


for model_type in "${!model_paths[@]}"; do
    readarray -t paths <<< "${model_paths[$model_type]}"
    
    # Traverse all model paths under this type
    for model_path in "${paths[@]}"; do
        # Skip empty lines
        [[ -z "$model_path" ]] && continue
        
        model_name=$(basename "$model_path")
        echo "Starting evaluation with model: $model_name (type: $model_type)"
        
        temp_config="/mnt/data/tmp/temp_config_${model_name}.yaml"
        cp "codebase/evaluation/src/datasel_config.yaml" "$temp_config"
        chmod 777 "$temp_config"
        
        sed -i "s|generator_model_path:.*|generator_model_path: \"${model_path//\//\\/}\"|" "$temp_config"
        sed -i "s|generator_model:.*|generator_model: \"${model_type}\"|" "$temp_config"

        
        
        echo "Current config for $model_name:"
        cat "$temp_config"

        for prompt_type in "${prompt_types[@]}"; do
            for dataset in "${!data[@]}"; do
                echo "Processing dataset: $dataset with model: $model_name"

                for method_name in "${methods_names[@]}"; do
                    echo "Processing method: $method_name with model: $model_name"

                    CUDA_VISIBLE_DEVICES=3 python -u codebase/evaluation/src/run_eval.py \
                        --config_path "$temp_config" \
                        --method_name "$method_name" \
                        --split "${data[$dataset]}" \
                        --dataset_name "$dataset" \
                        --prompt_type "$prompt_type" \
                        --prompt_template_type "${model_prompt_type[$model_type]}" \
                        --result_output_file "/mnt/data/mhqa/outputs/result_analysis/result.json"
                    
                    if [ $? -ne 0 ]; then
                        echo "Error processing $dataset with model $model_name"
                        #rm "$temp_config"
                        exit 1
                    fi
                done
            done
        done
        
        rm "$temp_config"
        echo "Completed evaluation with model: $model_name"
    done
done