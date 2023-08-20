#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

ARGUMENT_LIST=(
    "gpus:"  # specify how many gpus to use
    "model:" # specify which model to profile
)

# read arguments
opts=$(
    getopt \
        --longoptions "$(printf "%s," "${ARGUMENT_LIST[@]}")" \
        --name "$(basename "$0")" \
        --options "" \
        -- "$@"
)

eval set -- "$opts"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --gpus)
        used_gpus=$2
        shift 2
        ;;

    --model)
        selected_models=$2
        shift 2
        ;;

    --)
        shift
        break
        ;;

    *)
        echo "Usage: $0 --gpus <used_gpus> --model <selected_models>"
        echo "Example: $0 --gpus 8 --model megatron_345M,megatron_1.3B"
        echo "slected_models should be one or more of the following:"
        echo "megatron_345M,megatron_1.3B,megatron_6.7B,megatron_20B,gpt_89B,gpt_175B"
        echo "multiple models should be separated by comma"
        exit 2
        ;;
    esac
done

available_models=(
    megatron_345M
    megatron_1.3B
    megatron_6.7B
    megatron_20B
    gpt_89B
    gpt_175B
)
# parse selected_models into array, should be like "megatron_345M,megatron_1.3B"
IFS=',' read -ra selected_models <<<"$selected_models"
# check if selected_models is included in available_models
for model in "${selected_models[@]}"; do
    is_valid_model=false
    for available_model in "${available_models[@]}"; do
        if [ "$model" == "$available_model" ]; then
            is_valid_model=true
            break
        fi
    done
    if [ "$is_valid_model" == false ]; then
        echo "Invalid model: $model - this should not happen."
        exit 2
    fi
done

bsz=(8 16 32 64 128 256)
input_len=10
output_lens=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
use_ffns=(0 1)

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
gpu_name=${gpu_name// /_}
gpu_name=${gpu_name//-/_}
gpu_name=${gpu_name//NVIDIA_/}

# loop over selected_models bsz, output_len, use_ffn
for model in "${selected_models[@]}"; do
    echo "bs,output_len,use_ffn,avg_duration" >results_"$used_gpus"_"$model"_"$gpu_name".csv
    for bs in "${bsz[@]}"; do
        for ol in "${output_lens[@]}"; do
            for use_ffn in "${use_ffns[@]}"; do
                bash parallel_ablation.sh --gpus "$used_gpus" --bs "$bs" \
                    --input-len "$input_len" --output-len "$ol" \
                    --model "$model" --use-ffn "$use_ffn" \
                    >results_"$used_gpus"_"$model"_"$gpu_name".txt
                # parse results.txt and write to results.csv
                # each line in results.csv is: [INFO] request_batch_size 8 beam_width 1 head_num 128 size_per_head 160 total_output_len 110 decoder_layers 1 vocab_size 51200 FT-CPP-decoding-beamsearch-time 191.30 ms
                # there will be $used_gpus lines of records, and we only need the average of the last-1 column
                durations=$(grep "FT-CPP-decoding-beamsearch-time" results_"$used_gpus"_"$model"_"$gpu_name".txt | awk '{print $(NF-1)}')
                avg_duration=$(echo "$durations" | awk '{ total += $1; count++ } END { print total/count }')
                echo "$bs,$ol,$use_ffn,$avg_duration" >>results_"$used_gpus"_"$model"_"$gpu_name".csv
                rm results_"$used_gpus"_"$model"_"$gpu_name".txt
            done
        done
    done
done
