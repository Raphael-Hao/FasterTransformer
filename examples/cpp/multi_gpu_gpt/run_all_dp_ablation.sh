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
        profiled_model=$2
        shift 2
        ;;

    --)
        shift
        break
        ;;

    *)
        echo "Unexpected option: $1 - this should not happen."
        exit 2
        ;;
    esac
done

bsz=(1 2 4 8 16 32 64 128)
input_len=10
output_lens=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
use_ffns=(0 1)
echo "bs,output_len,use_ffn,avg_duration" >results_"$used_gpus"_"$profiled_model".csv

# loop over bsz, output_len, use_ffn
for bs in "${bsz[@]}"; do
    for ol in "${output_lens[@]}"; do
        for use_ffn in "${use_ffns[@]}"; do
            bash parallel_ablation.sh "$bs" "$input_len" "$ol" "$use_ffn" "$used_gpus" >results_"$used_gpus"_"$profiled_model".txt
            # parse results.txt and write to results.csv
            # each line in results.csv is: [INFO] request_batch_size 8 beam_width 1 head_num 128 size_per_head 160 total_output_len 110 decoder_layers 1 vocab_size 51200 FT-CPP-decoding-beamsearch-time 191.30 ms
            # there will be $used_gpus lines of records, and we only need the average of the last-1 column
            durations=$(grep "FT-CPP-decoding-beamsearch-time" results_"$used_gpus"_"$profiled_model".txt | awk '{print $(NF-1)}')
            avg_duration=$(echo "$durations" | awk '{ total += $1; count++ } END { print total/count }')
            echo "$bs,$ol,$use_ffn,$avg_duration" >>results_"$used_gpus"_"$profiled_model".csv
            rm results_"$used_gpus"_"$profiled_model".txt
        done
    done
done
