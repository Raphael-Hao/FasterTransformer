#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

used_gpus=$1

bsz=(8 16 32 64 128 256)
input_len=10
output_lens=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800)
use_ffns=(0 1)
echo "bs,output_len,use_ffn,avg_duration" >results_"$used_gpus".csv

# loop over bsz, output_len, use_ffn
for bs in "${bsz[@]}"; do
    for ol in "${output_lens[@]}"; do
        for use_ffn in "${use_ffns[@]}"; do
            bash parallel_ablation.sh "$bs" "$input_len" "$ol" "$use_ffn" "$used_gpus" >results_"$used_gpus".txt
            # parse results.txt and write to results.csv
            # each line in results.csv is: [INFO] request_batch_size 8 beam_width 1 head_num 128 size_per_head 160 total_output_len 110 decoder_layers 1 vocab_size 51200 FT-CPP-decoding-beamsearch-time 191.30 ms
            # there will be $used_gpus lines of records, and we only need the average of the last-1 column
            durations=$(grep "FT-CPP-decoding-beamsearch-time" results.txt | awk '{print $(NF-1)}')
            avg_duration=$(echo "$durations" | awk '{ total += $1; count++ } END { print total/count }')
            echo "$bs,$ol,$use_ffn,$avg_duration" >>results_"$used_gpus".csv
            rm results_"$used_gpus".txt
        done
    done
done
