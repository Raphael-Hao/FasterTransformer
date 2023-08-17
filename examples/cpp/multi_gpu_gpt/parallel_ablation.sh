#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

request_bs=$1
#input sequence length
input_len=$2
#output sequence length
output_len=$3
#use_ffn: 0 for false, 1 for true
int_use_ffn=$4
if [ "$int_use_ffn" -eq 0 ]
then
    use_ffn=false
else
    use_ffn=true
fi
used_gpus=$5

# generate start_ids.csv, $request_bs lines, each line has $input_len number, each number is a random number between 0 and 50256
python3 generate_start_ids.py --request_bs $request_bs --input_len $input_len
# modify parallel_ablation.ini to change the request_batch_size to $request_bs
sed -i "s/request_batch_size=.*/request_batch_size=$request_bs/g" parallel_ablation.ini
# change max_batch_size to $request_bs
sed -i "s/max_batch_size=.*/max_batch_size=$request_bs/g" parallel_ablation.ini
# modify parallel_ablation.ini to change the request_output_len to $output_len
sed -i "s/request_output_len=.*/request_output_len=$output_len/g" parallel_ablation.ini
# modify parallel_ablation.ini to change the use_ffn to $use_ffn
sed -i "s/use_ffn=.*/use_ffn=$use_ffn/g" parallel_ablation.ini
# modify parallel_ablation.ini to change the tensor_para_size to $used_gpus
sed -i "s/tensor_para_size=.*/tensor_para_size=$used_gpus/g" parallel_ablation.ini

# cd to build directory run parallel_ablation and cd back to examples/cpp/multi_gpu_gpt
cd ../../../build && mpirun -n "$used_gpus" ./bin/parallel_ablation && cd - || exit