#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

ARGUMENT_LIST=(
    "gpus:"
    "bs:"
    "input-len:"
    "output-len:"
    "model:"
    "use-ffn:"
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

    --bs)
        request_bs=$2
        shift 2
        ;;

    --input-len)
        input_len=$2
        shift 2
        ;;

    --output-len)
        output_len=$2
        shift 2
        ;;

    --model)
        model_name=$2
        shift 2
        ;;

    --use-ffn)
        int_use_ffn=$2
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

if [ "$int_use_ffn" -eq 0 ]; then
    use_ffn=false
else
    use_ffn=true
fi

# generate start_ids.csv, $request_bs lines, each line has $input_len number, each number is a random number between 0 and 50256
python3 generate_start_ids.py --request_bs "$request_bs" --input_len "$input_len" --used_gpus "$used_gpus"
# modify parallel_ablation.ini to change the model_name to $model_name
sed -i "s/model_name=.*/model_name=$model_name/g" parallel_ablation_"$used_gpus".ini
# modify parallel_ablation.ini to change the request_batch_size to $request_bs
sed -i "s/request_batch_size=.*/request_batch_size=$request_bs/g" parallel_ablation_"$used_gpus".ini
# change max_batch_size to $request_bs
sed -i "s/max_batch_size=.*/max_batch_size=$request_bs/g" parallel_ablation_"$used_gpus".ini
# modify parallel_ablation.ini to change the request_output_len to $output_len
sed -i "s/request_output_len=.*/request_output_len=$output_len/g" parallel_ablation_"$used_gpus".ini
# modify parallel_ablation.ini to change the use_ffn to $use_ffn
sed -i "s/use_ffn=.*/use_ffn=$use_ffn/g" parallel_ablation_"$used_gpus".ini
# modify parallel_ablation.ini to change the tensor_para_size to $used_gpus
sed -i "s/tensor_para_size=.*/tensor_para_size=$used_gpus/g" parallel_ablation_"$used_gpus".ini

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cd to build directory run parallel_ablation and cd back to examples/cpp/multi_gpu_gpt
# list all files in bin and echo them to a file
echo "Run parallel_ablation with $used_gpus GPUs"

cd "$SCRIPT_DIR"/../../../build &&
    mpirun -n "$used_gpus" parallel_ablation \
        "$SCRIPT_DIR"/examples/cpp/multi_gpu_gpt/parallel_ablation_"$used_gpus".ini \
        "$SCRIPT_DIR"/examples/cpp/multi_gpu_gpt/start_ids_"$used_gpus".csv && cd - || exit
