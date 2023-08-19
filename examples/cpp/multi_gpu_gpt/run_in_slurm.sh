#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

# make sure bashrc is loaded
PS1="$ "
# shellcheck source=/dev/null
source /lustre/home/acct-seecq/seecq/whcui/root_bashrc

ARGUMENT_LIST=(
    "parallel:"
    "model:"
    "gpus:"
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
    --parallel)
        parallel_mode=$2
        shift 2
        ;;

    --model)
        selected_models=$2
        shift 1
        ;;

    --gpus)
        used_gpus=$2
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

if [[ "${parallel_mode}" == tp ]]; then
    # check if used_gpus is set
    if [[ -z "${used_gpus}" ]]; then
        echo "used_gpus is not set!"
        exit 2
    fi
    bash run_all_tp_ablation.sh --gpus "${used_gpus}" --model "${selected_models}"
elif [[ "${parallel_mode}" == dp ]]; then
    bash run_all_dp_ablation.sh --model "${selected_models}"
else
    echo "Unsupport parallel mode: ${parallel_mode}"
    exit 2
fi
