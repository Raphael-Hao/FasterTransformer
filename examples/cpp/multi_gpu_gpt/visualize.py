# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

# %% prepare the data
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path.cwd() / "data"
figure_dir = Path.cwd() / "figures"
figure_dir.mkdir(exist_ok=True)

profiled_gpus = ["A100_SXM4_40GB", "Tesla_V100_SXM3_32GB"]
models = [
    "megatron_345M",
    "megatron_1.3B",
    "megatron_6.7B",
    "megatron_20B",
    "gpt_89B",
    "gpt_175B",
]

bsz = [8, 16, 32, 64, 128, 256]
seq_len_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
seq_len_2 = [100, 200, 300, 400, 500, 600, 700, 800]
gpu_nums = [2, 4, 8]

tp_duration_datas = {}
dp_duration_datas = {}
for gpu in profiled_gpus:
    tp_duration_datas[gpu] = {}
    dp_duration_datas[gpu] = {}
    for model in models:
        tp_duration_datas[gpu][model] = {}
        dp_duration_datas[gpu][model] = {}
        for gpu_num in gpu_nums:
            if gpu == "A100_SXM4_40GB" and gpu_num == 8:
                continue
            data_path = data_dir / gpu / str(gpu_num) / f"{model}.csv"
            data = pd.read_csv(data_path, header=0)
            duration_all = data[data.use_ffn == 1].reset_index(drop=True)
            duration_without_ffn = data[data.use_ffn == 0].reset_index(drop=True)
            duration_all["attn"] = duration_without_ffn["avg_duration"]
            duration_all["ffn"] = (
                duration_all["avg_duration"] - duration_without_ffn["avg_duration"]
            )
            duration_all = duration_all.drop(["use_ffn"], axis=1)
            tp_duration_datas[gpu][model][gpu_num] = duration_all

        dp_data_path = data_dir / gpu / "1" / f"{model}.csv"
        dp_duration = pd.read_csv(dp_data_path, header=0)
        dp_duration_all = dp_duration[dp_duration.use_ffn == 1].reset_index(drop=True)
        dp_duration_without_ffn = dp_duration[dp_duration.use_ffn == 0].reset_index(
            drop=True
        )
        dp_duration_all["attn"] = dp_duration_without_ffn["avg_duration"]
        dp_duration_all["ffn"] = (
            dp_duration_all["avg_duration"] - dp_duration_without_ffn["avg_duration"]
        )
        dp_duration_all = dp_duration_all.drop(["use_ffn"], axis=1)
        dp_duration_datas[gpu][model] = dp_duration_all

# %% prepare the plot environment
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("hatch", linewidth=8)
selected_seq_lens = [10, 60, 100, 400, 800]

# %% plot the latency increase with the number of bs under different number of gpus and different number of seq_len
for gpu in profiled_gpus:
    for model in models:
        if gpu == "A100_SXM4_40GB":
            gpu_nums = [2, 4]
        else:
            gpu_nums = [2, 4, 8]
        total_rows = len(gpu_nums)
        total_cols = len(selected_seq_lens)
        figure, axes = plt.subplots(
            total_rows, total_cols, figsize=(total_cols * 6, total_rows * 6)
        )
        for i, gpu_num in enumerate(gpu_nums):
            for j, seq_len in enumerate(selected_seq_lens):
                duration_data = tp_duration_datas[gpu][model][gpu_num][
                    (tp_duration_datas[gpu][model][gpu_num].output_len == seq_len)
                ]
                duration_ddp = dp_duration_datas[gpu][model][
                    dp_duration_datas[gpu][model].output_len == seq_len
                ]
                duration_dp_data = []
                for bs in bsz:
                    duration_dp_data.append(
                        duration_ddp[duration_ddp.bs == bs / gpu_num][
                            ["attn", "ffn", "avg_duration"]
                        ].values[0]
                    )
                duration_dp_data = np.array(duration_dp_data)
                axes[i, j].plot(
                    duration_data.bs,
                    duration_dp_data[:, 0],
                    label="attn_ddp",
                    linestyle="--",
                )
                axes[i, j].plot(
                    duration_data.bs,
                    duration_dp_data[:, 1],
                    label="ffn_ddp",
                    linestyle="--",
                )
                axes[i, j].plot(
                    duration_data.bs,
                    duration_dp_data[:, 2],
                    label="total_ddp",
                    linestyle="--",
                )
                axes[i, j].plot(duration_data.bs, duration_data.attn, label="attn")
                axes[i, j].plot(duration_data.bs, duration_data.ffn, label="ffn")
                axes[i, j].plot(
                    duration_data.bs, duration_data.avg_duration, label="total"
                )
                axes[i, j].set_title(f"gpu_num = {gpu_num}, seq_len = {seq_len}")
                axes[i, j].legend()
            axes[i, 0].set_ylabel("latency (ms)")
        figure_path = figure_dir / gpu / model / "latency_increase_with_bs.png"
        figure_path.parent.mkdir(exist_ok=True, parents=True)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %% plot the latency increase with the number of seq_len under different number of gpus and different number of bs
for model in models:
    for gpu in profiled_gpus:
        if gpu == "A100_SXM4_40GB":
            gpu_nums = [2, 4]
        else:
            gpu_nums = [2, 4, 8]
        total_rows = len(gpu_nums)
        total_cols = len(bsz)
        figure, axes = plt.subplots(
            total_rows, total_cols, figsize=(6 * total_cols, 6 * total_rows)
        )
        for i, gpu_num in enumerate(gpu_nums):
            for j, bs in enumerate(bsz):
                duration_data = tp_duration_datas[gpu][model][gpu_num][
                    (tp_duration_datas[gpu][model][gpu_num].bs == bs)
                ]
                duration_ddp = dp_duration_datas[gpu][model][
                    dp_duration_datas[gpu][model].bs == bs / gpu_num
                ]

                axes[i, j].plot(
                    duration_data.output_len,
                    duration_ddp.attn,
                    label="attn_ddp",
                    linestyle="--",
                )
                axes[i, j].plot(
                    duration_data.output_len,
                    duration_ddp.ffn,
                    label="ffn_ddp",
                    linestyle="--",
                )
                axes[i, j].plot(
                    duration_data.output_len,
                    duration_ddp.avg_duration,
                    label="total_ddp",
                    linestyle="--",
                )

                axes[i, j].plot(
                    duration_data.output_len, duration_data.attn, label="attn"
                )
                axes[i, j].plot(
                    duration_data.output_len, duration_data.ffn, label="ffn"
                )
                axes[i, j].plot(
                    duration_data.output_len, duration_data.avg_duration, label="total"
                )
                axes[i, j].set_title(f"gpu_num = {gpu_num}, bs = {bs}")
                axes[i, j].legend()
            axes[i, 0].set_ylabel("latency (ms)")
        figure_path = figure_dir / gpu / model / "latency_increase_with_seq_len.png"
        figure_path.parent.mkdir(exist_ok=True, parents=True)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %% plot the latency increase with the number of gpus under different number of bs and different number of seq_len
for model in models:
    for gpu in profiled_gpus:
        if gpu == "A100_SXM4_40GB":
            gpu_nums = [2, 4]
        else:
            gpu_nums = [2, 4, 8]
        total_rows = len(bsz)
        total_cols = len(selected_seq_lens)
        figure, axes = plt.subplots(
            total_rows, total_cols, figsize=(total_cols * 6, total_rows * 6)
        )
        for i, bs in enumerate(bsz):
            for j, seq_len in enumerate(selected_seq_lens):
                duration_data = []
                for gpu_num in gpu_nums:
                    duration_data.append(
                        tp_duration_datas[gpu][model][gpu_num][
                            (tp_duration_datas[gpu][model][gpu_num].bs == bs)
                            & (tp_duration_datas[gpu][model][gpu_num].output_len == seq_len)
                        ][["attn", "ffn", "avg_duration"]].values[0]
                    )
                duration_data = np.array(duration_data)

                duration_ddp = dp_duration_datas[gpu][model][
                    (dp_duration_datas[gpu][model].output_len == seq_len)
                ]
                duration_dp_data = []
                for gpu_num in gpu_nums:
                    duration_dp_data.append(
                        duration_ddp[duration_ddp.bs == bs / gpu_num][
                            ["attn", "ffn", "avg_duration"]
                        ].values[0]
                    )
                duration_dp_data = np.array(duration_dp_data)

                axes[i, j].plot(
                    gpu_nums, duration_dp_data[:, 0], label="attn_ddp", linestyle="--"
                )
                axes[i, j].plot(
                    gpu_nums, duration_dp_data[:, 1], label="ffn_ddp", linestyle="--"
                )
                axes[i, j].plot(
                    gpu_nums, duration_dp_data[:, 2], label="total_ddp", linestyle="--"
                )

                axes[i, j].plot(gpu_nums, duration_data[:, 0], label="attn")
                axes[i, j].plot(gpu_nums, duration_data[:, 1], label="ffn")
                axes[i, j].plot(gpu_nums, duration_data[:, 2], label="total")
                axes[i, j].set_title(f"bs = {bs}, seq_len = {seq_len}")
                axes[i, j].legend()
            axes[i, 0].set_ylabel("latency (ms)")
        figure_path = figure_dir / gpu / model / "latency_increase_with_gpu_num.png"
        figure_path.parent.mkdir(exist_ok=True, parents=True)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %%
