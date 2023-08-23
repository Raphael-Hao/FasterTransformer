# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui

# %% prepare the data
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path.cwd()
figure_dir = Path.cwd() / "figures"
figure_dir.mkdir(exist_ok=True)

gpu_nums = [2, 4, 8]
bsz = [8, 16, 32, 64, 128, 256]
seq_len_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
seq_len_2 = [100, 200, 300, 400, 500, 600, 700, 800]
models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
gpus = ["1", "2", "4", "8"]
data_filenames = {
    2: "results_2.csv",
    4: "results_4.csv",
    8: "results_8.csv",
}
duration_datas = {}
for gpu_num in gpu_nums:
    data_filename = data_filenames[gpu_num]
    data = pd.read_csv(data_dir / data_filename, header=0)
    duration_all = data[data.use_ffn == 1].reset_index(drop=True)
    duration_without_ffn = data[data.use_ffn == 0].reset_index(drop=True)
    duration_all["attn"] = duration_without_ffn["avg_duration"]
    duration_all["ffn"] = (
        duration_all["avg_duration"] - duration_without_ffn["avg_duration"]
    )
    duration_all = duration_all.drop(["use_ffn"], axis=1)
    duration_datas[gpu_num] = duration_all

ddp_duration = pd.read_csv(data_dir / "results_1.csv", header=0)
ddp_duration_datas = ddp_duration[ddp_duration.use_ffn == 1].reset_index(drop=True)
ddp_duration_without_ffn = ddp_duration[ddp_duration.use_ffn == 0].reset_index(
    drop=True
)
ddp_duration_datas["attn"] = ddp_duration_without_ffn["avg_duration"]
ddp_duration_datas["ffn"] = (
    ddp_duration_datas["avg_duration"] - ddp_duration_without_ffn["avg_duration"]
)
ddp_duration_datas = ddp_duration_datas.drop(["use_ffn"], axis=1)


# %% prepare the plot environment
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("hatch", linewidth=8)

# %% plot the latency increase with the number of bs under different number of gpus and different number of seq_len
selected_seq_lens = [10, 60, 100, 400, 800]
figure, axes = plt.subplots(3, 5, figsize=(30, 10))
for i, gpu_num in enumerate(gpu_nums):
    for j, seq_len in enumerate(selected_seq_lens):
        duration_data = duration_datas[gpu_num][
            (duration_datas[gpu_num].output_len == seq_len)
        ]
        duration_ddp = ddp_duration_datas[ddp_duration_datas.output_len == seq_len]
        duration_ddp_data = []
        for bs in bsz:
            duration_ddp_data.append(
                duration_ddp[duration_ddp.bs == bs / gpu_num][
                    ["attn", "ffn", "avg_duration"]
                ].values[0]
            )
        duration_ddp_data = np.array(duration_ddp_data)
        axes[i, j].plot(
            duration_data.bs, duration_ddp_data[:, 0], label="attn_ddp", linestyle="--"
        )
        axes[i, j].plot(
            duration_data.bs, duration_ddp_data[:, 1], label="ffn_ddp", linestyle="--"
        )
        axes[i, j].plot(
            duration_data.bs, duration_ddp_data[:, 2], label="total_ddp", linestyle="--"
        )
        axes[i, j].plot(duration_data.bs, duration_data.attn, label="attn")
        axes[i, j].plot(duration_data.bs, duration_data.ffn, label="ffn")
        axes[i, j].plot(duration_data.bs, duration_data.avg_duration, label="total")
        axes[i, j].set_title(f"gpu_num = {gpu_num}, seq_len = {seq_len}")
        axes[i, j].legend()
    axes[i, 0].set_ylabel("latency (ms)")
figure_path = figure_dir / "latency_increase_with_bs.png"
figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %% plot the latency increase with the number of seq_len under different number of gpus and different number of bs
figure, axes = plt.subplots(3, 6, figsize=(30, 10))
for i, gpu_num in enumerate(gpu_nums):
    for j, bs in enumerate(bsz):
        duration_data = duration_datas[gpu_num][(duration_datas[gpu_num].bs == bs)]
        duration_ddp = ddp_duration_datas[ddp_duration_datas.bs == bs / gpu_num]

        axes[i, j].plot(
            duration_data.output_len,
            duration_ddp.attn,
            label="attn_ddp",
            linestyle="--",
        )
        axes[i, j].plot(
            duration_data.output_len, duration_ddp.ffn, label="ffn_ddp", linestyle="--"
        )
        axes[i, j].plot(
            duration_data.output_len,
            duration_ddp.avg_duration,
            label="total_ddp",
            linestyle="--",
        )

        axes[i, j].plot(duration_data.output_len, duration_data.attn, label="attn")
        axes[i, j].plot(duration_data.output_len, duration_data.ffn, label="ffn")
        axes[i, j].plot(
            duration_data.output_len, duration_data.avg_duration, label="total"
        )
        axes[i, j].set_title(f"gpu_num = {gpu_num}, bs = {bs}")
        axes[i, j].legend()
    axes[i, 0].set_ylabel("latency (ms)")
figure_path = figure_dir / "latency_increase_with_seq_len.png"
figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %% plot the latency increase with the number of gpus under different number of bs and different number of seq_len
figure, axes = plt.subplots(6, 5, figsize=(30, 20))
for i, bs in enumerate(bsz):
    for j, seq_len in enumerate(selected_seq_lens):
        duration_data = []
        for gpu_num in gpu_nums:
            duration_data.append(
                duration_datas[gpu_num][
                    (duration_datas[gpu_num].bs == bs)
                    & (duration_datas[gpu_num].output_len == seq_len)
                ][["attn", "ffn", "avg_duration"]].values[0]
            )
        duration_data = np.array(duration_data)

        duration_ddp = ddp_duration_datas[(ddp_duration_datas.output_len == seq_len)]
        duration_ddp_data = []
        for gpu_num in gpu_nums:
            duration_ddp_data.append(
                duration_ddp[duration_ddp.bs == bs / gpu_num][
                    ["attn", "ffn", "avg_duration"]
                ].values[0]
            )
        duration_ddp_data = np.array(duration_ddp_data)

        axes[i, j].plot(gpu_nums, duration_ddp_data[:, 0], label="attn_ddp", linestyle="--")
        axes[i, j].plot(gpu_nums, duration_ddp_data[:, 1], label="ffn_ddp", linestyle="--")
        axes[i, j].plot(gpu_nums, duration_ddp_data[:, 2], label="total_ddp", linestyle="--")

        axes[i, j].plot(gpu_nums, duration_data[:, 0], label="attn")
        axes[i, j].plot(gpu_nums, duration_data[:, 1], label="ffn")
        axes[i, j].plot(gpu_nums, duration_data[:, 2], label="total")
        axes[i, j].set_title(f"bs = {bs}, seq_len = {seq_len}")
        axes[i, j].legend()
    axes[i, 0].set_ylabel("latency (ms)")
figure_path = figure_dir / "latency_increase_with_gpu_num.png"
figure.savefig(figure_path, dpi=300, bbox_inches="tight")

# %%
