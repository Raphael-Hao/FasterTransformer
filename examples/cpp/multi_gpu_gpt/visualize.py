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

gpu_nums = [2, 4, 8]
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
    duration_all["ffn"] = duration_all["avg_duration"] - duration_without_ffn["avg_duration"]
    duration_all = duration_all.drop(["use_ffn"], axis=1)
    print(duration_all)
# prepare the plot environment


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rc("hatch", linewidth=8)
fig_dir = "../figure/"
fig_filename = "multi_gpu_gpt.pdf"

# %% plot the latency increase with the number of bs under different number of gpus and different number of seq_len


# %% plot the latency increase with the number of seq_len under different number of gpus and different number of bs


# %% plot the latency increase with the number of gpus under different number of bs and different number of seq_len
