import os
from utils import *
from mpl_plot import *

def load_mem_size(filename):
    a, p, g, m, v = None, None, None, None, None
    with open(filename, "rb") as f:
        for line in f.readlines():
            if (b"allocating" in line) and (b"MiB" in line) and (b"model parameters" in line):
                p = float(line.split()[1])
            if (b"allocating" in line) and (b"MiB" in line) and (b"parameter gradients" in line):
                g = float(line.split()[1])
            if (b"allocating" in line) and (b"MiB" in line) and (b"activations" in line):
                a = float(line.split()[1])
            if (b"allocating" in line) and (b"MiB" in line) and (b"AdamW optimizer state m" in line):
                m = float(line.split()[1])
            if (b"allocating" in line) and (b"MiB" in line) and (b"AdamW optimizer state v" in line):
                v = float(line.split()[1])
    return a, p, g, m, v

def load_latency(filename):
    data_0, data_1 = [], []
    with open(filename, "rb") as f:
        for line in f.readlines():
            if (b"step" in line) and (b"ms" in line):
                loc_0, loc_1 = 14, 16
                data_0.append(float(line.split()[loc_0]))
                data_1.append(float(line.split()[loc_1]))
    return np.mean(data_0[10:]), np.mean(data_1[10:])

gpt, batch_size = "124M", [4, 8, 16, 32, 64, 128]
#gpt, batch_size = "350M", [4, 8, 16, 32, 64]

result_dir = "../log_gpt2"

memory = []
for bs in batch_size:
    result_f = f"{result_dir}/{gpt}.bs{bs}.p16"
    a, p, g, m, v = load_mem_size(result_f)
    memory.append([a, p, g, m, v])
memory = np.asarray(memory) / 1024.0
print(memory)

colors = [crimson, blue, teal]
            
fig_f = f"figures/{gpt}_memory.pdf"
fig, axs = plt.subplots(figsize=(8, 6))
x = list(range(1, len(batch_size)+1))
axs.bar(x, memory[:, 0], color=colors[0], edgecolor=black)
axs.bar(x, memory[:, 1] + memory[:, 2], bottom=memory[:, 0], color=colors[1], edgecolor=black)
axs.bar(x, memory[:, 3] + memory[:, 4], bottom=memory[:, 0] + memory[:, 1] + memory[:, 2], color=colors[2], edgecolor=black)
axs.set_xticks(x, batch_size)
x_label = "Batch Size"
y_label = "Memory Allocation (GB)"
axs.set_xlabel(x_label)
axs.set_ylabel(y_label)
fig.tight_layout()
plt.savefig(fig_f)
print(fig_f)

latency = []
for bs in batch_size:
    l = []
    for mv in [False, True]:
        if mv is False:
            result_f = f"{result_dir}/{gpt}.bs{bs}.p16"
        else:
            result_f = f"{result_dir}/{gpt}.bs{bs}.p16.mv_offload"
        fwdbwd, adamw = load_latency(result_f)
        l.append([fwdbwd, adamw])
    latency.append(l)
latency = np.asarray(latency)
print("latency = ", latency)

colors = [grey, pink]

fig_f = f"figures/{gpt}_latency.pdf"
fig, axs = plt.subplots(figsize=(10, 6))
x = []
for i in range(len(batch_size)):
    x0, x1 = 3*i, 3*i+1
    x.append(x0+0.5)
    axs.bar(x0, latency[i][0][0], color=colors[0], edgecolor=black)
    axs.bar(x0, latency[i][0][1], bottom=latency[i][0][0], color=colors[1], edgecolor=black)

    axs.bar(x1, latency[i][1][0], color=colors[0], edgecolor=black)
    axs.bar(x1, latency[i][1][1], bottom=latency[i][1][0], color=colors[1], edgecolor=black)
#
#axs.bar(x, memory[:, 1] + memory[:, 2], bottom=memory[:, 0], color=colors[1], edgecolor=black)
#axs.bar(x, memory[:, 3] + memory[:, 4], bottom=memory[:, 0] + memory[:, 1] + memory[:, 2], color=colors[2], edgecolor=black)
axs.set_xticks(x, batch_size)
x_label = "Batch Size"
y_label = "Latency (ms)"
axs.set_xlabel(x_label)
axs.set_ylabel(y_label)
fig.tight_layout()
plt.savefig(fig_f)
print(fig_f)
        