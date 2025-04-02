from itertools import islice

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import FuncNorm
from mpl_toolkits.axes_grid1 import ImageGrid


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

def try_to_convert(v):
    if v == "False":
        return False
    if v == "True":
        return True
    return int(v)

def get_from_paste(filename):
    text = open(filename, "rt").read()
    headers = []
    data = []
    for config, value in batched(text.splitlines(), 2):
        config_elems = config.split(",")
        if not headers:
            headers = [e.partition("=")[0] for e in config_elems]
        data.append((*(try_to_convert(e.partition("=")[-1]) for e in config_elems), float(value)))
    return pd.DataFrame(data, columns=headers + ["latency"])

old_latencies = get_from_paste(...)
new_latencies = get_from_paste(...)

ratios = pd.merge(new_latencies, old_latencies, how="left", left_on=["m", "n", "k"], right_on=["m", "n", "k"], suffixes=("_new", "_old"))
ratios = ratios.assign(ratio=ratios.latency_old / ratios.latency_new)

fig = plt.figure(figsize=(40.0, 10.0))
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(1, 4),
    axes_pad=0.5,
    share_all=True,
    cbar_location="right",
    cbar_mode="single",
    cbar_size="7%",
    cbar_pad=0.15,
)

log_amax = np.max(np.abs(np.log(ratios.ratio.to_numpy())))

for K, ax in zip([1024, 2048, 4096, 8192], grid):
    pivoted = ratios[(ratios.k == K)].pivot_table(index="m", columns="n", values="ratio")
    im = ax.imshow(np.log(pivoted.to_numpy()), origin="lower", vmin=-log_amax, vmax=log_amax, cmap="PiYG")
    m_vals, n_vals = pivoted.axes
    ax.set_xticks(np.arange(len(n_vals)), labels=[f"N={i}" for i in n_vals.values], fontsize=12)
    ax.set_yticks(np.arange(len(m_vals)), labels=[f"M={i}" for i in m_vals.values], fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.grid(False)
    ax.set_title(f"K={K}", fontsize=20)

norm = FuncNorm((lambda x: np.log(x), lambda x: np.exp(x)), np.exp(-log_amax), np.exp(log_amax))
ax.cax.colorbar(ScalarMappable(norm=norm, cmap="PiYG"))
plt.show()

counts, bins = np.histogram(np.log(ratios.ratio.to_numpy()), bins=500)
plt.stairs(counts, np.exp(bins), fill=True)
plt.xscale("function", functions=(lambda x: np.log(x), lambda x: np.exp(x)))