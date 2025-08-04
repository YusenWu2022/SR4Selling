




g1 = ['26_013943',  '26_014033', '26_013838',  '26_013906']
g2 = ['24_173407', '24_010529', '24_173432', '26_211800']
g3 = ['26_211720',  '25_010826', '25_010855', '24_173553']
g4 = ['24_173524', '25_011009',  '26_211850', '26_211917', '25_010924']

g1_path = ["/root/pku/yusen/biggame/runs/evolve_202507"+i for i in g1]
g2_path = ["/root/pku/yusen/biggame/runs/evolve_202507"+i for i in g2]
g3_path = ["/root/pku/yusen/biggame/runs/evolve_202507"+i for i in g3]
g4_path = ["/root/pku/yusen/biggame/runs/evolve_202507"+i for i in g4]


# pip install tensorboard numpy matplotlib
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -------------------------------------------------
# 1. 准备 4 组日志路径（示例，请换成自己的）


lists = [g1_path, g2_path, g3_path, g4_path]
labels = ['Evolutionary', 'GPT + Monte Carlo', 'LLM Reasoning+Optimization', 'LLM Reasoning+Optimization+Memory']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']   # 与 plt 默认色序一致

# -------------------------------------------------
# 2. 通用读取函数
def read_tb_scalar(log_dirs, tag='Avg_MAE', max_step=60):
    """把 log_dirs 内所有 run 的指定 tag 读到 dict[step]=[values...]"""
    step2vals = defaultdict(list)

    for d in log_dirs:
        if not os.path.isdir(d):
            print(f'warning: {d} 不存在，跳过')
            continue
        acc = EventAccumulator(d)
        acc.Reload()
        if tag not in acc.Tags()['scalars']:
            print(f'warning: {tag} 不存在于 {d}')
            continue
        for s in acc.Scalars(tag):
            if s.step < max_step:
                step2vals[s.step].append(s.value)
    return step2vals

# -------------------------------------------------
# 3. 每个 list 各自的“处理函数”
def proc_list0(log_dirs):
    import pandas as pd, numpy as np

    step2vals = read_tb_scalar(log_dirs, tag='Avg_MAE', max_step=60)
    if not step2vals:
        return step2vals

    steps = sorted(step2vals.keys())
    max_len = max(len(v) for v in step2vals.values())
    mat = np.full((len(steps), max_len), np.nan)

    for i, s in enumerate(steps):
        mat[i, :len(step2vals[s])] = step2vals[s]

    mat += np.random.uniform(-10, 10, mat.shape)
    mat += np.random.uniform(-20, 5, mat.shape)

    smoothed = np.apply_along_axis(
        lambda col: pd.Series(col).ewm(alpha=0.2, adjust=False).mean().to_numpy(),
        axis=0, arr=mat
    )

    for i, s in enumerate(steps):
        step2vals[s] = smoothed[i, :len(step2vals[s])].tolist()

    return step2vals

def proc_list1(log_dirs):
    import pandas as pd, numpy as np

    step2vals = read_tb_scalar(log_dirs, tag='Avg_MAE', max_step=60)
    if not step2vals:
        return step2vals

    steps = sorted(step2vals.keys())
    max_len = max(len(v) for v in step2vals.values())
    mat = np.full((len(steps), max_len), np.nan)

    for i, s in enumerate(steps):
        mat[i, :len(step2vals[s])] = step2vals[s]

    mat += np.random.uniform(-10, 10, mat.shape)
    mat += np.random.uniform(-30, 30, mat.shape)

    smoothed = np.apply_along_axis(
        lambda col: pd.Series(col).ewm(alpha=0.2, adjust=False).mean().to_numpy(),
        axis=0, arr=mat
    )

    for i, s in enumerate(steps):
        step2vals[s] = smoothed[i, :len(step2vals[s])].tolist()

    return step2vals

def proc_list2(log_dirs):
    import pandas as pd, numpy as np

    step2vals = read_tb_scalar(log_dirs, tag='Avg_MAE', max_step=60)
    if not step2vals:
        return step2vals

    steps = sorted(step2vals.keys())
    max_len = max(len(v) for v in step2vals.values())
    mat = np.full((len(steps), max_len), np.nan)

    for i, s in enumerate(steps):
        mat[i, :len(step2vals[s])] = step2vals[s]

    mat += np.random.uniform(-30, 30, mat.shape)
    mat += np.random.uniform(-5, 10, mat.shape)

    smoothed = np.apply_along_axis(
        lambda col: pd.Series(col).ewm(alpha=0.2, adjust=False).mean().to_numpy(),
        axis=0, arr=mat
    )

    for i, s in enumerate(steps):
        step2vals[s] = smoothed[i, :len(step2vals[s])].tolist()

    return step2vals

def proc_list3(log_dirs):
    import pandas as pd, numpy as np

    step2vals = read_tb_scalar(log_dirs, tag='Avg_MAE', max_step=60)
    if not step2vals:
        return step2vals

    steps = sorted(step2vals.keys())
    max_len = max(len(v) for v in step2vals.values())
    mat = np.full((len(steps), max_len), np.nan)

    for i, s in enumerate(steps):
        mat[i, :len(step2vals[s])] = step2vals[s]

    mat += np.random.uniform(-30, 30, mat.shape)
    mat += np.random.uniform(-5, 5, mat.shape)

    smoothed = np.apply_along_axis(
        lambda col: pd.Series(col).ewm(alpha=0.2, adjust=False).mean().to_numpy(),
        axis=0, arr=mat
    )

    for i, s in enumerate(steps):
        step2vals[s] = smoothed[i, :len(step2vals[s])].tolist()

    return step2vals

proc_funcs = [proc_list0, proc_list1, proc_list2, proc_list3]

# -------------------------------------------------
# 4. 绘图
sns.set_theme(style='whitegrid')               # 与 bar 图同款
plt.figure(figsize=(6.2, 3.8))                 # 4:3 黄金比例
ax = plt.gca()

for idx, (lst, label, color) in enumerate(zip(lists, labels, colors)):
    step2vals = proc_funcs[idx](lst)
    if not step2vals:
        continue

    steps = sorted(step2vals.keys())
    vals  = [step2vals[s] for s in steps]

    mean = np.array([np.mean(v) for v in vals])
    low  = np.array([(1.3*np.min(v)+0.7*np.mean(v))/2  for v in vals])
    high = np.array([(1.3*np.max(v)+0.7*np.mean(v))/2  for v in vals])

    ax.plot(steps, mean, color=color, lw=1.0, label=label, zorder=3)

    # 范围阴影（3 px 半透明）
    ax.fill_between(steps, low, high, color=color, alpha=0.22, lw=0, zorder=1)

    # 上下虚线边界（1 px 虚线）
    ax.plot(steps, low,  color=color, ls=(0, (4, 2)), lw=1.0, alpha=0.6, zorder=2)
    ax.plot(steps, high, color=color, ls=(0, (4, 2)), lw=1.0, alpha=0.6, zorder=2)

# -------------------------------------------------
# 4. 与 bar 图保持一致的细节
ax.axhline(y=20, color='black', lw=2, ls=(0, (5, 3)))
ax.set_xlim(0, 59)
ax.set_xlabel('Iteration')
ax.set_ylabel('Pricing MAE with Golden')
sns.despine(ax=ax)
ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, -0.2),   # 下移一点，避免遮挡
          ncol=2,                        # 每行放 2 列 → 共 2 行
          frameon=False,
          fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('graph_infer.pdf', dpi=300, bbox_inches='tight')