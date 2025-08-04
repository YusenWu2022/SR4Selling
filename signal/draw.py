# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. 原始数据
# data = {
#     "Algorithm": ["Original", "Style", "DirectPrompt", "ConvTransfer", "TemplateOnly", "StructuredRewritten"],
#     "Style Consistency (X)": [4.04, 8.10, 6.42, 7.45, 7.62, 7.39],
#     "Content Preservation (Y)": [7.70, 3.85, 7.34, 6.22, 5.91, 7.04],
#     "Expression Quality (Z)": [5.83, 6.20, 6.34, 6.08, 6.32, 6.26],
#     "Average": [None, None, 6.70, 6.58, 6.62, 6.90]
# }

# df = pd.DataFrame(data)
# df_long = (
#     df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
#       .dropna(subset=["Score"])
# )

# # 2. ★★★ 手动指定要加粗的 (算法, 指标) 对 ★★★
# #    想加粗就往里加元组；不想加粗就留空列表 []
# BOLD_CELLS = {
#     ("Original",  "Content Preservation (Y)"),
#     ("Style",  "Style Consistency (X)"),
#     ("DirectPrompt",  "Expression Quality (Z)"),
#     ("StructuredRewritten", "Average"),   # 示例：可再随意添加
# }

# # 3. 打标记
# df_long["IsBold"] = df_long.apply(
#     lambda r: (r["Algorithm"], r["Metric"]) in BOLD_CELLS,
#     axis=1
# )

# # 4. 绘图
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(11, 5))
# ax = sns.barplot(
#     data=df_long,
#     x="Algorithm",
#     y="Score",
#     hue="Metric",
#     palette="Set2",
#     width=0.75
# )

# # 5. 顶部数值标注（仅手动指定的加粗）
# for bar, is_bold in zip(ax.patches,
#                         df_long["IsBold"]):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2,
#             height + 0.08,
#             f"{height:.2f}",
#             ha='center', va='bottom',
#             fontsize=9,
#             weight='bold' if is_bold else 'normal')

# # 6. 图例放在图下方外部
# ax.legend(title=None,
#           loc='upper center',
#           bbox_to_anchor=(0.5, -0.25),
#           ncol=len(df_long["Metric"].unique()),
#           frameon=False)

# # 7. 其他美化
# # plt.title("Evaluation of Style Transfer Methods", fontsize=14)
# plt.xlabel("Algorithm", fontsize=12)
# plt.ylabel("Score", fontsize=12)
# plt.xticks(rotation=20)
# sns.despine()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.25)

# # 8. 保存/展示
# plt.savefig("cluster_bar_manual_bold.pdf", dpi=300, bbox_inches='tight')







# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. 原始数据
# data = {
#     "Model": ["llama2-7b", "baichuan2-7b", "qwen2.5-7b",
#               "llama3-8b", "mistral-7b", "GPT4", "human"],
#     r"$r_1$": [0.472, 0.343, 0.677, 0.554, 0.496, 0.701, 0.996],
#     r"$r_2$": [0.510, 0.624, 0.266, 0.565, 0.491, 0.732, 0.804],
#     r"$r_3$": [0.26, 0.40, 0.47, 0.55, 0.41, 0.67, 0.72],
#     r"$r_4$": [0.27, 0.44, 0.52, 0.55, 0.44, 0.69, 0.72],
#     "HSII":  [0.600, 0.522, 0.855, 0.898, 0.703, 1.399, 2.149]
# }


# # data = {
# #     "Model": ["GPT4", "deepseek-R1", "Claude", "llama4-Scout", "human"],
# #     r"$r_1$": [0.701, 0.785, 0.766, 0.680, 0.996],
# #     r"$r_2$": [0.732, 0.721, 0.749, 0.734, 0.804],
# #     r"$r_3$": [0.67, 0.69, 0.66, 0.70, 0.72],
# #     r"$r_4$": [0.69, 0.68, 0.65, 0.69, 0.72],
# #     "HSII":  [1.399, 1.560, 1.518, 1.373, 2.149]
# # }

# df = pd.DataFrame(data)
# num_cols = [c for c in df.columns if c != "Model"]
# df[num_cols] = df[num_cols].round(2)


# df_long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# # 2. ★ 手动指定要加粗的 (模型, 指标) 对 ★
# BOLD_CELLS = {
#     ("qwen2.5-7b", r"$r_1$"),
#     ("baichuan2-7b",   r"$r_2$"),
#     ("llama3-8b",    r"$r_3$"),
#     ("llama3-8b",    r"$r_4$"),
#     ("llama3-8b",    "HSII"),
#     ("human",        r"$r_1$"),
#     ("human",        r"$r_2$"),
#     ("human",        r"$r_3$"),
#     ("human",        r"$r_4$"),
#     ("human",        "HSII"),
# }

# # BOLD_CELLS = {
# #     ("deepseek-R1", r"$r_1$"),
# #     ("Claude",   r"$r_2$"),
# #     ("llama4-Scout",    r"$r_3$"),
# #     ("GPT4",    r"$r_4$"),
# #     ("deepseek-R1",    "HSII"),
# #     ("human",        r"$r_1$"),
# #     ("human",        r"$r_2$"),
# #     ("human",        r"$r_3$"),
# #     ("human",        r"$r_4$"),
# #     ("human",        "HSII"),
# # }

# df_long["IsBold"] = df_long.apply(
#     lambda r: (r["Model"], r["Metric"]) in BOLD_CELLS, axis=1
# )

# # 3. 绘图
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(11, 5))
# ax = sns.barplot(
#     data=df_long,
#     x="Model",
#     y="Score",
#     hue="Metric",
#     palette="Set2",
#     width=0.75
# )

# # 4. 顶部数值标注
# for bar, is_bold in zip(ax.patches,
#                         df_long["IsBold"]):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2,
#             height + 0.03,
#             f"{height:.2f}" if height < 1 else f"{height:.2f}",
#             ha='center', va='bottom',
#             fontsize=9,
#             weight='bold' if is_bold else 'normal', rotation=60)

# # 5. 图例放在下方外部
# ax.legend(title=None,
#           loc='upper center',
#           bbox_to_anchor=(0.5, -0.25),
#           ncol=len(df_long["Metric"].unique()),
#           frameon=False)

# # 6. 美化
# # plt.title("Evaluation of Major LLMs on Our Bench", fontsize=14)
# plt.xlabel("Model", fontsize=12)
# plt.ylabel("Score", fontsize=12)
# plt.xticks(rotation=20)
# sns.despine()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.25)

# # 7. 保存/展示
# plt.savefig("llm_bar_manual_bold.pdf", dpi=300, bbox_inches='tight')
# # plt.show()








import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 原始数据
data = {
    "Method": [
        "Evolutionary Algorithm\nRule Search",
        "Reinforcement Learning\nSequence Rule Search",
        "GPT + Monte Carlo\nRule Search",
        "LLM Reasoning\n(Direct Rule Generation)",
        "LLM Reasoning\n(Structure + Optimizer)\n",
        "LLM Reasoning\n(Structure + Optimizer)\n - Finetuned"
    ],
    "Test Total Sales Volume\n($\\times10^6$)":      [4.885, 5.162, 5.379, 5.026, 6.041, 6.120],
    "Test Total Profit\n($\\times10^5$)":            [3.428, 3.910, 4.407, 4.095, 5.012, 5.306],
    "Customers Constraints Violating":               [0.0, 251.3, 44.6, 129.2, 86.9, 85.8],
    "First Epoch to Reach\nTarget Sales Volume":     [243.2, 185.5, 94.6, 199.0, 53.6, 46.3],
    "First Epoch to Reach\nTarget Profit":           [270.8, 213.9, 95.1, 188.4, 59.7, 48.0]
}

# 2. 长数据格式
df = pd.DataFrame(data)
df_long = df.melt(id_vars="Method", var_name="Metric", value_name="Value")

# 3. ★ 手动指定要加粗的 (Method, Metric) 对
BOLD_CELLS = {
    ("LLM Reasoning\n(Structure + Optimizer)\n - Finetuned", "Test Total Sales Volume\n($\\times10^6$)"),
    ("LLM Reasoning\n(Structure + Optimizer)\n - Finetuned", "Test Total Profit\n($\\times10^5$)"),
    ("LLM Reasoning\n(Structure + Optimizer)\n - Finetuned", "First Epoch to Reach\nTarget Sales Volume"),
    ("LLM Reasoning\n(Structure + Optimizer)\n - Finetuned", "First Epoch to Reach\nTarget Profit"),
    ("Evolutionary Algorithm\nRule Search",   "Customers Constraints Violating"),
    
}

df_long["IsBold"] = df_long.apply(
    lambda r: (r["Method"], r["Metric"]) in BOLD_CELLS, axis=1
)

# 4. 画图
sns.set_theme(style="whitegrid")
plt.figure(figsize=(13, 6))
ax = sns.barplot(
    data=df_long,
    x="Method",
    y="Value",
    hue="Metric",
    palette="Set2",
    width=0.75
)

# 5. 顶部数值标签（旋转 60°）
for bar, is_bold in zip(ax.patches,
                        df_long["IsBold"]):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(df_long["Value"]) * 0.015,   # 自适应小偏移
        f"{height:.2f}",
        ha='center', va='bottom',
        fontsize=9,
        rotation=60,
        weight='bold' if is_bold else 'normal'
    )

# 6. 图例放下方外部
ax.legend(title=None,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.20),
          ncol=3,
          frameon=False)

# 7. 细节
plt.title("Adversarial Evaluation between Rule Search Methods at N = 50 Iterations", fontsize=14)
plt.xlabel("")                      # 已有 Method 标签，可留空
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=0)              # 横坐标标签已换行，无需再旋转
sns.despine()
plt.tight_layout()
plt.subplots_adjust(bottom=0.30)

# 8. 保存 / 展示
plt.savefig("rule_search_bar.pdf", bbox_inches='tight')



