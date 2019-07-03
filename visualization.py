import ast
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

tf.enable_eager_execution()

from constants import nb_class
from tracking import get_dataframes

pd.plotting.register_matplotlib_converters()

###############################
# Methods for data formatting #
###############################

def order_by_label(df, key):
    # Return an array containing, for each category, a Series of datetimes
    output = []
    for label in range(nb_class):
        label_output = df[df["label"] == label][key].values
        if len(label_output) == 0:
            label_output = []
        output.append(label_output)
    return output

def get_wrong_prob(row):
    probs_copy = [p for p in row["probs"]]
    probs_copy.pop(row["label"])
    return np.max(probs_copy)

#########################################
# Initializing dataframes and variables #
#########################################

df = get_dataframes()
df["probs"] = df["logits"].map(lambda logits: tf.nn.softmax(eval(logits)).numpy().tolist())
df["label_prob"] = df.apply(lambda row: row["probs"][row["label"]], axis=1)
df["wrong_prob"] = df.apply(get_wrong_prob, axis=1)

df_tail = df.tail(10000)
df_pre_tail = df.tail(20000).head(10000)

############
# Plotting #
############

plt.figure(figsize=(18, 12))

# Prob of label over iterations
plt.subplot(3, 3, 1)
label_probs = df["label_prob"].values
plt.plot(label_probs)
plt.ylim(0., 1.)
plt.title("Right Prob")

# Prob of label density
plt.subplot(3, 3, 2)
plt.violinplot([df_tail["label_prob"], df_pre_tail["label_prob"]])
plt.xticks([1, 2], ["Tail", "PreTail"])
plt.ylim(0., 1.)
plt.title("Right Prob density")

# Prob of label density per label
plt.subplot(3, 3, 3)
prob_per_label = order_by_label(df_tail, "label_prob")
plt.violinplot(prob_per_label)
plt.xticks([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5])
plt.ylim(0., 1.)
plt.title("Right Prob density per label - Tail")

# Entropy over iterations
plt.subplot(3, 3, 4)
label_probs = df["entropy"].values
plt.plot(label_probs, color="orange")
plt.title("Entropy")

# Entropy density
plt.subplot(3, 3, 5)
parts = plt.violinplot([df_tail["entropy"], df_pre_tail["entropy"]])
for pb in parts["bodies"]:
    pb.set_facecolor("orange")
parts["cmins"].set_color("orange")
parts["cmaxes"].set_color("orange")
parts["cbars"].set_color("orange")
plt.xticks([1, 2], ["Tail", "PreTail"])
plt.title("Entropy density")

# Entropy per label
plt.subplot(3, 3, 6)
entropy_per_label = order_by_label(df_tail, "entropy")
parts = plt.violinplot(entropy_per_label)
for pb in parts["bodies"]:
    pb.set_facecolor("orange")
parts["cmins"].set_color("orange")
parts["cmaxes"].set_color("orange")
parts["cbars"].set_color("orange")
plt.xticks([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5])
plt.title("Entropy density per label - Tail")

# Wrong Prob over iterations
plt.subplot(3, 3, 7)
label_probs = df["label_prob"].values
plt.plot(label_probs, color="green")
plt.ylim(0., 1.)
plt.title("Wrong Prob")

# Wrong Prob density
plt.subplot(3, 3, 8)
parts = plt.violinplot([df_tail["wrong_prob"], df_pre_tail["wrong_prob"]])
for pb in parts["bodies"]:
    pb.set_facecolor("green")
parts["cmins"].set_color("green")
parts["cmaxes"].set_color("green")
parts["cbars"].set_color("green")
plt.xticks([1, 2], ["Tail", "PreTail"])
plt.ylim(0., 1.)
plt.title("Wrong Prob density")

# Wrong Prob density per label
plt.subplot(3, 3, 9)
wrong_prob_per_label = order_by_label(df_tail, "wrong_prob")
parts = plt.violinplot(wrong_prob_per_label)
for pb in parts["bodies"]:
    pb.set_facecolor("green")
parts["cmins"].set_color("green")
parts["cmaxes"].set_color("green")
parts["cbars"].set_color("green")
plt.xticks([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5])
plt.ylim(0., 1.)
plt.title("Wrong Prob density per label - Tail")

plt.show()
