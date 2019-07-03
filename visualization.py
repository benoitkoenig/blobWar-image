import ast
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

tf.enable_eager_execution()

from tracking import get_dataframes

pd.plotting.register_matplotlib_converters()

#########################################
# Initializing dataframes and variables #
#########################################

df = get_dataframes()
df["probs"] = df["logits"].map(lambda logits: tf.nn.softmax(eval(logits)).numpy().tolist())
df["label_prob"] = df.apply(lambda row: row["probs"][row["label"]], axis=1)

df_tail = df.tail(10000)
df_pre_tail = df.tail(20000).head(10000)

############
# Plotting #
############

plt.figure(figsize=(18, 12))

# Prob of label over iterations
plt.subplot(2, 2, 1)
label_probs = df["label_prob"].values
plt.plot(label_probs)
plt.title("Right Prob")

# Entropy over iterations
plt.subplot(2, 2, 2)
label_probs = df["entropy"].values
plt.plot(label_probs, color="orange")
plt.title("Entropy")

# Prob density
plt.subplot(2, 2, 3)
plt.violinplot([df_tail["label_prob"], df_pre_tail["label_prob"]])
plt.xticks([1, 2], ["Tail", "PreTail"])
plt.title("Right Prob density")

# Entropy density
plt.subplot(2, 2, 4)
parts = plt.violinplot([df_tail["entropy"], df_pre_tail["entropy"]])
for pb in parts["bodies"]:
    pb.set_facecolor("orange")
parts["cmins"].set_color("orange")
parts["cmaxes"].set_color("orange")
parts["cbars"].set_color("orange")
plt.xticks([1, 2], ["Tail", "PreTail"])
plt.title("Entropy density")

plt.show()
