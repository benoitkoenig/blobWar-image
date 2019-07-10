import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from constants import nb_class
from tracking import get_dataframes

tf.enable_eager_execution()
pd.plotting.register_matplotlib_converters()

###############################
# Methods for data formatting #
###############################

def get_probs_per_label(df):
    output = [[], [], [], [], [], [], []]
    def handle_row(row):
        classification_logits = eval(row["classification_logits"])
        right_labels = eval(row["label_boxes"])
        for i in range(len(classification_logits)):
            logits = classification_logits[i]
            right_label = right_labels[i]
            probs = tf.nn.softmax(logits).numpy().tolist()
            label_prob = probs[right_label]
            output[right_label].append(label_prob)
    df.apply(handle_row, axis=1)
    for i in range(len(output)):
        if (output[i] == []):
            output[i] = -1.
    return output

#########################################
# Initializing dataframes and variables #
#########################################

df = get_dataframes()

############
# Plotting #
############

plt.figure(figsize=(18, 12))

# Prob of label tail
plt.subplot(3, 1, 1)
probs_per_label = get_probs_per_label(df.tail(10000))
plt.violinplot(probs_per_label)
plt.xticks([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
plt.ylim(0., 1.)
plt.title("Label Prob density - Tail")

# Prob of label pre-tail
plt.subplot(3, 1, 2)
probs_per_label = get_probs_per_label(df.tail(20000).head(10000))
plt.violinplot(probs_per_label)
plt.xticks([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
plt.ylim(0., 1.)
plt.title("Label Prob density - Pre-Tail")

# Prob of label overall
plt.subplot(3, 1, 3)
probs_per_label = get_probs_per_label(df)
plt.violinplot(probs_per_label)
plt.xticks([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
plt.ylim(0., 1.)
plt.title("Label Prob density - Overall")

plt.show()
