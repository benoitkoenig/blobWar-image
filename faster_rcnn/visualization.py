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

def get_n_probs_per_label(df):
    outputs = []
    for n in range(7):
        outputs.append([[], [], [], [], [], [], []])
    def handle_row(row):
        classification_logits = eval(row["classification_logits"])
        right_labels = eval(row["label_boxes"])
        for i in range(len(classification_logits)):
            logits = classification_logits[i]
            right_label = right_labels[i]
            probs = tf.nn.softmax(logits).numpy().tolist()
            for n in range(7):
                n_prob = probs[n]
                outputs[right_label][n].append(n_prob)
    df.apply(handle_row, axis=1)
    for n in range(7):
        for i in range(len(outputs[n])):
            if (outputs[n][i] == []):
                outputs[n][i] = -1.
    outputs.append(outputs)
    return outputs

#########################################
# Initializing dataframes and variables #
#########################################

df = get_dataframes()

############
# Plotting #
############

plt.figure(figsize=(18, 12))

# Prob of label tail
for i in range(10):
    j = i + 1
    all_probs_per_label = get_n_probs_per_label(df.tail(500 * j).head(500))
    plt.subplot(5, 2, j)
    probs_per_label = []
    for k in range(7):
        probs_per_label.append(all_probs_per_label[k][k])
    parts = plt.violinplot(probs_per_label)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
    plt.ylim(0., 1.)
    for pc in parts["bodies"]:
        pc.set_alpha(1)
    parts["cmins"].set_alpha(0)
    parts["cmaxes"].set_alpha(0)
    parts["cbars"].set_alpha(0)
    plt.title("Label Prob density {}".format(i))

# all_probs_per_label = get_n_probs_per_label(df.tail(10000))

# Prob of n label tail
# for i in range(7):
#     plt.subplot(5, 2, 4 + i)
#     probs_per_label = all_probs_per_label[i]
#     parts = plt.violinplot(probs_per_label)
#     plt.xticks([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
#     plt.ylim(0., 1.)
#     for pc in parts["bodies"]:
#         pc.set_alpha(1)
#         pc.set_facecolor("#D43F3A")
#     parts["cmins"].set_alpha(0)
#     parts["cmaxes"].set_alpha(0)
#     parts["cbars"].set_alpha(0)
#     plt.title("Prob density of {}".format(i))

plt.show()
