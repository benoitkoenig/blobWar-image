import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from constants import nb_class
from tracking import get_dataframes

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2
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
                outputs[n][i] = [-1.]
    outputs.append(outputs)
    return outputs

def get_precision_distribution(df):
    outputs = [[[], []], [[], []]]
    def handle_row(row):
        no_regr_precision = eval(row["no_regr_surface_precision"])[0]
        final_precision = eval(row["final_surface_precision"])[0]
        outputs[0][0].append(no_regr_precision[0] / no_regr_precision[1])
        outputs[0][1].append(final_precision[0] / final_precision[1])
        outputs[1][0].append(no_regr_precision[0])
        outputs[1][1].append(final_precision[0])
    df.apply(handle_row, axis=1)
    return outputs

#########################################
# Initializing dataframes and variables #
#########################################

df = get_dataframes()
nb_rows = df["index"].count()
print("Dataframe size: {}".format(nb_rows))

df_tail = df.tail(1000)

all_probs_per_label = get_n_probs_per_label(df_tail)
precision_data = get_precision_distribution(df_tail)

############
# Plotting #
############

fig = plt.figure(figsize=(18, 12))
fig.canvas.set_window_title("Faster-RCNN graph - Last 1000 rows over {} total".format(nb_rows))

# Prob of label tail
plt.subplot(5, 2, 1)
probs_per_label = []
for k in range(7):
    probs_per_label.append(all_probs_per_label[k][k])
parts = plt.violinplot(probs_per_label)
plt.xticks([])
plt.ylim(0., 1.)
plt.yticks([0., 1.])
for pc in parts["bodies"]:
    pc.set_alpha(1)
parts["cmins"].set_alpha(0)
parts["cmaxes"].set_alpha(0)
parts["cbars"].set_alpha(0)
plt.title("Label Prob density")

# Prob of n label tail
for i in range(7):
    plt.subplot(5, 2, 2 + i)
    probs_per_label = all_probs_per_label[i]
    parts = plt.violinplot(probs_per_label)
    plt.xticks([])
    plt.ylim(0., 1.)
    plt.yticks([0., 1.])
    for pc in parts["bodies"]:
        pc.set_alpha(1)
        pc.set_facecolor("#D43F3A")
    parts["cmins"].set_alpha(0)
    parts["cmaxes"].set_alpha(0)
    parts["cbars"].set_alpha(0)
    plt.title("Prob density of {}".format(i))

# Precision distribution
plt.subplot(5, 2, 9)
parts = plt.violinplot(precision_data[0])
plt.xticks([1, 2], ["No Regr", "Final"])
plt.ylim(0., 1.)
plt.yticks([0., 1.])
for pc in parts["bodies"]:
    pc.set_alpha(1)
    pc.set_color("#F3C43A")
parts["cmins"].set_alpha(0)
parts["cmaxes"].set_alpha(0)
parts["cbars"].set_alpha(0)
plt.title("Precision density")

# Coverage distribution
plt.subplot(5, 2, 10)
parts = plt.violinplot(precision_data[1])
plt.xticks([1, 2], ["No Regr", "Final"])
plt.yticks([144], ["Blob\nSurface"])
for pc in parts["bodies"]:
    pc.set_alpha(1)
    pc.set_color("#F3C43A")
parts["cmins"].set_alpha(0)
parts["cmaxes"].set_alpha(0)
parts["cbars"].set_alpha(0)
ax = plt.gca()
ax.axhline(y=144, color="black", lw=1., alpha=.2)
plt.title("Coverage density")

plt.show()
