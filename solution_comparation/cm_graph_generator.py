import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          axis=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])


    stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')
        print(figsize)
    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    # plt.figure(figsize=figsize)
    sns.set(font_scale=1.3)

    sns.heatmap(cf, ax=axis,annot_kws={"size": 14},annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        axis.set_ylabel('True label',fontsize=14)
        axis.set_xlabel('Predicted label' + stats_text,fontsize=14)
    else:
        axis.xlabel(stats_text)

    if title:
        axis.set_title(title,fontsize=14)

cm1={'tn': 951.0, 'tp': 32, 'fp': 38, 'fn': 14} #perfect
cm2={'tn': 969.0, 'tp': 32, 'fp': 20, 'fn': 14} #perfect conf2
cm3={'tn': 875.0, 'tp': 28, 'fp': 114, 'fn': 18} #mi_median conf1
cm4={'tn': 963.0, 'tp': 32, 'fp': 26, 'fn': 14} #mi_median conf2

labels = ['TN', 'FP', 'FN', 'TP']
categories = ['no arcs', 'arcs']
fig, axs = plt.subplots(2,2)

make_confusion_matrix(np.array([[cm1['tn'],cm1['fp']],[cm1['fn'],cm1['tp']]]),
                      group_names=labels,
                      categories=categories,
                      figsize=(4,4),
                      title='Perfect ordering conf1',
                      sum_stats=False,
                      axis = axs[0,0],
cbar=False
                      )
make_confusion_matrix(np.array([[cm2['tn'],cm2['fp']],[cm2['fn'],cm2['tp']]]),
                      group_names=labels,
                      categories=categories,
                      figsize=(4,4),
                      sum_stats=False,
                      axis = axs[0,1],cbar=True,                      title='Perfect ordering conf2',

                      )

make_confusion_matrix(np.array([[cm3['tn'],cm3['fp']],[cm3['fn'],cm3['tp']]]),
                      group_names=labels,
                      categories=categories,
                      figsize=(4,4),
                      sum_stats=False,
                      axis = axs[1,0],cbar=False,                      title='MI_MEDIAN ordering conf1',

                      )
make_confusion_matrix(np.array([[cm4['tn'],cm4['fp']],[cm4['fn'],cm4['tp']]]),
                      group_names=labels,
                      categories=categories,
                      figsize=(4,4),
                      sum_stats=False,
                      axis = axs[1,1],cbar=True,                      title='MI_MEDIAN ordering conf2',

                      )

fig.tight_layout()

plt.show()
