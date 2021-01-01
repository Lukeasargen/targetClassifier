

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sn


num_classes = 5
data_size = 20

target_names = [
    "class1", "class2", "class3", "class4", "class5"
]


target = np.random.randint(num_classes, size=data_size)
preds = np.random.randint(num_classes, size=data_size)

print(target)

cm = confusion_matrix(target, preds)

precsion, recall, f1_score, ret = precision_recall_fscore_support(target, preds)

print(precsion)
print(recall)
print(f1_score)
print(ret)


# axis=1 is the target label axis
# normalize by how many labels recieved their correct label
c = cm.astype(np.float).sum(axis=1)
print(c)

cm = cm/c


sn.set(font_scale=1.4)

heatmap = sn.heatmap(cm, annot=True, annot_kws={"size": 16},
    xticklabels=target_names, yticklabels=target_names,
    cmap="RdYlGn"
    # cmap=sn.color_palette("viridis", as_cmap=True)
    )

# heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30) 

plt.show()


