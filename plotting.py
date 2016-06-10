import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

ground = pickle.load(open("ground.p", "rb"))
predicted = pickle.load(open("predicted.p", "rb"))

data = confusion_matrix(ground[:,11], predicted[:,11])
data = data.astype(int)
print data

import matplotlib.pyplot as plt
column_labels = list('012')
row_labels = list('012')
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)


import matplotlib.pyplot as plt
import numpy as np


for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

plt.colorbar(heatmap)

plt.show()
