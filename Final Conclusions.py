import matplotlib.pyplot as plt
import numpy as np

Xtr_all = [.9888, .9829, .8588]
Xval_all = [.7598, .8045, .8500]
Xtr_6 = [.9888, .9841, .8588]
Xval_6 = [.7765, .7818, .8682]
Xtr_3 = [.9228, .9237, .8417]
Xval_3 = [.7877, .8091, .8409]

X = ['Baseline', 'GSMOTE', 'Grid Search']

fig, ax = plt.subplots(1, 3, figsize=(27,9))
ax[0].plot(X, Xtr_3)
ax[0].plot(X, Xval_3)
ax[0].set_title('Top 3 Features')
ax[1].plot(X, Xtr_6)
ax[1].plot(X, Xval_6)
ax[1].set_title('Top 6 Features')
ax[2].plot(X, Xtr_all)
ax[2].plot(X, Xval_all)
ax[2].set_title('All Features')

fig.supylabel('Accuracy')
fig.supxlabel('Model')
fig.legend(['Training', 'Validation'], loc='upper left', bbox_to_anchor=(0.9,0.85))


plt.show()