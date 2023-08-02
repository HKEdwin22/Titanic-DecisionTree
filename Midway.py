import matplotlib.pyplot as plt

tr = [0.9888, 0.9228, 0.9829, 0.9237]
val = [0.7598, 0.7877, 0.8045, 0.8091]

x = ['Baseline', 'MI', 'GSMOTE', 'GSMOTE-MI']

plt.plot(x, tr, color='#6a79a7')
plt.plot(x, val, color='#d767ad')

plt.title('Performance of Models')
plt.legend(['Training', 'Validation'], loc='upper right', bbox_to_anchor=(1.05,1))
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.show()