import matplotlib.pyplot as plt

model = ['baseline', 'MI', 'GSMOTE', 'GSMOTE-MI']
scr_val = [0.7374, 0.7542, 0.8136, 0.8091]
scr_tr = [1.0000, 0.8033, 0.9818, 0.8132]

plt.plot(model, scr_tr, color='#6a79a7')
plt.plot(model, scr_val, color='#d767ad')
plt.legend(['Training', 'Validation'], loc='upper right', bbox_to_anchor=(0.98,0.98))
plt.title('Performance of Models Trained with Different Methods')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.show()

pass