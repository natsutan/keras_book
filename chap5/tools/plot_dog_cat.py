import pickle
import os
import matplotlib.pyplot as plt

save_dir = '../save'
history_save = os.path.join(save_dir, 'history_tpu_v2.pickle')

history = pickle.load(open(history_save, 'rb'))

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy(TPU)')
plt.legend()
plt.savefig(os.path.join(save_dir, 'plot1_tpu_v2.png'))
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation acc')
plt.title('Training and Validation loss(TPU)')
plt.legend()

plt.savefig(os.path.join(save_dir,'plot2_tpu_v2.png'))
