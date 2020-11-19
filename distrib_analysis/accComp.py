import numpy as np
import matplotlib.pyplot as plt



accs_cb_shield = {u'E_true':93.1, u'N_spec':86.1}
accs_cb_epos = {u'E_true':91.8, u'N_spec':87.5}
accs_cnn_shield = {u'E_true':93.7, u'N_spec':92.5}
accs_cnn_epos = {u'E_true':91.8, u'N_spec':93.1}

plt.figure(figsize=(6, 3), dpi=160)
plt.scatter(*zip(*accs_cb_shield.items()), c='b', marker='x')
plt.scatter(*zip(*accs_cb_epos.items()), c='r', marker='x')
plt.scatter(*zip(*accs_cnn_shield.items()), c='b', marker='+')
plt.scatter(*zip(*accs_cnn_epos.items()), c='r', marker='+')
plt.ylabel('Accuracy, %')
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.ylim([80, 100])
plt.show()

