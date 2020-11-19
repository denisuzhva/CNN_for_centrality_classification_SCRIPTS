import matplotlib.pyplot as plt

data = {"N_ideal(spec)":15.58, "N_ideal(E_true)":19.82, "N_cut_based":18.76, "N_cnn(spec)":15.14, "N_cnn(E_true)":17.05}

plt.bar(data.keys(), data.values())
plt.grid()
plt.show()
