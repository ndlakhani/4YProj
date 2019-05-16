import numpy as np
import matplotlib.pyplot as plt

latticelist = np.load("predictdata.npy")
X, Y = np.meshgrid(range(32), range(32))

lattice = np.real(latticelist[100])
lattice1 = lattice.reshape(32,32)

plt.matshow(lattice1)