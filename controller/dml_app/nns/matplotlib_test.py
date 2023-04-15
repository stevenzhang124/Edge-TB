import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

x = np.random.randn(60)
y = np.random.randn(60)

plt.scatter(x, y, s=20)

out_png = 'out_file.png'
plt.savefig(out_png, dpi=150)