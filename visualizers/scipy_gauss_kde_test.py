from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import time

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

def plot_2d_density(m1, m2, kernel):
    xmin, xmax, ymin, ymax = m1.min(), m1.max(), m2.min(), m2.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()]) #generates x,y pairs corresponding to a fine grid
    t1 = time.time()
    Z = np.reshape(kernel(positions).T, X.shape)
    print(time.time() - t1)

    fig, ax = plt.subplots()
    mappable = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, #need to rotate because 0,0 is top left
               extent=[xmin, xmax, ymin, ymax])
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(mappable=mappable, cax=cax)
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()

if __name__ == "__main__":
    m1,m2 = measure(100000)

    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)

    plot_2d_density(m1, m2, kernel)