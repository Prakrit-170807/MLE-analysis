import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
class rand_distribution:
    def __init__(self, _name, _rsamples):
        self.name = _name
        self.samples = self.format_samples(_rsamples)
        self.mean = None
        self.cov = None
        self.probablitis = None
    def run(self):
        self.mean = self.get_mean()
        self.cov = self.get_cov()
        self.probablitis = self.get_probablitis()
    def format_samples(self, _rsamples):
        np_samples = np.asarray(_rsamples)
        return np_samples.reshape(-1, 2, 1)
    def get_mean(self):
        return (1 / len(self.samples)) * np.sum(self.samples, axis=0)
    def get_cov(self):
        diffs = self.samples - self.mean
        sum = 0
        for dif in diffs:
            sum += np.matmul(dif, np.transpose(dif))
        return (1 / len(self.samples)) * sum

    def get_probablitis(self, samples=None):
        if samples is None:
            samples = self.samples
        exp=np.matmul(np.matmul(np.transpose(samples-self.mean,(0,2,1)), np.linalg.inv(self.cov)), (samples - self.mean))
        denom = np.sqrt(np.linalg.det((2 * np.pi) ** 2 * self.cov))
        prob = np.exp(-0.5 * exp) / denom
        return prob.flatten()
    def plot(self):
        plt.scatter(self.samples[:, 0], self.samples[:, 1])
        plt.title(self.name)
        min_x = np.min(self.samples[:, 0])
        max_x = np.max(self.samples[:, 0])
        min_y = np.min(self.samples[:, 1])
        max_y = np.max(self.samples[:, 1])
        x = np.linspace(min_x - .5, max_x + .5, 100)
        y = np.linspace(min_y - .5, max_y + .5, 100)
        rX, rY = np.meshgrid(x, y)
        Xgrid = np.vstack((rX.flatten(), rY.flatten())).T
        rZ = self.get_probablitis(Xgrid.reshape(-1, 2, 1)).reshape(rX.shape)
        plt.contour(rX, rY, rZ)
        plt.show()
def plot_3d(RD0, RD1, RD2, style='3D'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_x = np.min(np.concatenate((RD0.samples[:, 0], RD1.samples[:, 0], RD2.samples[:, 0])))
    max_x = np.max(np.concatenate((RD0.samples[:, 0], RD1.samples[:, 0], RD2.samples[:, 0])))
    min_y = np.min(np.concatenate((RD0.samples[:, 1], RD1.samples[:, 1], RD2.samples[:, 1])))
    max_y = np.max(np.concatenate((RD0.samples[:, 1], RD1.samples[:, 1], RD2.samples[:, 1])))
    x = np.linspace(min_x - .5, max_x + .5, 100)
    y = np.linspace(min_y - .5, max_y + .5, 100)
    rX, rY = np.meshgrid(x, y)
    Xgrid = np.vstack((rX.flatten(), rY.flatten())).T
    rZ0 = RD0.get_probablitis(Xgrid.reshape(-1, 2, 1)).reshape(rX.shape)
    rZ1 = RD1.get_probablitis(Xgrid.reshape(-1, 2, 1)).reshape(rX.shape)
    rZ2 = RD2.get_probablitis(Xgrid.reshape(-1, 2, 1)).reshape(rX.shape)
    if style != '3D':
        max_Z = np.maximum.reduce([rZ0, rZ1, rZ2])
        plt.contourf(rX, rY, max_Z, levels=360, cmap='RdYlBu')
    else:
        max_indxes = np.argmax(np.stack((rZ0, rZ1, rZ2), axis=2), axis=2)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Z_color = colors[max_indxes]
        rZ = np.maximum.reduce([rZ0, rZ1, rZ2])
        ax.plot_surface(rX, rY, rZ, facecolors=Z_color, linewidth=0.7, antialiased=True, alpha=0.55)
    plt.title("3D plot of all distributions")
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('Probability')
    rl0_patch = mpatches.Patch(color='red', label=RD0.name)
    if style != '3D':
        rl1_patch = mpatches.Patch(color='yellow', label=RD1.name)
    else:
        rl1_patch = mpatches.Patch(color='green', label=RD1.name)
    rl2_patch = mpatches.Patch(color='blue', label=RD2.name)
    plt.legend(handles=[rl0_patch, rl1_patch, rl2_patch])
    plt.show()
def main():
    samples_data = pd.read_csv("rand_MLE.csv", header=None, sep=None)
    print(samples_data)
    grpd = samples_data.groupby(2)

    rdfs = [group[1].drop(2, axis=1) for group in samples_data.groupby(2)]
    rdis0 = rand_distribution("rdis0", rdfs[0])
    rdis0.run()
    rdis1 = rand_distribution("rdis1", rdfs[1])
    rdis1.run()
    rdis2 = rand_distribution("rdis2", rdfs[2])
    rdis2.run()
    plot_3d(rdis0, rdis1, rdis2, style="3D")
if __name__ == "__main__":
    main()
