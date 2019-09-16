
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.ndimage

class RunningPlot:
    def __init__(self, t=1e-6):
        self.t = t

    def __enter__(self):
        plt.clf()

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.draw()
        plt.pause(self.t)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def reward_plotter(rewards, title, col='b', smooth_factor=0, include_sd=False):
    means = np.mean(rewards, 0)
    if smooth_factor >= 0:
        try:
            means = scipy.ndimage.filters.gaussian_filter1d(means, smooth_factor)
        except ZeroDivisionError:
            pass
    plt.plot(means, col, label=title, alpha=0.75)
    if include_sd:
        sds = np.std(rewards, 0)
        if smooth_factor >= 0:
            try:
                pass
                sds = scipy.ndimage.filters.gaussian_filter1d(sds, smooth_factor)
            except ZeroDivisionError:
                pass
        plt.plot(means + sds, col, alpha=0.1)
        plt.plot(means - sds, col, alpha=0.1)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()


def render_helper(env, title, i):
    with RunningPlot(0.1):
        plt.figure(1, figsize=(4, 4))
        plt.imshow(env.render())
        plt.title(title + ', step: {}'.format(i))


def plot_helper(title, e, agent, env):
    with RunningPlot():
        # Up, down, left, right
        plt.figure(2, figsize=(8, 4))
        plt.suptitle(title + ', episode: ' + str(e), x=0.1, y=1, fontsize=20, horizontalalignment='left')

        plt.subplot(231)
        plt.axis('off')
        plt.title('Value of Up')
        img1 = plt.imshow(agent.Qtable[:,:,0])
        plt.axis('equal', frameon=True)
        colorbar(img1)

        plt.subplot(232)
        plt.axis('off')
        plt.title('Value of Right')
        img2 = plt.imshow(agent.Qtable[:,:,1])
        plt.axis('equal', frameon=True)
        colorbar(img2)

        plt.subplot(234)
        plt.axis('off')
        plt.title('Value of Down')
        img3 = plt.imshow(agent.Qtable[:,:,2])
        # img3 = plt.imshow(np.max(agent.Qtable, -1))
        plt.axis('equal', frameon=True)
        colorbar(img3)

        plt.subplot(235)
        plt.axis('off')
        plt.title('Value of Left')
        img4 = plt.imshow(agent.Qtable[:,:,3])
        # img4 = plt.imshow(np.max(agent.Qtable, -1))
        plt.axis('equal', frameon=True)
        colorbar(img4)

        plt.subplot(233)
        plt.axis('off')
        plt.title('Best Action')
        img4 = plt.imshow(np.argmax(agent.Qtable, -1))
        # img4 = plt.imshow(np.max(agent.Qtable, -1))
        plt.axis('equal', frameon=True)
        colorbar(img4)

        plt.subplot(236)
        plt.title('Movement Heatmap')
        img5 = plt.imshow(env.heat_map)
        plt.axis('equal')
        colorbar(img5)
