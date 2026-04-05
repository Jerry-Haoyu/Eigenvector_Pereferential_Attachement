from epam.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

def configure_ax(ax, title):
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Graph Probability Signal")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_graph_signal(model, title, save_path):
    fig, ax = plt.subplots()
    plot = ax.plot(np.arange(len(model.pfevec)), model.pfevec / np.sum(model.pfevec))[0]
    configure_ax(ax, title)
    plt.savefig(save_path)
    
    
def animate_graph_signal(model, title, save_path):
    text="Start to Animate Graph Signal"
    print(text.center(20), "-")
    fig, ax = plt.subplots()
    configure_ax(ax, title)
    evs = model.evs
    line = ax.plot(np.arange(len(model.pfevec)), model.pfevec / np.sum(model.pfevec))[0]
    max_frames = 1000
    total_frames = min(model.T, max_frames)
    stride = max(int(model.T / max_frames), 1)
    pbar = tqdm(total=total_frames)
    
    def update(frame):
        pbar.update(n=1)
        evec = evs[stride * frame]
        line.set_data(np.arange(len(evec)), evec/ np.sum(evec))
        return line, 
        
    ani = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=True)
    ani.save(filename=save_path, writer='ffmpeg')


if __name__ == "__main__":
    N=100
    A = np.diag(np.ones(N-1), k=1) + np.diag(np.ones(N-1), k=-1)
    model = Simulator(T=100, m=1, N=1, A0=A, output_path="output/tmp", store_ev=True)
    model.simulate()
    # plot_graph_signal(model, "testing positivity", "data/tmp/testing_positivity")
    animate_graph_signal(model, title="Graph Signal", save_path="data/tmp/test.mp4")