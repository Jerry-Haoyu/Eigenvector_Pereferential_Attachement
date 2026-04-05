import numpy as np
import scipy as sp
import multiprocessing
import os
import random
from scipy.sparse.linalg import eigsh
from utils.graph import AdjMatrix as AdjMatrix
from utils.graph.DynamicCSR import DynamicCSR
from utils.solver.power import power_iter

import tqdm
import time
import matplotlib.pyplot as plt
import plotext as termplt
import logging 

logger = logging.getLogger("Simulator")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

class Simulator: 
    def __init__(self, 
                 T : int, 
                 m : int, 
                 simulation_index : int, 
                 output_path : str,
                 A0 : DynamicCSR,
                 AdjMatrix = DynamicCSR, 
                 solver = power_iter
                 ):
        """
        Preferential Attachment Simulator 
            args: 
                T : number of time steps 
                m : number of nodes to attach at each iteration
                N : size of the ensemble
                A0 : initial matrix
        """
        self.simulation_index = simulation_index    
        self.output_path = output_path
        #----------------------simulation hyperparameters---------------------------
        self.T = T
        self.m = m
        self.solver = solver
        self.rng = np.random.default_rng() 
        
        #-------------------------------histories----------------------------------
        self.pfevecs = []
        self.pfevals = []
        self.convergence_history = []
        self.oscillations = []
        
        self.A : DynamicCSR = A0
        
    def compute_pfevec(self, shift=None, x0=None):
        if x0 is None:
            x0 = np.random.randn(self.A.shape) 
            x0 = x0 / x0.sum()
        if shift is None:
            shift = 1.0
            
        pfeval, pfevec, stats = self.solver(self.A.matvec, x0=x0, shift=shift, maxiter=int(1e5))
 
        if (pfevec > 0.0).all():
            status = 'success'
        else:
            status = 'negative component'
        
        self.pfevals.append(pfeval)
        self.pfevecs.append(pfevec)
        self.convergence_history.append(stats['iter'])
        self.oscillations.append(stats['oscillation'])
            
        return status
    
    def store_adjacency_matrix(self):
        A_csr = self.A.get_csr()
        sp.sparse.save_npz(file=os.path.join(self.output_path, "FinalAdjMatrix"), matrix=A_csr)
        
    
    def plot_stat(self, data, file_name, title, x_label):
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        fig.savefig(os.path.join(self.output_path,file_name))
        
    def save_convergence_history_plot(self):
        file_name="convergence_history.png"
        title=f"Eigensolver Iteration Taken At Each Epoch For {self.simulation_index}th simulation"
        x_label="simulation steps"
        self.plot_stat(self.convergence_history, file_name, title, x_label)

    
    def save_final_signal_plot(self):
        file_name="final_signal.png"
        title=f"Probability Signal Plot At Epoch {self.T} For {self.simulation_index}th Simulation"
        x_label="nodal_index"
        self.plot_stat(self.pfevecs[self.T-1], file_name, title, x_label)
        
    def save_oscillations_plot(self):
        file_name="oscillations.png"
        title=f"Power Iteration Oscillation At Each Step For {self.simulation_index}th Simulation "
        x_label="simulation steps"
        self.plot_stat(self.oscillations, file_name, title, x_label)
        
       
        

    def simulate(self):
        logging.info(f"{self.simulation_index}th simulation start; total steps: {self.T}, m = {self.m}")
        
        #reseed for independence
        random.seed(os.getpid() + random.randint(0, 1000))
        
        start_time = time.perf_counter()
        #----------compute pfevec, pfeval for A0--------------
        status = self.compute_pfevec()
        x0_ = self.pfevecs[0]
        #------------------simulation loop--------------------
        for t in tqdm.tqdm(range(1,self.T+1)):
            status = self.compute_pfevec(x0=x0_, shift=self.pfevals[t-1]/2)

            p = self.pfevecs[t] / np.sum(self.pfevecs[t])
            
            selected_vertices=self.rng.choice(np.arange(self.A.shape), p=p, size=self.m)

            self.A.add_vertex_with_edges(v=selected_vertices)
            
            x0_ = np.hstack([self.pfevecs[t], 1 / t])

        end_time = time.perf_counter() 
        
        logging.info(f"{self.simulation_index}th simulation finished in {end_time - start_time:.4f} seconds")
        
        logging.info(f"Writing final adjancency matrix and stats to {os.path.join(self.output_path, "FinalAdjMatrix")}")
        
        start_time = time.perf_counter()
        self.store_adjacency_matrix()
        self.save_convergence_history_plot()
        self.save_final_signal_plot()
        self.save_oscillations_plot()
        end_time = time.perf_counter() 
        
        logging.info(f"{self.simulation_index}th simulation finished storing final result in {end_time - start_time:.4f} seconds")
        
    
class EPAMEnsemble():
    def __init__(self, 
                 N : int,
                 T : int, 
                 m : int, 
                 output_path : str,
                 AdjMatrix = DynamicCSR, 
                 solver = power_iter, 
                 A0 : np.ndarray | None = None):
        self.N = N
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        #if not given a starting sate, initialize a clique of size m+1
        if(A0 is None): 
            A_initial = AdjMatrix(np.ones((m+1, m+1)) - np.eye(m+1))
        else:
            A_initial = AdjMatrix(A0)
        self.simulators = []
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for n in range(N):
            per_task_path = os.path.join(output_path, str(n))
            os.makedirs(per_task_path)
            self.simulators.append(Simulator(T, m, n, per_task_path, A_initial))

    def spawn_single_simulation(self, simulator):
        simulator.simulate()
    
    def ensemble_simulate(self):
        with multiprocessing.Pool(processes=self.N) as pool:
            pool.map(self.spawn_single_simulation, self.simulators)
        
        
    

        
    
    
        