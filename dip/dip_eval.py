import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Objective
from neuromancer.loss import PenaltyLoss, AggregateLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase, plot_trajectories
import neuromancer.psl as psl
from neuromancer.dynamics import ode, integrators

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dip import DoubleInvertedPendulum

## Instantiate DIP
nx, nu = 6, 1
ts = 0.05

# white-box ODE model with no-plant model mismatch
gt_ode = DoubleInvertedPendulum()                   # ODE system equations implemented in PyTorch

# integrate continuous time ODE
integrator = integrators.RK4_Trap(gt_ode, h=torch.tensor(ts))   # RK4, RK4_Trap, Runge_Kutta_Fehlberg, LeapFrog
dynamics = Node(integrator, ['X', 'U'], ['X'], name='model')
observation = Node(lambda x: torch.cos(x[:,1:3]), ['X'], ['Y'])

## Various experimental parameters for comparison
constrained_pos = True
nsteps = 75
lr = 0.003
opt_soap = 'soap'
opt_adam = 'adam'
loss_gc = 'gc'
loss_l2 = 'l2'

## Instantiate closed-loop systems
mlps = [blocks.MLP_bounds(nx, nu, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.GELU,
                min = -10.0, max = 10.0,
                hsizes=[256, 256]) for _ in range(4)]

PATH_0 = f"neuromancer/models/policy_{constrained_pos}_{nsteps}_0.003_{opt_soap}_{loss_gc}.pth" 
PATH_1 = f"neuromancer/models/policy_{constrained_pos}_{nsteps}_0.003_{opt_soap}_{loss_l2}_soft.pth" 
PATH_2 = f"neuromancer/models/policy_{constrained_pos}_{nsteps}_0.001_{opt_adam}_{loss_gc}.pth" 
PATH_3 = f"neuromancer/models/policy_{constrained_pos}_{nsteps}_0.001_{opt_adam}_{loss_l2}_soft.pth" 

paths = [PATH_0, PATH_1, PATH_2, PATH_3]

for (i,p) in enumerate(paths):
    mlps[i].load_state_dict(torch.load(p))
    # Set the model to evaluation mode if you're using it for inference
    mlps[i].eval() 

policies = [Node(mlp, ['X'], ['U'], name=f'policy_{i}') for (i,mlp) in enumerate(mlps)]

# closed loop system definition
nsteps_eval = 100
samples = 1000
cl_systems = [System([pi, dynamics, observation], nsteps=nsteps_eval) for pi in policies]

## Evaluate all policies over same data
data_init = torch.zeros((samples, 1, nx))
data_init[:,:,1:3] = torch.pi #* (torch.rand_like(test[:,:,1:3])*0.2 + 0.9)
data = {'X': 0.1*torch.randn((samples, 1, nx)) + data_init}
trajectories = [cl(data)['Y'].detach().numpy() for cl in cl_systems]

trajectories_median = [np.median(traj, axis=0) for traj in trajectories]
trajectories_low = [np.min(traj, axis=0) for (traj,med) in zip(trajectories,trajectories_median)]
trajectories_high = [np.max(traj, axis=0) for (traj,med) in zip(trajectories,trajectories_median)]

## Plotting
plt.style.use('neuromancer/ieee.mplstyle')
sns.set(palette='Set2', style='ticks')

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=2.5)

params = {
        "text.usetex" : True,
        "font.family" : "serif",
        "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

fig, ax = plt.subplots(2,2, sharex=True, sharey=True, layout='constrained')
plt.yticks([-1, 0, 1], [-1, 0, 1])


ax[0,0].plot(trajectories_median[0][:,0])
ax[0,0].plot(trajectories_median[0][:,1], linestyle='--')
ax[0,0].fill_between(range(nsteps_eval), trajectories_low[0][:,0], trajectories_high[0][:,0], alpha=0.3)
ax[0,0].fill_between(range(nsteps_eval), trajectories_low[0][:,1], trajectories_high[0][:,1], alpha=0.3)
ax[0,0].set_title("Gaussian cost")
ax[0,0].set_ylabel("SOAP")

ax[0,1].plot(trajectories_median[1][:,0])
ax[0,1].plot(trajectories_median[1][:,1], linestyle='--')
ax[0,1].fill_between(range(nsteps_eval), trajectories_low[1][:,0], trajectories_high[1][:,0], alpha=0.3)
ax[0,1].fill_between(range(nsteps_eval), trajectories_low[1][:,1], trajectories_high[1][:,1], alpha=0.3)
ax[0,1].set_title("Quadratic cost")

ax[1,0].plot(trajectories_median[2][:,0])
ax[1,0].plot(trajectories_median[2][:,1], linestyle='--')
ax[1,0].fill_between(range(nsteps_eval), trajectories_low[2][:,0], trajectories_high[2][:,0], alpha=0.3)
ax[1,0].fill_between(range(nsteps_eval), trajectories_low[2][:,1], trajectories_high[2][:,1], alpha=0.3)
ax[1,0].set_ylabel("Adam")
# ax[1,0].set_yticks([-1, 0, 1])
# ax[1,0].set_yticklabels([-1, r"$\cos(\theta)$", 1], rotation='vertical')

ax[1,1].plot(trajectories_median[3][:,0])
ax[1,1].plot(trajectories_median[3][:,1], linestyle='--')
ax[1,1].fill_between(range(nsteps_eval), trajectories_low[3][:,0], trajectories_high[3][:,0], alpha=0.3)
ax[1,1].fill_between(range(nsteps_eval), trajectories_low[3][:,1], trajectories_high[3][:,1], alpha=0.3)

plt.savefig("neuromancer/figures/DIP_policies.png")
plt.savefig("neuromancer/figures/DIP_policies.pdf")
# plt.show()





