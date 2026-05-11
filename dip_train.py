"""
Learning to stabilize a double inverted pendulum using Differentiable Predictive Control (DPC).

This script uses NeuroMANCER to implement a model-based policy optimization algorithm
that exploits the differentiability of neural network dynamics models.
"""

import torch
import matplotlib.pyplot as plt
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Objective
from neuromancer.loss import PenaltyLoss, AggregateLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import integrators

from dip import DoubleInvertedPendulum
from soap import SOAP


# Configuration parameters
loss_fn = "prob"
constrained_action = True
gamma = 1.0

nsteps = 75
lr = 0.003
opt = 'soap'  # 'adam' or 'soap'
loss = 'gc'  # 'gc' or 'l2'
constrained_pos = True

# System dimensions
nx, nu = 6, 1
ts = 0.05  # Time step

# ============================================================================
# Define nodes and system
# ============================================================================

gamma_dynamics = Node(lambda x: x*gamma, ['gamma'], ['gamma'])

# White-box ODE model
gt_ode = DoubleInvertedPendulum()

# Integrate continuous time ODE
integrator = integrators.RK4_Trap(gt_ode, h=torch.tensor(ts))
dynamics = Node(integrator, ['X', 'U'], ['X'], name='model')

# Observation model
observation = Node(lambda x: 1.0 - torch.cos(x[:,1:3]), ['X'], ['Y'])

# Define cost function
if loss == 'gc':
    prob = lambda x, g: -g*torch.exp(-0.5*(x**2).sum(axis=1, keepdims=True) / 0.5**2) / nsteps
elif loss == 'l2':
    prob = lambda x, g: g*0.5*(x**2).sum(axis=1, keepdims=True) / nsteps
cost = Node(prob, ['Y', 'gamma'], ['l'])

# Define policy network
if constrained_action:
    mlp = blocks.MLP_bounds(nx, nu, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.GELU,
                    min = -10.0, max = 10.0,
                    hsizes=[256, 256])
else:
    mlp = blocks.MLP(nx, nu, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.SiLU,
                    hsizes=[64, 64])

policy = Node(mlp, ['X'], ['U'], name='policy')

# Closed loop system definition
cl_system = System([policy, dynamics, observation, gamma_dynamics, cost], nsteps=nsteps)


# ============================================================================
# Custom loss class
# ============================================================================

class LogLoss(AggregateLoss):
    """
    Logarithmic penalty loss function.
        https://en.wikipedia.org/wiki/Penalty_method
    """

    def __init__(self, objectives, constraints):
        """
        :param objectives: (list (Objective)) list of neuromancer objective classes
        :param constraints: (list (Constraint)) list of neuromancer constraint classes
        """
        super().__init__(objectives, constraints)

    def forward(self, input_dict):
        """
        :param input_dict: (dict {str: torch.Tensor}) Values from forward pass calculations
        :return: (dict {str: torch.Tensor}) input_dict appended with calculated loss values
        """
        objectives_dict = self.calculate_objectives(input_dict)
        input_dict = {**input_dict, **objectives_dict}
        fx = objectives_dict['objective_loss']
        penalties_dict = self.calculate_constraints(input_dict)
        input_dict = {**input_dict, **penalties_dict}
        penalties = penalties_dict['penalty_loss']
        input_dict['loss'] = -torch.log(-1*fx) + penalties
        return input_dict


# ============================================================================
# Generate training dataset
# ============================================================================

samples = 100000
split = int(samples*1.0)
dev_samples = 10000
td = torch.zeros((samples, 1, nx))
td[0:split,:,1:3] = torch.pi

dev_td = torch.zeros((dev_samples, 1, nx))
dev_td[:,:,1:3] = torch.pi

train_data = DictDataset({'X': 0.1*torch.randn((samples, 1, nx)) + td, 'gamma': torch.ones(samples, 1, 1)}, name='train')
dev_data = DictDataset({'X': 0.1*torch.randn((dev_samples, 1, nx)) + dev_td, 'gamma': torch.ones(dev_samples, 1, 1)}, name='dev')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=2048,
                                           collate_fn=train_data.collate_fn, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=2048,
                                         collate_fn=dev_data.collate_fn, shuffle=True)


# ============================================================================
# Define optimization problem
# ============================================================================

u = variable('U')
x = variable('X')

pos_max = 5
state_lower_bound_penalty = (1. / nsteps)*(x[0] > -1*pos_max)
state_upper_bound_penalty = (1. / nsteps)*(x[0] < pos_max)

lpred = variable('l')
l_loss = Objective(var=lpred, name='stage_loss')

if constrained_pos:
    constraints = [state_lower_bound_penalty, state_upper_bound_penalty]
else:
    constraints = []

if loss == 'gc':
    obj = LogLoss([l_loss], constraints)
elif loss == 'l2':
    obj = PenaltyLoss([l_loss], constraints)

problem = Problem([cl_system], obj)

if opt == "adam":
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
elif opt == "soap":
    optimizer = SOAP(policy.parameters(), lr=lr)


# ============================================================================
# Train the model
# ============================================================================

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=50,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=0,
)

best_model = trainer.train()


# ============================================================================
# Save the trained model
# ============================================================================

PATH = f"dip/models/new_policy_{constrained_pos}_{nsteps}_{lr}_{opt}_{loss}_soft.pth"
torch.save(mlp.state_dict(), PATH)

# Load the saved model
mlp.load_state_dict(torch.load(PATH))

# Set the model to evaluation mode
mlp.eval()


# ============================================================================
# Evaluate the model
# ============================================================================

# Load best model weights
problem.load_state_dict(best_model)

# Test with extended horizon
test = torch.zeros((1, 1, nx))
test[:,:,1:3] = torch.pi

data = {'X': 0.1*torch.randn((1, 1, nx)) + test, 'gamma': torch.ones(1, 1, 1, dtype=torch.float32)}
nsteps_eval = 200
cl_system.nsteps = nsteps_eval
trajectories = cl_system(data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(trajectories['Y'].detach().reshape(nsteps_eval, 2).numpy())
plt.xlabel('Time step')
plt.ylabel('Observation')
plt.title('Double Inverted Pendulum Stabilization')
plt.legend(['Angle 1', 'Angle 2'])
plt.grid(True)
plt.savefig('dip/figures/evaluation_results.png')
plt.show()

print("Training complete! Model saved to:", PATH)
