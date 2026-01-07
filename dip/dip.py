import torch
from neuromancer.dynamics.ode import ODESystem

class DoubleInvertedPendulum(ODESystem):
    """https://www3.math.tu-berlin.de/Vorlesungen/SS12/Kontrolltheorie/matlab/inverted_pendulum.pdf"""
    def __init__(self, insize=8, outsize=6):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.m0 = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)
        self.m1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.m2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.l1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.l2 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.g = torch.nn.Parameter(torch.tensor([9.80665]), requires_grad=True)

    def single_sample_ode(self, x, u):

        q, theta1, theta2 = x[0], x[1], x[2]
        qdot, theta1dot, theta2dot = x[3], x[4], x[5]

        M = torch.vstack((torch.hstack((self.m0 + self.m1 + self.m2, self.l1*(self.m1 + self.m2)*torch.cos(theta1), self.m2*self.l2*torch.cos(theta2))),
                          torch.hstack((self.l1*(self.m1 + self.m2)*torch.cos(theta1), self.l1**2 * (self.m1 + self.m2), self.l1*self.l2*self.m2*torch.cos(theta1-theta2))),
                          torch.hstack((self.l2*self.m2*torch.cos(theta2), self.l1*self.l2*self.m2*torch.cos(theta1-theta2), self.l2**2 * self.m2))))
        
        f = torch.hstack((self.l1*(self.m1 + self.m2)*theta1dot**2 * torch.sin(theta1) + self.m2*self.l2*theta2dot**2 * torch.sin(theta2),
                          -self.l1*self.l2*self.m2*theta2dot**2 * torch.sin(theta1 - theta2) + self.g*(self.m1 + self.m2)*self.l1*torch.sin(theta1),
                          self.l1*self.l2*self.m2*theta1dot**2 * torch.sin(theta1-theta2) + self.g*self.l2*self.m2*torch.sin(theta2))) 
        input = torch.hstack((u, torch.zeros_like(u), torch.zeros_like(u)))
        disturbance = 0.0*torch.randn(3)

        sol = torch.linalg.solve(M, f + input + disturbance, left=True)

        dx = torch.hstack((qdot, theta1dot, theta2dot, sol))
    
        return dx

    
    def ode_equations(self, x, u):
        
        return torch.vmap(self.single_sample_ode, randomness='different')(x,u)

        
