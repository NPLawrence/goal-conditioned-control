import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import multivariate_normal

sns.set(palette='Set2', style='ticks')

class scenarioCSTR_ParticleStateEnv(gym.Env):
    """
    Gym environment that implements a particle filter state estimator.
    PF implementation adapted from https://aleksandarhaber.com/clear-and-concise-particle-filter-tutorial-with-python-implementation-part-3-python-implementation-of-particle-filter-algorithm
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,
                    num_steps=75,
                    same_state=None,
                    reward_func="prob",
                    render_mode="rgb_array",
                    num_particles=100,
                    combine_particles=True,
                    estimate_state=True,
                    track_estimator=True,
                    mean_state=True,
                    stochastic_env=False,
                    randomized_env=False,
                    tol=0.01,
                    eval="",
                    path=''):        
        super().__init__()

        self.stochastic_env = stochastic_env # If True, the environment is stochastic and follows a branching process. If False, only the model is stochastic and the environment is deterministic.
        self.randomized_env = randomized_env

        ## dynamics stuff
        # Certain parameters
        self.K0_ab = 1.287e12 # K0 [h^-1]
        self.K0_bc = 1.287e12 # K0 [h^-1]
        self.K0_ad = 9.043e9 # K0 [l/mol.h]
        self.R_gas = 8.3144621e-3 # Universal gas constant
        self.E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
        self.E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
        self.E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
        self.H_R_ab = 4.2 # [kj/mol A]
        self.H_R_bc = -11.0 # [kj/mol B] Exothermic
        self.H_R_ad = -41.85 # [kj/mol A] Exothermic
        self.Rou = 0.9342 # Density [kg/l]
        self.Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
        self.Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
        self.A_R = 0.215 # Area of reactor wall [m^2]
        self.V_R = 10.01 #0.01 # Volume of reactor [l]
        self.m_k = 5.0 # Coolant mass[kg]
        self.T_in = 130.0 # Temp of inflow [Celsius]
        self.K_w = 4032.0 # [kj/h.m^2.K]
        self.C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

        self.alpha_range = [0.90, 1.1] # uncertain alpha/beta ranges for our model https://www.do-mpc.com/en/latest/example_gallery/CSTR.html
        self.beta_range = [0.90, 1.1]

        n, m, d = 4, 2, 2
        self.ts = 0.005

        self.tol = tol

        self.goal = np.array([0.6]) # desired concentration of B (second component)
        self.init_state = np.array([0.8, 0.5, 134.14, 130.0]) # reference for initializing the environment at a reasonable place

        self.state_low = np.array([0.1, 0.1, 50.0, 50.0])
        self.state_high = np.array([4.0, 4.0, 200.0, 200.0])

        self.state_cov = 0.05*np.eye(n) # this environment can be modified to have process noise rather than parameter uncertainty
        self.obs_cov = 0.05*np.eye(d)
        self.state_distribution = multivariate_normal(mean=np.zeros(n), cov=self.state_cov)
        self.obs_distribution = multivariate_normal(mean=np.zeros(d), cov=self.obs_cov)

        ## particle filter stuff
        self.estimate_state = estimate_state
        self.mean_state = mean_state if self.estimate_state else False 
        self.num_particles = num_particles
        self.combine_particles = combine_particles if not self.mean_state else 1
        self.track_estimator = track_estimator
        if self.track_estimator:
            self.all_state_estimate = []
            self.all_state_var = []
            self.all_state_extreme = []

        ## environment stuff
        self.num_steps = num_steps
        self.same_state = same_state

        self.reward_func = reward_func

        self.episode = 0

        self.state_space = spaces.Box(low=0.7*self.init_state, high=1.1*self.init_state, dtype=np.float32) # this is used for initializing the underlying system
        if self.mean_state or not self.estimate_state:
            self.observation_space = spaces.Box(low=0.7*self.init_state, high=1.1*self.init_state, dtype=np.float32)
        elif self.combine_particles:
            self.observation_space = spaces.Box(low=-1.0*np.ones(n*self.num_particles), high=np.ones(n*self.num_particles), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-1.0*np.ones((self.num_particles,n)), high=np.ones((self.num_particles,n)), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.5, -8.5]), high=np.array([10.0, 0.0]), dtype=np.float32)

        ## rendering stuff
        self.render_mode = render_mode
        self.state_hist = []
        if self.render_mode in ["human", "rgb_array"]:
            if self.render_mode == "human":
                plt.ion() # Turn on interactive mode for live updates
            if self.track_estimator and self.estimate_state:
                self.fig, self.ax = plt.subplots(4, 1, figsize=(5, 5),layout='constrained')

                self.line1, = self.ax[0].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps') # Initialize an empty line plot
                self.line2, = self.ax[0].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps')
                self.line3, = self.ax[0].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='grey')
                self.line4, = self.ax[0].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='grey')
                
                self.line5, = self.ax[1].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps') # Initialize an empty line plot
                self.line6, = self.ax[1].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps')
                self.line7, = self.ax[1].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='grey')
                self.line8, = self.ax[1].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='grey')
                self.line9, = self.ax[1].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='red')

                self.line10, = self.ax[2].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps')
                
                self.line11, = self.ax[3].plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps')
                
                self.ax[3].set_xlabel("Time Step")
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            else:
                self.fig, self.ax = plt.subplots(figsize=(5, 5),layout='constrained')
                self.line1, = self.ax.plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps', lw=1.0, ls='--', c='grey') # Initialize an empty line plot
                self.line2, = self.ax.plot(range(len(self.state_hist)), self.state_hist, drawstyle='steps') # Initialize an empty line plot
                self.ax.set_xlabel("Time Step")
                self.ax.set_ylabel("Concentration")
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        
        ## misc
        self.eval = eval
        self.path=path

    def step(self, action):
        self.t += 1

        self.action = action

        if self.stochastic_env:
            alpha = np.random.rand()*(self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
            beta = np.random.rand()*(self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        else:
            alpha, beta = self.alpha, self.beta
        
        self.state = self._transition(self.state, action, alpha=alpha, beta=beta) # + self.state_distribution.rvs()
        self.observation = self._obs(self.state) #+ self.obs_distribution.rvs()

        if self.estimate_state:
            self._particle_filter()
            
            mean = self.state_pf
            var = np.mean((self.particles - mean)**2, axis=0)
            max_particle = np.max(self.particles, axis=0)
            min_particle = np.min(self.particles, axis=0)

            obs = self.particles_state if not self.mean_state else mean
            if np.any(np.isnan(obs)):
                print("observation step")
            if self.track_estimator:
                self.all_state_estimate.append([self.state, self.state_pf])
                self.all_state_var.append(var)
                self.all_state_extreme.append([max_particle, min_particle])
        else:
            obs = self.state

        
        info = self._get_info()
        reward, terminated, truncated = info["reward"], info["terminated"], info["TimeLimit.truncated"]

        if self.render_mode in ["human", "rgb_array"]:
            self.state_hist.append(self.state)

        return obs, reward, terminated, truncated, info
    
    def _transition(self, state, action, alpha=1.0, beta=1.0):
        return np.clip(self._rk4_integrate(state, action, alpha=alpha, beta=beta), 0.99*self.state_low, self.state_high*1.01) # if clipped the environment will also terminate 

    def _obs(self, state):
        return state[[2,3]]

    def compute_reward(self):

        axis = 0

        action = self.action

        if self.estimate_state:
            state = self.state_pf
            if self.reward_func == "prob":
                reward = self.goal_prob
            elif self.reward_func == "quadratic":
                reward = -self.goal_cost
        else:
            state = self.state
            if self.reward_func == "prob":
                reward = np.exp(-0.5*((self.goal[0] - state[1]) / 0.01)**2) # choosing 0.01 to align with the particle filter reward
        
            elif self.reward_func == "quadratic":
                reward = -0.5*((self.goal[0] - state[1]) / 0.01)**2
            elif self.reward_func == "gaussian":
                reward = np.exp(-0.5*((self.goal[0] - state[1]) / 0.05)**2) # choosing 0.05 to align with the observation uncertainty
            elif self.reward_func == "binary":
                is_target = np.linalg.norm(state, np.inf, axis=axis) < 0.1
                reward = 1*is_target

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # true alpha/beta parameters
        if self.randomized_env:
            self.alpha = np.random.rand()*(1.1 - 0.90) + 0.90
            self.beta = np.random.rand()*(1.1 - 0.90) + 0.90
        else:
            self.alpha = 1.0
            self.beta = 1.0

        self.episode += 1
        self.state_hist = []
        self.particle_param = [] # ordered list of parameters according to weights in the particle filter
        self.cumulative_time_near_goal = 0.0
        self.time_near_state = 0.0

        if self.track_estimator:
            self.all_state_estimate = []
            self.all_state_var = []
            self.all_state_extreme = []

        x, self.action = self._reset_state()
        self.state = self._transition(x, self.action, alpha=self.alpha, beta=self.beta)
        self.observation = self._obs(self.state)

        if self.estimate_state:
            self._init_particle_filter()
            self._particle_filter()
            mean = self.state_pf
            obs = self.particles_state if not self.mean_state else mean
        else:
            obs = self.state
        self.t = 0.0
        info = self._get_info()

        if self.render_mode in ["human", "rgb_array"]:
            self.state_hist.append(self.state)
            self.render()        

        return obs, info
        
    def _reset_state(self):
        if self.same_state is not None: # dictates whether the system starts from the same state between episodes
            state, action, alpha, beta = self.same_state
            self.alpha = alpha
            self.beta = beta
            return state, action
        else:
            return np.random.uniform(low=self.state_space.low, high=self.state_space.high), self.action_space.sample()

    def _get_info(self):

        reward = self.compute_reward()
        distance = np.abs(self.goal[0] - self.state[1])

        self.cumulative_time_near_goal += np.exp(-0.5*(distance / self.tol)**2)
        output_distance = np.linalg.norm(self.observation).item()
        terminated = np.any(self.state > self.state_high) or np.any(self.state < self.state_low)
        truncated = self.t == self.num_steps
        
        return {"time": self.t, "distance": distance, "output_distance": output_distance, "reward": reward, "terminated": terminated, "TimeLimit.truncated": truncated, "full_state": self.state, "cumulative_time_near_goal": self.cumulative_time_near_goal, "time_near_state":self.time_near_state}

    def _init_particle_filter(self):
        self.particles = [np.random.uniform(low=self.state_space.low, high=self.state_space.high) for _ in range(self.num_particles)]
        if self.combine_particles:
            self.particles_state = np.concatenate(self.particles)
        else:
            self.particles_state = np.concatenate(self.particles).reshape(self.num_particles, -1)
        self.weights = [1 / self.num_particles for _ in range(self.num_particles)]
        

    def _particle_filter(self):

        # step 1: advance the particles using the state transition probability
        rand_alpha = np.random.rand(self.num_particles)*(self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
        rand_beta = np.random.rand(self.num_particles)*(self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        new_states = [self._transition(state, self.action, alpha=alpha, beta=beta) for state, alpha, beta in zip(self.particles, rand_alpha, rand_beta)]
        # new_states = [state + self.state_distribution.rvs() for state in new_states_nonoise]
        if np.any(np.isnan(new_states)):
            print("new states")

        # step 2: update weights using measurement model
        new_weights_unnormalized = [np.clip(multivariate_normal(mean=self._obs(state), cov=self.obs_cov).pdf(self.observation),a_min=1e-8,a_max=None) * weight for (state,weight) in zip(new_states, self.weights)] # clipping to avoid some pathological NaNs
        if np.any(np.isnan(new_weights_unnormalized)):
            print("new_weights_unnormalized")        
        new_weights = new_weights_unnormalized / sum(new_weights_unnormalized)
        if np.any(np.isnan(new_weights)):
            print("new_weights") 
            print(new_weights_unnormalized)  
        self.particle_param = [self.alpha, self.beta, rand_alpha[np.argsort(new_weights)], rand_beta[np.argsort(new_weights)]]

        # step 2.5: estimate the probability of being at the goal
        # This is an approximation of the indicator at the goal times weights associated with the observation
        # We can directly see how estimation and control are in competition here
        goal_weights_unnormalized = [np.exp(-0.5*(self.goal - state[1])**2 / self.tol**2)* weight for (state,weight) in zip(new_states, new_weights)]
        goal_costs_unnormalized = [0.5*(self.goal - state[1])**2 / self.tol**2 * weight for (state,weight) in zip(new_states, new_weights)]
        self.goal_prob = np.mean(goal_weights_unnormalized)
        self.goal_cost = np.mean(goal_costs_unnormalized)

        # get mean state
        self.state_pf = sum(w*x for w, x in zip(new_weights, new_states))
        self.time_near_state += np.mean(np.exp(-0.5*(self.particles - self.state_pf)**2 / self.tol**2))

        # step 3: resampling
        tmp1=[val**2 for val in new_weights]
        Neff=1/(np.array(tmp1).sum())
        # resample if this condition is met
        if Neff<(self.num_particles//3):
            resampled_state_idx=np.random.choice(np.arange(self.num_particles), self.num_particles, p=new_weights)
            new_states = [new_states[idx] for idx in resampled_state_idx]
            new_weights = [1 / self.num_particles for _ in range(self.num_particles)]

        self.particles = new_states
        # self.particles_state = np.concatenate(new_states)
        if self.combine_particles:
            self.particles_state = np.concatenate(new_states)
        else:
            self.particles_state = np.concatenate(new_states).reshape(self.num_particles, -1)
        self.weights = new_weights
        
        return self.state_pf, self.goal_prob


    def _ode_equations(self, x, alpha=1.0, beta=1.0):
        C_a = x[0] # state: Concentration of A in CSTR (mol/m^3)
        C_b = x[1] # concentration of B
        T_R = x[2] # temperature inside the reactor
        T_K = x[3] # temperature of the cooling jacket 
        
        F = x[4] # control: feed
        Q_dot = x[5] # heat flow

        T_dif = T_R-T_K
        K_1 = beta * self.K0_ab * np.exp((-self.E_A_ab)/((T_R+273.15)))
        K_2 =  self.K0_bc * np.exp((-self.E_A_bc)/((T_R+273.15)))
        K_3 = self.K0_ad * np.exp((-alpha*self.E_A_ad)/((T_R+273.15)))
        
        C_a_dot = 10*F*(self.C_A0 - C_a) -K_1*C_a - K_3*(C_a**2)
        C_b_dot = -10*F*C_b + K_1*C_a - K_2*C_b
        T_R_dot = ((K_1*C_a*self.H_R_ab + K_2*C_b*self.H_R_bc + K_3*(C_a**2)*self.H_R_ad)/(-self.Rou*self.Cp)) + 10*F*(self.T_in-T_R) +(((self.K_w*self.A_R)*(-T_dif))/(self.Rou*self.Cp*self.V_R))
        T_K_dot = (1000.0*Q_dot + self.K_w*self.A_R*(T_dif))/(self.m_k*self.Cp_k)

        return np.array([C_a_dot, C_b_dot, T_R_dot, T_K_dot, 0.0, 0.0])

    def _rk4_integrate(self, x, u, alpha=1.0, beta=1.0):
        h = self.ts
        x_aug = np.concatenate((x,u))
        k1 = self._ode_equations(x_aug, alpha=alpha, beta=beta)                   # k1 = f(x_i, t_i)
        k2 = self._ode_equations(x_aug + h*k1/2.0, alpha=alpha, beta=beta)        # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self._ode_equations(x_aug + h*k2/2.0, alpha=alpha, beta=beta)         # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self._ode_equations(x_aug + h*k3, alpha=alpha, beta=beta)            # k4 = f(y_i + h*k3, t_i + h)
        x_next = x_aug + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
        return x_next[0:4]

    def render(self):
        if self.render_mode is None:
            return None

        if self.track_estimator and self.estimate_state:
            
            obs = [self._obs(s[0]) for s in self.all_state_estimate]

            states = [s[0] for s in self.all_state_estimate]
            states_pf = [s[1] for s in self.all_state_estimate]
            states_pf_min = [s[1] for s in self.all_state_extreme]
            states_pf_max = [s[0] for s in self.all_state_extreme]

            self.line10.set_data(range(len(states)), [o[0] for o in obs])

            self.line11.set_data(range(len(states)), [o[1] for o in obs])

            self.line1.set_data(range(len(states)), [s[0] for s in states])
            self.line2.set_data(range(len(states)), [s[0] for s in states_pf])
            self.line3.set_data(range(len(states)), [s[0] for s in states_pf_min])
            self.line4.set_data(range(len(states)), [s[0] for s in states_pf_max])

            self.line5.set_data(range(len(states)), [s[1] for s in states])
            self.line6.set_data(range(len(states)), [s[1] for s in states_pf])
            self.line7.set_data(range(len(states)), [s[1] for s in states_pf_min])
            self.line8.set_data(range(len(states)), [s[1] for s in states_pf_max])
            self.line9.set_data(range(len(states)), [self.goal[0] for _ in states])

            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.ax[2].relim()
            self.ax[2].autoscale_view()
            self.ax[3].relim()
            self.ax[3].autoscale_view()

        else:
            self.line1.set_data(range(len(self.state_hist)), np.array([self.goal[0] for _ in self.state_hist]))
            self.line2.set_data(range(len(self.state_hist)), np.array([s[1] for s in self.state_hist]))
            self.ax.relim()
            self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self.render_mode == "human":
            plt.pause(0.01) # Pause to allow GUI to update
            return None

        elif self.render_mode == "rgb_array":

            self.fig.canvas.draw()
            image_rgba = np.asarray(self.fig.canvas.buffer_rgba())

            # If only RGB is needed, discard the alpha channel
            image_rgb = image_rgba[:, :, :3]
            return image_rgb
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

if __name__ == '__main__':
    print("Checking env")

    import matplotlib.pyplot as plt
    from envs.TerminalFrameCapture import TerminalFrameCapture

    # run environment
    num_steps = 100
    env = scenarioCSTR_ParticleStateEnv(reward_func="prob", estimate_state=True, track_estimator=True, num_steps=num_steps, num_particles=100, mean_state=True)
        
    env = TerminalFrameCapture(env, path='testing/run')
    env.reset(seed=0)
    ret = 0.0
    for t in range(num_steps):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        ret += reward * 0.99**t
        
    plt.show()
    env.close()