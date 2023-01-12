import numpy as np
from scipy.linalg import expm

# calc time
import time

# animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization.bloch import Bloch

# ML part
import torch

# Calculation stuff for plotting
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)


def calc_location(state):
    # We assume the state is given as an array.
    npstate = np.array(state, dtype=complex)
    x = np.matmul(np.conjugate(npstate), np.matmul(s1, np.transpose(npstate))).real
    y = np.matmul(np.conjugate(npstate), np.matmul(s2, np.transpose(npstate))).real
    z = np.matmul(np.conjugate(npstate), np.matmul(s3, np.transpose(npstate))).real
    return [x, y, z]

# Plotting stuff

seed = 0
np.random.seed(seed)


class QubitEnv:
    def __init__(self, seed, n_time_steps):
        self.n_time_steps = n_time_steps
        # define action space variables
        self.n_actions = 4  # action space size
        delta_t = 2 * np.pi / self.n_time_steps  # set a value for the time step

        # define Pauli matrices
        identity = np.array([[1.0, 0.0], [0.0, +1.0]], dtype=complex)
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        self.action_space = []
        for generator in [identity, sigma_x, sigma_y, sigma_z]:
            self.action_space.append(expm(-1j * delta_t * generator))

        self.actions = np.array([0, 1, 2, 3])

        self.S_target = np.array([0.0, 0.0], dtype=float)
        self.psi_target = self.sphere_to_qubit(self.S_target)

        self.set_seed(seed)

        self.state = self.set_state(False)
        self.psi = self.sphere_to_qubit(self.state)

    def set_seed(self, seed):
        np.random.seed(seed)

    def set_state(self, random=True):
        """Initiate state to a random state or to a fixed state for reproducibility"""
        if random:
            theta = np.pi * np.random.uniform(0.0, 1.0)
            phi = 2 * np.pi * np.random.uniform(0.0, 1.0)
        else:
            # start from south pole of Bloch sphere
            theta = np.pi
            phi = 0.0

        self.state = np.array([theta, phi], dtype=float)
        self.psi = self.sphere_to_qubit(self.state)

        return self.state

    def sphere_to_qubit(self, s):
        """
        Take as input the spherical state s, and return the quantum state |psi>
        """
        theta, phi = s
        psi = np.array([np.cos(0.5 * theta), np.exp(1j * phi) * np.sin(0.5 * theta)], dtype=complex)
        return psi

    def qubit_to_sphere(self, psi):
        """
        Take as input the quantum state |psi>, and return the spherical state s
        """
        # take away unphysical global phase
        alpha = np.angle(psi[0])
        psi_new = np.exp(-1j * alpha) * psi

        # find Bloch sphere angles
        theta = 2.0 * np.arccos(psi_new[0]).real
        phi = np.angle(psi_new[1])

        return np.array([theta, phi], dtype=float)

    def step(self, action):
        """Taking one time step by multiplying with an operator out of the action space list."""
        # apply gate to quantum state
        psi_old = self.psi.copy()
        self.psi = self.action_space[action].dot(self.psi)

        # compute RL state
        self.state = self.qubit_to_sphere(self.psi)

        # compute reward
        reward = np.abs(self.psi_target.conj().dot(self.psi)) ** 2

        # check if state is terminal #TODO
        if np.abs(reward - 1) < 10e-6:
            done = True
        else:
            done = False

        return self.state, reward, done


n_time_steps = 50
Env = QubitEnv(seed, n_time_steps)

# Starting with network stuff
# device="cpu"

n_obs = 2  # TODO
# Input parameters should be the time step and the state
Network = torch.nn.Sequential(
    # torch.nn.Flatten(), #2 -> Flattened
    torch.nn.Linear(n_obs, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.ReLU(),
    # torch.nn.Linear(256, 128),
    # torch.nn.ReLU(),
    torch.nn.Linear(8,4),
    torch.nn.LogSoftmax()
)

optimizer = torch.optim.Adam(Network.parameters(),lr=1e-4,amsgrad=True,weight_decay=0.8e-4)
#1e-4 1e-4  good?
#1e-4 1e-3 bad
#1e-3 1e-3 yeah good
#5e-4 1e-3 lr cant keep up with weight decay?
#5e-4 5e-4 nope

#batchsize 64, 1e-4 1e-4, 512 256 argmax !!!!!! suddenly trash
#batchsize 128, 1e-4 1e-4, 512 256 nö

#trying prob again:
#batchsize 64, 1e-4 1e-4, 512 256 prob kacke
#batchsize 64, 1e-3 1e-3, 512 256 prob kacke

#current:


# define number of training episodes
N_episodes = 400 # total number of training episodes
N_traj = 64  # 128 # number of trajectories in the batch
# preallocate data using arrays initialized with zeros


states = torch.zeros((N_traj, Env.n_time_steps, 2), dtype=torch.float)#,requires_grad=False
actions = torch.zeros((N_traj, Env.n_time_steps), dtype=torch.int64)#,requires_grad=False
returns = torch.zeros((N_traj, Env.n_time_steps), dtype=torch.float)#requires_grad=False)



# mean reward at the end of the episode
mean_final_reward = torch.zeros(N_episodes, dtype=torch.float,requires_grad=False)
# standard deviation of the reward at the end of the episode
std_final_reward = torch.zeros_like(mean_final_reward,requires_grad=False)
# batch minimum at the end of the episode
min_final_reward = torch.zeros_like(mean_final_reward,requires_grad=False)
# batch maximum at the end of the episode
max_final_reward = torch.zeros_like(mean_final_reward,requires_grad=False)

for episode in range(N_episodes):

    for j in range(N_traj):
        optimizer.zero_grad()

        Env.set_state(random=True)
        rewards = torch.zeros((Env.n_time_steps,), dtype=torch.float)#,requires_grad=False

        for time_step in range(Env.n_time_steps):
            # select state
            state = torch.as_tensor(Env.state[:], dtype=torch.float)#.requires_grad_(False)
            states[j, time_step, :] = state[:]


            # select an action according to current policy
            with torch.no_grad():
                pi_s = Network(state)
                #action = np.random.choice(Env.actions, p=pi_s)
                # a = torch.tensor([0,1,2,3])
                # action= a[pi_s.multinomial(num_samples=1)]
                action = torch.argmax(pi_s)
                actions[j, :] = action

                # take an environment step
                _, reward, _ = Env.step(action)

                # store reward
                rewards[time_step] = reward

        # compute reward-to-go
        returns[j, :] = torch.flip(input=torch.cumsum(input=torch.flip(input=rewards,dims=[0]),dim=0),dims=[0])

    # compute gradient and update model
    # compute policy predictions and choose those which were taken
    preds_select = Network(states).gather(2, actions.unsqueeze(2)).squeeze()
    # combute the baseline
    with torch.no_grad():
        baseline = returns.mean(dim=0)

    #add L2 Regularization to avoid over fitting
    Network.state_dict()
        
    loss = -(preds_select * (returns-1*baseline)).sum().mean()

    #torch.nn.utils.clip_grad_value_(Network.parameters(), 100)
    loss.backward()
    #torch.autograd.backward(loss)
    optimizer.step()

    # check performance
    with torch.no_grad():
        mean_final_reward[episode] = returns[:, -1].mean(dim=0)
        std_final_reward[episode] = torch.std(returns[:, -1])
        min_final_reward[episode], max_final_reward[episode] = torch.min(returns[:, -1]), torch.max(returns[:, -1])

    # print results every 10 epochs
    # if episode % 5 == 0:
    print("episode {}".format(episode))
    print("mean reward: {:0.4f}".format(mean_final_reward[episode]))
    print("return standard deviation: {:0.4f}".format(std_final_reward[episode]))
    print("min return: {:0.4f}; max return: {:0.4f}\n".format(min_final_reward[episode], max_final_reward[episode]))

    if(min_final_reward[episode]>0.99):
        break

print(mean_final_reward.mean(),mean_final_reward.max(),mean_final_reward.min())
print(min_final_reward.mean(),min_final_reward.max())
print(std_final_reward.mean(),std_final_reward.min())
# Plotting

# fig = plt.figure(figsize=(5, 5))
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# sphere = Bloch(axes=ax)
#
# def animate(i):
#     sphere.clear()
#     sphere.add_vectors(calc_location(Env.psi_target))
#     sphere.add_vectors(calc_location(psis[i]))
#     if i>0:
#         # x,y,z=zip(calc_location(psis[i]),calc_location(psis[i-1]))
#         # ax.plot3D(x,y,z,'ro--')
#         for j in range(i):
#             sphere.add_points(calc_location(psis[j]))
#     sphere.make_sphere()
#     return ax
#
# def init():
#     sphere.vector_color = ['r']
#     sphere.point_color=['b']
#     return ax
#
#
#
# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     np.arange(len(psis)),
#     init_func=init,
#     blit=False,
#     repeat=False
# )
# HTML(ani.to_jshtml())


# for testing purposes


# Env.set_state(False)
#
# done = False
# j = 0
# psis=np.zeros((n_time_steps,2),dtype=complex)
# psis[0]=Env.psi.copy()
# while j < n_time_steps-1:
#     # pick a random action
#     action = np.random.choice([0, 1, 2, 3])  # equiprobable policy
#     psi_new, reward, done = Env.step(action)
#     j+=1
#     psis[j]=psi_new
#
#     if done:
#         print('\nreached terminal state!')
#         break
# # pick a random action
# action = np.random.choice([0, 1, 2, 3])  # equiprobable policy
# # take an environment step
# state = Env.state.copy()
# state_p, reward, done = Env.step(action)
# print("{}. s={}, a={}, r={}, s'={}\n".format(j, state, action, np.round(reward, 6), state_p))
# j += 1
# if done:
#     print('\nreached terminal state!')
#     break
