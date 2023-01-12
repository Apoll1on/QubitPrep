import numpy as np
import torch
import math
import random
from collections import namedtuple, deque
from itertools import count

# animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization.bloch import Bloch
from torch import nn

# define Pauli matrices
identity = np.array([[1.0, 0.0], [0.0, +1.0]], dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def calc_location(state):
    # We assume the state is given as an array.
    npstate = np.array(state, dtype=complex)
    x = np.matmul(np.conjugate(npstate), np.matmul(sigma_x, np.transpose(npstate))).real
    y = np.matmul(np.conjugate(npstate), np.matmul(sigma_y, np.transpose(npstate))).real
    z = np.matmul(np.conjugate(npstate), np.matmul(sigma_z, np.transpose(npstate))).real
    return [x, y, z]


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QubitEnv():

    def __init__(self, seed, n_timesteps, error=10e-6):
        self.err = error
        self.delta_t = 2 * np.pi / n_timesteps
        self.n_actions = 4
        self.action_space = []
        for generator in [identity, sigma_x, sigma_y, sigma_z]:
            self.action_space.append(torch.linalg.matrix_exp(-1j * self.delta_t * torch.from_numpy(generator)))
        print(self.action_space)
        self.actions = [0, 1, 2, 3]

        self.state = self.init_state(random=True)
        self.psi = self.state_to_psi(self.state)

        self.target_state = torch.tensor([1, 0], dtype=torch.cdouble)
        self.target_psi = self.state_to_psi(self.target_state)

        torch.manual_seed(seed)

    def step(self, action, nstep):
        self.psi = torch.matmul(self.action_space[action], self.psi)
        self.state = self.psi_to_state(self.psi)
        reward_ = torch.abs(torch.dot(torch.conj(self.psi), self.target_psi)) ** 2
        reward_ = torch.tensor((reward_), dtype=torch.float)
        done = False

        if torch.abs(1 - reward_) < self.err or nstep * self.delta_t >= 2 * np.pi:
            done = True
        return self.state, reward_, done

    def init_state(self, random=True):
        if random:
            theta = np.pi * torch.rand(1)
            phi = 2 * np.pi * torch.rand(1)
        else:
            # start from south pole of Bloch sphere
            theta = np.pi
            phi = 0.0

        self.state = torch.tensor([theta, phi], dtype=torch.double)
        self.psi = self.state_to_psi(self.state)
        return self.state

    def state_to_psi(self, s):
        """Take as input the RL state s, and return the quantum state |psi>"""
        theta, phi = s
        psi = torch.tensor([torch.cos(0.5 * theta), torch.exp(1j * phi) * torch.sin(0.5 * theta)], dtype=torch.cdouble)
        return psi

    def psi_to_state(self, psi):
        """
        Take as input the RL state s, and return the quantum state |psi>
        """
        # take away unphysical global phase
        alpha = torch.angle(psi[0])
        psi_new = torch.exp(-1j * alpha) * psi

        # find Bloch sphere angles
        theta = 2.0 * torch.arccos(psi_new[0]).real
        phi = torch.angle(psi_new[1])

        return torch.tensor([theta, phi], dtype=torch.double)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16,16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return nn.functional.relu(self.layer3(x))


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 750
TAU = 0.005
LR = 1e-3
device = "cpu"

env = QubitEnv(0, 50)
n_observations = 2
n_actions = env.n_actions

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action_testing(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return torch.tensor([[torch.argmax(target_net(state))]], device=device, dtype=torch.long)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.tensor([[torch.argmax(policy_net(state))]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[np.random.choice(env.actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # # Compute a mask of non-final states and concatenate the batch elements
    # # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                         batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.stack([s for s in batch.next_state
    #                                      if s is not None])
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = torch.argmax(target_net(non_final_next_states))

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_states = torch.stack(batch.next_state)
    with torch.no_grad():
        next_state_values = torch.argmax(target_net(next_states))

    # Compute the expected Q values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



num_episodes = 5000

steps_needed = []


for i_episode in range(num_episodes):
    print(i_episode)
    state = env.init_state(True)
    state = torch.tensor(state, dtype=torch.float32, device=device)
    for t in count():
        action = select_action(state)
        observation, reward, done = env.step(action, t)
        reward = torch.tensor([reward], device=device)

        #
        # if done:
        #     next_state = None
        # else:
        #     next_state = torch.tensor(observation, dtype=torch.float32, device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            steps_needed.append(t)
            break

target_net.eval()
policy_net.eval()

steps = np.array(steps_needed)
plt.figure(1)
plt.title('Result')
plt.xlabel('Episode')
plt.ylabel('Steps needed')
plt.plot(steps)
# Take 100 episode averages and plot them too
means = np.cumsum(steps) / np.arange(1, len(steps) + 1)
plt.plot(means)
plt.show()

n_testing = 150
teststeps = np.empty(n_testing)
for i in range(n_testing):
    state = env.init_state(random=True)
    state = torch.tensor(state, dtype=torch.float32)
    j = 0
    while True:
        action = select_action_testing(state)
        state, _, done = env.step(action, nstep=j)
        state = torch.tensor(state, dtype=torch.float32)
        j = j + 1
        if done:
            break
    teststeps[i] = j

plt.figure(1)
plt.title('Testing')
plt.xlabel('Episode')
plt.ylabel('Steps needed')
plt.plot(teststeps)
# Take 100 episode averages and plot them too
means = np.cumsum(teststeps) / np.arange(1, len(teststeps) + 1)
plt.plot(means)
plt.show()

torch.save(target_net.state_dict(), "targetstatedict.pt")
