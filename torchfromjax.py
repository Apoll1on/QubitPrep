import torch

# animation
import numpy as np


identity = np.array([[1.0, 0.0], [0.0, +1.0]], dtype=np.complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex)

class QubitEnv:

    def __init__(self,seed,n_timesteps,error=10e-6):
        self.err=error
        self.delta_t=2*np.pi/n_timesteps
        self.n_actions=4
        self.action_space = []
        for generator in [identity, sigma_x, sigma_y, sigma_z]:
            self.action_space.append(torch.linalg.matrix_exp(-1j * self.delta_t * torch.from_numpy(generator)))

        print(self.action_space)
        self.actions=[0,1,2,3]
        torch.manual_seed(seed)

        self.state=self.init_state(random=True)
        self.psi=self.state_to_psi(self.state)

        self.target_state=torch.tensor([1,0],dtype=torch.cdouble)
        self.target_psi=self.state_to_psi(self.target_state)



    def step(self,action,nstep):
        self.psi=torch.matmul(self.action_space[action],self.psi)
        self.state=self.psi_to_state(self.psi)
        reward_=torch.abs(torch.dot(self.psi,self.target_psi))**2
        reward_=torch.tensor((reward_),dtype=torch.float)
        done=False

        if torch.abs(1-reward_)<self.err or nstep*self.delta_t>=2*np.pi:
            done=True
        return self.state,reward_,done

    def init_state(self, random=True):
        if random:
            theta = np.pi*torch.rand(1)
            phi = 2*np.pi*torch.rand(1)
        else:
            # start from south pole of Bloch sphere
            theta = np.pi
            phi = 0.0

        self.state = torch.tensor([theta, phi], dtype=torch.double)
        self.psi = self.state_to_psi(self.state)
        return self.state

    def state_to_psi(self,s):
        """Take as input the RL state s, and return the quantum state |psi>"""
        theta, phi = s
        psi = torch.tensor([torch.cos(0.5 * theta), torch.exp(1j * phi) * torch.sin(0.5 * theta)],dtype=torch.cdouble)
        return psi

    def psi_to_state(self,psi):
        """
        Take as input the RL state s, and return the quantum state |psi>
        """
        # take away unphysical global phase
        alpha = torch.angle(psi[0])
        psi_new = torch.exp(-1j * alpha) * psi

        # find Bloch sphere angles
        theta = 2.0 * torch.arccos(psi_new[0]).real
        phi = torch.angle(psi_new[1])

        return torch.tensor([theta, phi],dtype=torch.double)


BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-4
device="cpu"

env=QubitEnv(0,50)
n_observations=2
n_actions = 4

policyNetwork = torch.nn.Sequential(
    torch.nn.Linear(n_observations,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,n_actions),
    torch.nn.LogSoftmax()
).to(device)

targetNetwork = torch.nn.Sequential(
    torch.nn.Linear(n_observations,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,n_actions),
    torch.nn.LogSoftmax()
).to(device)

targetNetwork.load_state_dict(policyNetwork.state_dict())
policyNetwork.train()
targetNetwork.train()

