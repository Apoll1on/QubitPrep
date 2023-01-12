import numpy as np
from scipy.linalg import expm
import jax.numpy as jnp # jax's numpy version with GPU support
from jax import random # used to define a RNG key to control the random input in JAX
from jax.example_libraries import stax # neural network library
from jax.example_libraries.stax import Dense, Relu, LogSoftmax # neural network layers
from jax import grad
from jax.tree_util import tree_flatten  # jax params are stored as nested tuples; use this to manipulate tuples
from jax.example_libraries import optimizers  # gradient descent optimizers
from jax import jit


# fix seed
seed=0
np.random.seed(seed)

# fix output array
np.set_printoptions(suppress=True,precision=2)


class QubitEnv():
    """
    Gym style environment for RL. You may also inherit the class structure from OpenAI Gym.
    Parameters:
        n_time_steps:   int
                        Total number of time steps within each episode
        seed:   int
                seed of the RNG (for reproducibility)
    """

    def __init__(self, n_time_steps, seed):
        """
        Initialize the environment.

        """

        self.n_time_steps = n_time_steps

        ### define action space variables
        self.n_actions = 4  # action space size
        delta_t = 2 * np.pi / n_time_steps  # set a value for the time step
        # define Pauli matrices
        Id = np.array([[1.0, 0.0], [0.0, +1.0]])
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

        self.action_space = []
        for generator in [Id, sigma_x, sigma_y, sigma_z]:
            self.action_space.append(expm(-1j * delta_t * generator))

        self.action_space[0]=    torc

        self.actions = np.array([0, 1, 2, 3])

        ### define state space variables
        self.S_target = np.array([0.0, 0.0])
        self.psi_target = self.RL_to_qubit_state(self.S_target)

        # set seed
        self.set_seed(seed)
        self.reset()

    def step(self, action):
        """
        Interface between environment and agent. Performs one step in the environemnt.
        Parameters:
            action: int
                    the index of the respective action in the action array
        Returns:
            output: ( object, float, bool)
                    information provided by the environment about its current state:
                    (state, reward, done)
        """

        # apply gate to quantum state
        self.psi = self.action_space[action].dot(self.psi)

        # compute RL state
        self.state = self.qubit_to_RL_state(self.psi)

        # compute reward
        reward = np.abs(self.psi_target.conj().dot(self.psi)) ** 2

        # check if state is terminal
        done = False

        return self.state, reward, done

    def set_seed(self, seed=0):
        """
        Sets the seed of the RNG.

        """
        np.random.seed(seed)

    def reset(self, random=True):
        """
        Resets the environment to its initial values.
        Returns:
            state:  object
                    the initial state of the environment
            random: bool
                    controls whether the initial state is a random state on the sphere or a fixed initial state.
        """

        if random:
            theta = np.pi * np.random.uniform(0.0, 1.0)
            phi = 2 * np.pi * np.random.uniform(0.0, 1.0)
        else:
            # start from south pole of Bloch sphere
            theta = np.pi
            phi = 0.0

        self.state = np.array([theta, phi])
        self.psi = self.RL_to_qubit_state(self.state)

        return self.state

    def render(self):
        """
        Plots the state of the environment. For visulization purposes only. Feel free to ignore.

        """
        pass

    def RL_to_qubit_state(self, s):
        """
        Take as input the RL state s, and return the quantum state |psi>
        """
        theta, phi = s
        psi = np.array([np.cos(0.5 * theta), np.exp(1j * phi) * np.sin(0.5 * theta)])
        return psi

    def qubit_to_RL_state(self, psi):
        """
        Take as input the RL state s, and return the quantum state |psi>
        """
        # take away unphysical global phase
        alpha = np.angle(psi[0])
        psi_new = np.exp(-1j * alpha) * psi

        # find Bloch sphere angles
        theta = 2.0 * np.arccos(psi_new[0]).real
        phi = np.angle(psi_new[1])

        return np.array([theta, phi])

# set seed of rng (for reproducibility of the results)
n_time_steps = 60 # steps of each episode

# create environment and reset it to a random initial state
env=QubitEnv(n_time_steps,seed)
env.reset(random=True)



# set key for the RNG (see JAX docs)
rng = random.PRNGKey(seed)

# define functions which initialize the parameters and evaluate the model
initialize_params, predict = stax.serial(
                                            ### fully connected DNN
                                            Dense(512), # 512 hidden neurons
                                            Relu,
                                            Dense(256), # 256 hidden neurons
                                            Relu,
                                            Dense(env.n_actions), # 4 output neurons
                                            LogSoftmax # NB: computes the log-probability
                                        )

# initialize the model parameters
input_shape = (-1,env.n_time_steps,2) # -1: number of MC points, number of time steps, size of state vector
output_shape, inital_params = initialize_params(rng, input_shape) # fcc layer 28x28 pixes in each image

print('\noutput shape of the policy network is {}.\n'.format(output_shape))


# test network
states=np.ones((3,env.n_time_steps,2), dtype=np.float32)

predictions = predict(inital_params, states)
# check the output shape
print(predictions.shape)

### define loss and accuracy functions

from jax import grad
from jax.tree_util import tree_flatten  # jax params are stored as nested tuples; use this to manipulate tuples


def l2_regularizer(params, lmbda):
    """
    Define l2 regularizer: $\lambda \ sum_j ||theta_j||^2 $ for every parameter in the model $\theta_j$

    """
    return lmbda * jnp.sum(jnp.array([jnp.sum(jnp.abs(theta) ** 2) for theta in tree_flatten(params)[0]]))


def pseudo_loss(params, trajectory_batch):
    """
    Define the pseudo loss function for policy gradient.

    params: object(jax pytree):
        parameters of the deep policy network.
    trajectory_batch: tuple (states, actions, returns) containing the RL states, actions and returns (not the rewards!):
        states: np.array of size (N_MC, env.n_time_steps,2)
        actions: np.array of size (N_MC, env.n_time_steps)
        returns: np.array of size (N_MC, env.n_time_steps)

    Returns:
        -J_{pseudo}(\theta)

    """
    # extract data from the batch
    states, actions, returns = trajectory_batch
    # compute policy predictions
    preds = predict(params, states)
    # combute the baseline
    baseline = jnp.mean(returns, axis=0)
    # select those values of the policy along the action trajectory
    preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=2), axis=2).squeeze()
    # return negative pseudo loss function (want to maximize reward with gradient DEscent)
    return -jnp.mean(jnp.sum(preds_select * (returns - baseline))) + l2_regularizer(params, 0.001)


### define generalized gradient descent optimizer and a function to update model parameters



step_size = 0.001  # step size or learning rate

# compute optimizer functions
opt_init, opt_update, get_params = optimizers.adam(step_size)


# define function which updates the parameters using the change computed by the optimizer
#@jit  # Just In Time compilation speeds up the code; requires to use jnp everywhere; remove when debugging
def update(i, opt_state, batch):
    """
    i: int,
        counter to count how many update steps we have performed
    opt_state: object,
        the state of the optimizer
    batch: np.array
        batch containing the data used to update the model

    Returns:
    opt_state: object,
        the new state of the optimizer

    """
    # get current parameters of the model
    current_params = get_params(opt_state)
    # compute gradients
    grad_params = grad(pseudo_loss)(current_params, batch)
    # use the optimizer to perform the update using opt_update
    return opt_update(i, grad_params, opt_state)


### Train model

import time

# define number of training episodes
N_episodes = 201  # total number of training episodes
N_MC = 64  # 128 # number of trajectories in the batch

# preallocate data using arrays initialized with zeros
state = np.zeros((2,), dtype=np.float32)

states = np.zeros((N_MC, env.n_time_steps, 2), dtype=np.float32)
actions = np.zeros((N_MC, env.n_time_steps), dtype=np.int64)
returns = np.zeros((N_MC, env.n_time_steps), dtype=np.float32)

# mean reward at the end of the episode
mean_final_reward = np.zeros(N_episodes, dtype=np.float32)
# standard deviation of the reward at the end of the episode
std_final_reward = np.zeros_like(mean_final_reward)
# batch minimum at the end of the episode
min_final_reward = np.zeros_like(mean_final_reward)
# batch maximum at the end of the episode
max_final_reward = np.zeros_like(mean_final_reward)

print("\nStarting training...\n")

# set the initial model parameters in the optimizer
opt_state = opt_init(inital_params)

# loop over the number of training episodes
for episode in range(N_episodes):

    ### record time
    start_time = time.time()

    # get current policy  network params
    current_params = get_params(opt_state)

    # MC sample
    for j in range(N_MC):

        # reset environment to a random initial state
        # env.reset(random=False) # fixed initial state
        env.reset(random=True)  # Haar-random initial state (i.e. uniformly sampled on the sphere)

        # zero rewards array (auxiliary array to store the rewards, and help compute the returns)
        rewards = np.zeros((env.n_time_steps,), dtype=np.float32)

        # loop over steps in an episode
        for time_step in range(env.n_time_steps):
            # select state
            state[:] = env.state[:]
            states[j, time_step, :] = state

            # select an action according to current policy
            pi_s = np.exp(predict(current_params, state))
            action = np.random.choice(env.actions, p=pi_s)
            actions[j, time_step] = action

            # take an environment step
            state[:], reward, _ = env.step(action)

            # store reward
            rewards[time_step] = reward

        # compute reward-to-go
        returns[j, :] = jnp.cumsum(rewards[::-1])[::-1]

    # define batch of data
    trajectory_batch = (states, actions, returns)

    # update model
    opt_state = update(episode, opt_state, trajectory_batch)

    ### record time needed for a single epoch
    episode_time = time.time() - start_time

    # check performance
    mean_final_reward[episode] = jnp.mean(returns[:, -1])
    std_final_reward[episode] = jnp.std(returns[:, -1])
    min_final_reward[episode], max_final_reward[episode] = np.min(returns[:, -1]), np.max(returns[:, -1])

    # print results every 10 epochs
    # if episode % 5 == 0:
    print("episode {} in {:0.2f} sec".format(episode, episode_time))
    print("mean reward: {:0.4f}".format(mean_final_reward[episode]))
    print("return standard deviation: {:0.4f}".format(std_final_reward[episode]))
    print("min return: {:0.4f}; max return: {:0.4f}\n".format(min_final_reward[episode], max_final_reward[episode]))