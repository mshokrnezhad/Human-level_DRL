import numpy as np
import torch as T
from DQN import DeepQNetwork
from replay_memory import ReplayBuffer


class Agent():
    def __init__(self, input_dims, n_actions, lr, mem_size, batch_size, epsilon, gamma=0.99, eps_dec=5e-7,
                 eps_min=0.01, replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.replace = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name+" "+self.algo+"_q_eval",
                                   checkpoint_dir=self.checkpoint_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name+" "+self.algo+"_q_next",
                                   checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(
                self.q_eval.device)  # converting observation to tensor,
            # and observation is in the list because our convolution expects an input tensor of shape batch size
            # by input dims.
            q_values = self.q_eval.forward(state)
            action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, resulted_state, done):
        self.memory.store_transition(
            state, action, reward, resulted_state, done)

    def sample_memory(self):
        state, action, reward, resulted_state, done = self.memory.sample_buffer(
            self.batch_size)
        state = T.tensor(state).to(self.q_eval.device)
        reward = T.tensor(reward).to(self.q_eval.device)
        done = T.tensor(done).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        resulted_state = T.tensor(resulted_state).to(self.q_eval.device)

        return state, reward, done, action, resulted_state

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, reward, done, action, resulted_state = self.sample_memory()

        indexes = np.arange(self.batch_size, dtype=np.longlong)
        action = action.long()
        done = done.bool()
        
        prediction = self.q_eval.forward(state)[indexes, action]  # dims: batch_size * n_actions

        next_result = self.q_next.forward(resulted_state).max(dim=1)[0]
        next_result[done] = 0.0  # for terminal states, target should be reward
        target = reward + self.gamma * next_result

        loss = self.q_eval.loss(target, prediction).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()
