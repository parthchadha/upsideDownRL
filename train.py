import numpy as np
import gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pdb
from collections import deque
from sortedcontainers import SortedDict
import random

# If an agent is in a given state and desires a given return over a given horizon,
# which action should it take next?

# Input state s_t and command c_t=(dr_t, dh_t) 
# where dr_t is the desired return and dh_t is desired time horizon 

class BehaviorFunc(nn.Module):
	def __init__(self, state_size, action_size, args):
		super(BehaviorFunc, self).__init__()
		self.args = args
		self.fc1 = nn.Linear(state_size, 128)
		self.fc2 = nn.Linear(2, 128)
		self.fc3 = nn.Linear(128, action_size)
		self.command_scale = args.command_scale

	def forward(self, state, desired_return, desired_horizon):
		x = torch.sigmoid(self.fc1(state))
		concat_command = torch.cat((desired_return, desired_horizon), 1)*self.command_scale
		y = torch.sigmoid(self.fc2(concat_command))
		x = x * y
		return F.softmax(self.fc3(x))

class UpsideDownRL(object):
	def __init__(self, env, args):
		super(UpsideDownRL, self).__init__()
		self.env = env
		self.args = args
		self.nb_actions  = self.env.action_space.n
		self.state_space = self.env.observation_space.shape[0]
		self.experience	 = SortedDict()
		self.B = BehaviorFunc(self.state_space, self.nb_actions, args).cuda()
		self.use_random_actions = True

	def gen_episode(self, dr, dh):
		state = self.env.reset()
		episode_data = []
		states = []
		rewards = []
		actions = []
		total_reward = 0
		while True:
			action = self.select_action(state, dr, dh)#np.random.randint(self.nb_actions)
			next_state, reward, is_terminal, _ = self.env.step(action)
			if self.args.render:
				self.env.render()

			states.append(state)
			actions.append(action)
			rewards.append(reward)
			total_reward += reward
			state = next_state
			dr = dr - reward
			dh = max(dh - 1, 1)
			if is_terminal:
				break

		return total_reward, states, actions, rewards

	def fill_replay_buffer(self):
		dr, dh = self.get_desired_return_and_horizon()
		self.experience.clear()
		for i in range(self.args.replay_buffer_capacity):
			total_reward, states, actions, rewards = self.gen_episode(dr, dh)
			self.experience.__setitem__(total_reward, (states, actions, rewards))

		if self.args.verbose:
			if self.use_random_actions:
				print("Filled replay buffer with random actions")
			else:
				print("Filled replay buffer using BehaviorFunc")
		self.use_random_actions = False

	def select_action(self, state, desired_return=None, desired_horizon=None):
		if self.use_random_actions:
			action = np.random.randint(self.nb_actions)
		else:
			action_prob = self.B(torch.from_numpy(state).cuda(), 
									torch.from_numpy(np.array(desired_return, dtype=np.float32)).reshape(-1, 1).cuda(), 
									torch.from_numpy(np.array(desired_horizon, dtype=np.float32).reshape(-1, 1)).cuda()
								)

			# create a categorical distribution over action probabilities
			dist = Categorical(action_prob)
			action = dist.sample().item()
		return action

	def get_desired_return_and_horizon(self):
		if (self.use_random_actions):
			return 0, 0

		h = []
		r = []
		for i in range(self.args.explore_buffer_len):
			episode = self.experience.popitem() #will return in sorted order
			h.append(len(episode[1][0]))
			r.append(episode[0])

		mean_horizon_len = np.mean(h)
		mean_reward = np.random.uniform(low=np.mean(r), high=np.mean(r)+np.std(r))
		return mean_reward, mean_horizon_len

	def trainBehaviorFunc(self):
		experience_dict = dict(self.experience)
		experience_values = list(experience_dict.values())		
		for i in range(self.args.train_iter):
			state = []
			dr = []
			dh = []
			indices = np.random.choice(len(experience_values), self.args.batch_size, replace=True)
			train_episodes = [experience_values[i] for i in indices]
			t1 = [np.random.choice(len(e[0])-2, 1)  for e in train_episodes]
			
			for pair in zip(t1, train_episodes):
				state.append(pair[1][0][pair[0][0]])
				dr.append(np.sum(pair[1][2][pair[0][0]:]))
				dh.append(len(pair[1][0])-pair[0][0])				

		
			state = torch.from_numpy(np.array(state)).cuda()
			dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1,1)).cuda()
			dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1,1)).cuda()
			output = self.B(state, dr, dh)

	def evaluate(self):
		testing_rewards = []
		testing_steps  = []
		for i in range(self.args.evaluate_trials):
			dr, dh = self.get_desired_return_and_horizon()
			total_reward, states, actions, rewards = self.gen_episode(dr, dh)
			testing_rewards.append(total_reward)
			testing_steps.append(len(rewards))

		print("Mean reward achieved : {}".format(np.mean(testing_rewards)))
		

	def train(self):
		self.fill_replay_buffer()
		iterations = 0
		while True:
			self.trainBehaviorFunc()
			self.fill_replay_buffer()

			if iterations % self.args.eval_every_k_epoch == 0:
				self.evaluate()

			iterations += 1



def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for UpsideDown RL")
	parser.add_argument("--render", action='store_true')
	parser.add_argument("--verbose", action='store_true')
	parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--command_scale", type=float, default=0.01)
	parser.add_argument("--replay_buffer_capacity", type=int, default=53)#1000)
	parser.add_argument("--explore_buffer_len", type=int, default=3)
	parser.add_argument("--eval_every_k_epoch", type=int, default=1)
	parser.add_argument("--evaluate_trials", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--train_iter", type=int, default=25)

	parser.add_argument("--exp_name", type=str)
	
	args = parser.parse_args()


	env = gym.make("LunarLander-v2")
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	print("created agent")
	agent = UpsideDownRL(env, args)
	agent.train()
	

	env.close()


if __name__ == "__main__":
	main()