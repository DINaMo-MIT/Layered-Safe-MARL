import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import csv
import torch.nn as nn
import cv2
from collections import defaultdict

def _t2n(x):
	return x.detach().cpu().numpy()

class GMPERunner(Runner):
	"""
		Runner class to perform training, evaluation and data 
		collection for the MPEs. See parent class for details
	"""
	dt = 0.1
	def __init__(self, config):
		super(GMPERunner, self).__init__(config)
		self.horizon = self.all_args.horizon
		self.action_space_type = self.envs.action_space[0].__class__.__name__

		# for training.
		self.num_total_episode = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
		self.episode_info_list = [{} for _ in range(self.n_rollout_threads)]
		# episode info collected from each GraphSubprocVecEnv
		self.episode_travel_time_mean = self.episode_length
		self.episode_travel_distance_mean = 0
		self.done_percentage = 0
		self.episode_num_reached_goal_mean = 0
		self.episode_conflict_percentage_mean = 0
		self.episode_min_distance_mean = 0
		self.episode_min_distance_min = 0
		self.episode_multiple_engagement = 0


	def run(self):

		self.warmup()

		start = time.time()
		num_total_episode = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
		print("number of total episode: ", num_total_episode)

		# This is where the episodes are actually run.
		for episode in range(num_total_episode):


			if self.use_linear_lr_decay:
				self.trainer.policy.lr_decay(episode, num_total_episode)
			flag = 0
			self.active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
									dtype=np.int32)
			for step in range(self.episode_length):




				if self.all_args.use_masking:
									# Sample actions
					if step == 0:
						# active_agents = None
						# active_mask = None
						# finished = None
						values, actions, action_log_probs, rnn_states, \
							rnn_states_critic, actions_env = self.collect(step)
						obs, agent_id, node_obs, adj, rewards, \
							dones, infos = self.envs.step(actions_env, episode)
						# # For no-Collab uncomment the line
						rewards = rewards[:,:, np.newaxis]
						finished,finished_list = self.get_finished(dones)
						self.active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)

						dones_env = np.all(dones, axis=1)
						self.active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
												self.num_agents, 1), dtype=np.float32)
						available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.envs.action_space[0].n), dtype=np.float32)

						data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
								dones, infos, values, actions, action_log_probs, 
								rnn_states, rnn_states_critic, available_actions)
						# insert data into buffer
						self.insert_with_mask(data)
					else:
						
				
						ret = []
						for e in range(self.n_rollout_threads):
							for a in range(self.all_args.num_agents):
								if self.active_masks[e, a,0]:
									ret.append((e, a, self.active_masks[e, a,0]))
						self.active_agents = ret

						values, actions, action_log_probs, rnn_states, \
							rnn_states_critic, actions_env, available_actions = self.collect_with_mask(step,self.active_agents,self.active_masks,finished)

						obs, agent_id, node_obs, adj, rewards, \
							dones, infos = self.envs.step(actions_env, episode)

						# Calculate the number of elements to update
						num_elements = np.sum(dones.astype(int))

						# Create an array of zeros with the correct shape
						zeros_array = np.zeros((num_elements, 1), dtype=np.float32)

						# Reshape dones to match the shape of self.active_masks
						reshaped_dones = dones[..., np.newaxis]

						# Use broadcasting to update self.active_masks
						if not reshaped_dones.all():
							self.active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
										dtype=np.int32)
						self.active_masks[reshaped_dones] = 0

						finished,finished_list = self.get_finished(dones)

						dones_env = np.all(dones, axis=1)
						self.active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
												self.num_agents, 1), dtype=np.float32)



						# # For no-Collab uncomment the line
						rewards = rewards[:,:, np.newaxis]
						data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
								dones, infos, values, actions, action_log_probs, 
								rnn_states, rnn_states_critic, available_actions)
						# insert data into buffer
						self.insert_with_mask(data, active_agents=self.active_agents,active_masks=self.active_masks,finished=finished)

				else:
					# Sample actions
					values, actions, action_log_probs, rnn_states, \
						rnn_states_critic, actions_env = self.collect(step)

					# Obs reward and next obs
					obs, agent_id, node_obs, adj, rewards, \
						dones, infos = self.envs.step(actions_env)

					# # For no-Collab uncomment the line
					rewards = rewards[:,:, np.newaxis]
				
					data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
							dones, infos, values, actions, action_log_probs, 
							rnn_states, rnn_states_critic)

					# insert data into buffer
					self.insert(data) 

				for (i_thread, info_thread) in enumerate(infos):
					if len(info_thread) > self.num_agents:
						# this means the thread is resetting and attached env_info to the original info.
						self.episode_info_list[i_thread] = info_thread[self.num_agents]
			# compute return and update network
			self.compute()

			self.parse_episode_info(self.episode_info_list)
			# flush after parsing info.
			self.episode_info_list = [{} for _ in range(self.n_rollout_threads)]
			train_infos = self.train()
			train_infos["average_episode_travel_time"] = self.episode_travel_time_mean
			train_infos["average_episode_travel_distance"] = self.episode_travel_distance_mean
			train_infos["done_percentage"] = self.done_percentage
			train_infos["average_episode_num_reached_goal"] = self.episode_num_reached_goal_mean
			train_infos["average_episode_conflict_percentage"] = self.episode_conflict_percentage_mean
			train_infos["average_episode_min_distance_mean"] = self.episode_min_distance_mean
			train_infos["average_episode_min_distance_min"] = self.episode_min_distance_min
			train_infos["average_episode_multiple_engagement_percentage"] = self.episode_multiple_engagement
			# TODO: More info.

			# post process
			total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
			# save model
			if self.checkpoint_interval > 0:
				if (episode % self.checkpoint_interval == 0) and episode > 0:
					self.save(self.checkpoint_dir, episode)
			if (episode % self.save_interval == 0 or episode == num_total_episode - 1):
				self.save()

			# log information
			if episode % self.log_interval == 0:
				end = time.time()
				env_infos = self.process_infos(infos)
				avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
				train_infos["average_episode_rewards"] = avg_ep_rew
				print(f"Avg Ep rewards: {avg_ep_rew:.2f} | "
					f"avg travel time: {self.episode_travel_time_mean:.2f} "
					f"distance: {self.episode_travel_distance_mean:.2f} "
					f"done (%): {100 * self.done_percentage:.1f} "
					f"num reached goal: {self.episode_num_reached_goal_mean:.2f} "
					f"conflict (%): {100 * self.episode_conflict_percentage_mean:.1f} "
					f"min distance: {self.episode_min_distance_mean:.2f} "
					f"| Total env steps: {total_num_steps} "
					f"training complete {total_num_steps / self.num_env_steps * 100:.3f}")
				for agent_id in range(self.num_agents):
					str1 = f'agent{agent_id}/distance_mean'
					str2 = f'agent{agent_id}/distance_variance'
					str3 = f'agent{agent_id}/mean_variance'
					str4 = f'agent{agent_id}/dist_to_goal'
					str5 = f'agent{agent_id}/individual_rewards'
					str6 = f'agent{agent_id}/num_agent_collisions'
					print(str1, str2, str3)
					print(f"mean: {env_infos[str1][0]:.3f} \t"
						f"Var: {env_infos[str2][0]:.3f} \t "
						f"Mean/Var: {env_infos[str3][0]:.3f} \t "
						f"Dist2goal: {env_infos[str4][0]:.3f} \t "
						f"Rew: {env_infos[str5][0]:.3f} \t "
						f"Col: {env_infos[str6][0]:.3f} \t ")
				self.log_train(train_infos, total_num_steps)
				self.log_env(env_infos, total_num_steps)


			# eval
			if episode % self.eval_interval == 0 and self.use_eval:
				self.eval(total_num_steps)

	def parse_episode_info(self, prev_episode_info):
		# prev_episode_info is a tuple of dictionaries.
		# collect all values in a numpy array and take mean.
		travel_time_mean_list = []
		travel_distance_mean_list = []
		done_percentage_list = []
		num_reached_goal_mean_list = []
		conflict_percentage_list = []
		min_distance_mean_list =[]
		min_distance_min_list = []
		multiple_engagement_list = []
		for thread_episode_info in prev_episode_info:
			travel_time_mean_list.append(thread_episode_info['travel_time_mean'])
			travel_distance_mean_list.append(thread_episode_info['travel_distance_mean'])
			done_percentage_list.append(thread_episode_info['done_percentage'])
			num_reached_goal_mean_list.append(thread_episode_info['num_reached_goal_mean'])
			conflict_percentage_list.append(thread_episode_info['conflict_percentage'])
			min_distance_mean_list.append(thread_episode_info['min_distance_mean'])
			min_distance_min_list.append(thread_episode_info['min_distance_min'])
			multiple_engagement_list.append(thread_episode_info['multiple_engagement_percentage'])
		# print("travel_time_mean_list: ", travel_time_mean_list)
		# print("travel_distance_mean_list: ", travel_distance_mean_list)
		self.episode_travel_time_mean = np.mean(travel_time_mean_list)
		self.episode_travel_distance_mean = np.mean(travel_distance_mean_list)
		self.done_percentage = np.mean(done_percentage_list)
		self.episode_num_reached_goal_mean = np.mean(num_reached_goal_mean_list)
		self.episode_conflict_percentage_mean = np.mean(conflict_percentage_list)
		self.episode_min_distance_mean = np.mean(min_distance_mean_list)
		self.episode_min_distance_min = np.min(min_distance_min_list)
		self.episode_multiple_engagement = np.mean(multiple_engagement_list)

	def warmup(self):
		# reset env
		try:
			obs, agent_id, node_obs, adj, prev_episode_info = self.envs.reset()
			self.parse_episode_info(prev_episode_info)
		except Exception as e:
			print(f"Error during environment reset: {e}")
	
		# replay buffer
		if self.use_centralized_V:
			# (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
			share_obs = obs.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
			share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, 
																	axis=1)
			# (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
			share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
			share_agent_id = np.expand_dims(share_agent_id, 
											1).repeat(self.num_agents, axis=1)
		else:
			share_obs = obs
			share_agent_id = agent_id


		self.buffer.share_obs[0] = share_obs.copy()
		self.buffer.obs[0] = obs.copy()
		self.buffer.node_obs[0] = node_obs.copy()
		self.buffer.adj[0] = adj.copy()
		self.buffer.agent_id[0] = agent_id.copy()
		self.buffer.share_agent_id[0] = share_agent_id.copy()




	def get_finished(self,dones):
		finished=[]
		f=[]
		bools=[]
		current_env=0
		for env in range(len(dones)) :
			if current_env != env:
				current_env = env
				finished.append(f)
				f=[]
			for agent in range(len(dones[env])):
				if dones[env][agent]==True:
					f.append(agent)
					bools.append(True)
					
				else:
					bools.append(False)
		finished.append(f)
		return finished,bools


	@torch.no_grad()
	def collect_with_mask(self, step:int,active_agents,active_masks,finished) -> Tuple[arr, arr, arr, arr, arr, arr, arr]:
		self.trainer.prep_rollout()
		flag = False
        ### look and see who is active and force the avilable actions to be limited 
		all_actions=[]
		avail_actions_list=[]
		envs_aa=[]
		for env in range(self.n_rollout_threads): #len(active_mask)):
			avail_actions_list=[]
			for a in range(self.all_args.num_agents):
				
					if a in finished[env]:
						available_actions = np.zeros((self.envs.action_space[0].n))
						available_actions[int(self.envs.action_space[0].n/2)] = 1 ## TODO: Find a better representation for stop action
						flag= True
					else:
						available_actions=np.ones((self.envs.action_space[0].n))
						
					avail_actions_list.append(available_actions)
					
			envs_aa.append(avail_actions_list)
					
		aa= np.asarray(envs_aa)

		ab = np.array([[[1., 1., 1., 1., 1.],
  [2. ,2., 1., 1., 1.],
  [3., 3., 1. ,1. ,1.]],

 [[4., 4., 1., 1., 1.],
  [5., 5., 1., 1., 1.],
  [6., 6., 1., 1., 1.]]])

		value, action, action_log_prob, rnn_states, rnn_states_critic \
			= self.trainer.policy.get_actions(
						np.concatenate(self.buffer.share_obs[step]),
						np.concatenate(self.buffer.obs[step]),
						np.concatenate(self.buffer.node_obs[step]),
						np.concatenate(self.buffer.adj[step]),
						np.concatenate(self.buffer.agent_id[step]),
						np.concatenate(self.buffer.share_agent_id[step]),
						np.concatenate(self.buffer.rnn_states[step]),
						np.concatenate(self.buffer.rnn_states_critic[step]),
						np.concatenate(self.buffer.masks[step]),
						available_actions = np.reshape(aa, (self.all_args.num_agents*self.n_rollout_threads, self.envs.action_space[0].n), order='C'))

		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob), 
											self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states), 
								self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 
											self.n_rollout_threads))
		# rearrange action
		if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
			for i in range(self.envs.action_space[0].shape):
				uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 
															1)[actions[:, :, i]]
				if i == 0:
					actions_env = uc_actions_env
				else:
					actions_env = np.concatenate((actions_env, 
												uc_actions_env), axis=2)
		elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
			actions_env = np.squeeze(np.eye(
									self.envs.action_space[0].n)[actions], 2)
		else:
			raise NotImplementedError

		if flag:
			avail_actions = aa
		else:
			avail_actions_list=[]

			for env in range(len(active_masks)):
				aa=[]
				for a in range(len(active_masks[env])):
					available_actions=np.ones((self.envs.action_space[0].n))
					aa.append(available_actions)
				avail_actions_list.append(aa)

			
			avail_actions= np.asarray(avail_actions_list)

		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env, avail_actions)


	@torch.no_grad()
	def collect(self, step:int) -> Tuple[arr, arr, arr, arr, arr, arr]:
		self.trainer.prep_rollout()
		value, action, action_log_prob, rnn_states, rnn_states_critic \
			= self.trainer.policy.get_actions(
							np.concatenate(self.buffer.share_obs[step]),
							np.concatenate(self.buffer.obs[step]),
							np.concatenate(self.buffer.node_obs[step]),
							np.concatenate(self.buffer.adj[step]),
							np.concatenate(self.buffer.agent_id[step]),
							np.concatenate(self.buffer.share_agent_id[step]),
							np.concatenate(self.buffer.rnn_states[step]),
							np.concatenate(self.buffer.rnn_states_critic[step]),
							np.concatenate(self.buffer.masks[step]))
		# [self.envs, agents, dim]
		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob), 
											self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states), 
								self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 
											self.n_rollout_threads))
		# rearrange action
		if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
			for i in range(self.envs.action_space[0].shape):
				uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 
															1)[actions[:, :, i]]
				if i == 0:
					actions_env = uc_actions_env
				else:
					actions_env = np.concatenate((actions_env, 
												uc_actions_env), axis=2)
		elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
			actions_env = np.squeeze(np.eye(
									self.envs.action_space[0].n)[actions], 2)
			
		elif self.envs.action_space[0].__class__.__name__ == 'Box':
			# handling the "Box" action space
			# For example, can clip the actions to fit within the box boundaries and scale them to match the range of the action space
			actions_env = actions
		else:
			raise NotImplementedError
		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env)

	def insert(self, data):
		obs, agent_id, node_obs, adj, agent_id, rewards, dones, \
			infos, values, actions, action_log_probs, \
			rnn_states, rnn_states_critic = data

		dones_env = np.all(dones, axis=1)
		rnn_states[dones] = np.zeros(((dones).sum(),
												self.recurrent_N, 
												self.hidden_size), 
												dtype=np.float32)
		rnn_states_critic[dones] = np.zeros(((dones).sum(),
										*self.buffer.rnn_states_critic.shape[3:]), 
										dtype=np.float32)
		masks = np.ones((self.n_rollout_threads, 
						self.num_agents, 1), 
						dtype=np.float32)

		masks[dones] = np.zeros(((dones).sum(), 1), 
										dtype=np.float32)
		active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
							   dtype=np.float32)
		active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)
		active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
					    		self.num_agents, 1), dtype=np.float32)

		# if centralized critic, then shared_obs is concatenation of obs from all agents
		if self.use_centralized_V:
			# TODO stack agent_id as well for agent specific information
			# (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
			share_obs = obs.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
			share_obs = np.expand_dims(share_obs, 
										1).repeat(self.num_agents, axis=1)
			# (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
			share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
			share_agent_id = np.expand_dims(share_agent_id, 
											1).repeat(self.num_agents, axis=1)
		else:
			share_obs = obs
			share_agent_id = agent_id
		self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, 
						rnn_states, rnn_states_critic, actions, action_log_probs, 
						values, rewards, masks,None, active_masks=active_masks)


	def insert_with_mask(self, data,active_agents=None,active_masks = None, finished=None):
		obs, agent_id, node_obs, adj, agent_id, rewards, dones, \
			infos, values, actions, action_log_probs, \
			rnn_states, rnn_states_critic, available_actions = data


		dones_env = np.all(dones, axis=1)
		rnn_states[dones] = np.zeros(((dones).sum(),
												self.recurrent_N, 
												self.hidden_size), 
												dtype=np.float32)
		rnn_states_critic[dones] = np.zeros(((dones).sum(),
										*self.buffer.rnn_states_critic.shape[3:]), 
										dtype=np.float32)
		masks = np.ones((self.n_rollout_threads, 
						self.num_agents, 1), 
						dtype=np.float32)


		masks[dones] = np.zeros(((dones).sum(), 1), 
										dtype=np.float32)

		active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
							   dtype=np.float32)
		active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)
		active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
								self.num_agents, 1), dtype=np.float32)

		# if centralized critic, then shared_obs is concatenation of obs from all agents
		if self.use_centralized_V:
			# TODO stack agent_id as well for agent specific information
			# (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
			share_obs = obs.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
			share_obs = np.expand_dims(share_obs, 
										1).repeat(self.num_agents, axis=1)
			# (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
			share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
			share_agent_id = np.expand_dims(share_agent_id, 
											1).repeat(self.num_agents, axis=1)
		else:
			share_obs = obs
			share_agent_id = agent_id

		self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, 
						rnn_states, rnn_states_critic, actions, action_log_probs, 
						values, rewards, masks,active_masks=active_masks,available_actions=available_actions)


	@torch.no_grad()
	def compute(self):
		"""Calculate returns for the collected data."""
		self.trainer.prep_rollout()
		next_values = self.trainer.policy.get_values(
							np.concatenate(self.buffer.share_obs[-1]),
							np.concatenate(self.buffer.node_obs[-1]),
							np.concatenate(self.buffer.adj[-1]),
							np.concatenate(self.buffer.share_agent_id[-1]),
							np.concatenate(self.buffer.rnn_states_critic[-1]),
							np.concatenate(self.buffer.masks[-1]))
		next_values = np.array(np.split(_t2n(next_values), 
								self.n_rollout_threads))
		self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

	@torch.no_grad()
	def eval(self, total_num_steps:int):     
		eval_episode_rewards = []
		eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

		eval_rnn_states = np.zeros((self.n_eval_rollout_threads, 
									*self.buffer.rnn_states.shape[2:]), 
									dtype=np.float32)
		eval_masks = np.ones((self.n_eval_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)

		for eval_step in range(self.episode_length):
			self.trainer.prep_rollout()
			eval_action, eval_rnn_states = self.trainer.policy.act(
												np.concatenate(eval_obs),
												np.concatenate(eval_node_obs),
												np.concatenate(eval_adj),
												np.concatenate(eval_agent_id),
												np.concatenate(eval_rnn_states),
												np.concatenate(eval_masks),
												deterministic=True)
			eval_actions = np.array(np.split(_t2n(eval_action), 
											self.n_eval_rollout_threads))
			eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 
											self.n_eval_rollout_threads))
			
			if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
				for i in range(self.eval_envs.action_space[0].shape):
					eval_uc_actions_env = np.eye(
								self.eval_envs.action_space[0].high[i] + 
														1)[eval_actions[:, :, i]]
					if i == 0:
						eval_actions_env = eval_uc_actions_env
					else:
						eval_actions_env = np.concatenate((eval_actions_env, 
															eval_uc_actions_env), 
															axis=2)
			elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
				eval_actions_env = np.squeeze(np.eye(
							self.eval_envs.action_space[0].n)[eval_actions], 2)
			else:
				raise NotImplementedError

			# Obser reward and next obs
			eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_rewards, \
				eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
			eval_episode_rewards.append(eval_rewards)
			eval_dones_env = np.all(eval_dones, axis=1)

			eval_rnn_states[eval_dones_env] = np.zeros((
													(eval_dones_env == True).sum(), 
													self.recurrent_N, 
													self.hidden_size), 
													dtype=np.float32)
			eval_masks = np.ones((self.n_eval_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)
			eval_masks[eval_dones_env] = np.zeros((
												(eval_dones_env == True).sum(), 1), 
												dtype=np.float32)

		eval_episode_rewards = np.array(eval_episode_rewards)
		eval_env_infos = {}
		eval_env_infos['eval_average_episode_rewards'] = np.sum(
												np.array(eval_episode_rewards), 
												axis=0)
		eval_average_episode_rewards = np.mean(
									eval_env_infos['eval_average_episode_rewards'])
		print("eval average episode rewards of agent: " + 
											str(eval_average_episode_rewards))
		self.log_env(eval_env_infos, total_num_steps)

	def save_images(self, img_list, filename):

		for i, img in enumerate(img_list):
			imageio.imwrite(filename + str(i) + '.png', img)


	@staticmethod
	def map_action_index_to_one_hot_vector(action_index, num_actions):
		"""
			Map the action index to a one hot vector
			action_index: array of action index: (n_rollout_thread, num_agents, num_timestep)
		"""
		num_threads = action_index.shape[0]
		num_agents = action_index.shape[1]
		num_timestep = action_index.shape[2]
		shape = action_index.shape + (num_actions,)
		action = np.zeros(shape)
		indices = np.indices(action_index.shape)
		action[indices[0], indices[1], indices[2], action_index] = 1
		return action

	@torch.no_grad()
	def render(self, get_metrics:bool=False):
		"""
			Visualize the env.
			get_metrics: bool (default=False)
				if True, just return the metrics of the env and don't render.
		"""
		envs = self.envs
		self.reset_number = 0
		all_frames = []
		total_dists_traveled, total_time_taken = [], []
		rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = [], [], [],[]
		dist_mean_arr, time_mean_arr = [],[]

		dists_trav_list = np.zeros((self.num_agents))
		time_taken_list = np.zeros((self.num_agents))

		# Set up video writer
		from multiagent.config import eval_scenario_type
		# scenario_name = self.all_args.scenario_name
		if "bayarea" in self.all_args.scenario_name:
			scenario_name = self.all_args.scenario_name
		else:
			scenario_name = "random" if self.all_args.scenario_name == "navigation_graph_safe" else eval_scenario_type
		# check if scenario_name has word "bay"
		if "bayarea" in self.all_args.scenario_name:
			put_label_in_video = True
		else:
			put_label_in_video = False
		if not get_metrics and self.all_args.save_gifs:
			file_name = '/'+ scenario_name + '_num_agent' + str(self.all_args.num_agents) \
						+ '_landmark' + str(self.all_args.num_landmarks) \
						+ '_safety_' + str(self.all_args.use_safety_filter) \
						+ '_world_size' + str(self.all_args.world_size) + '_seed' + str(self.all_args.seed) + '.mp4'
			file_path = str(self.gif_dir) + file_name
			frame_shape = envs.render('rgb_array')[0][0].shape[:2]  # Get height and width
			fps = int(1 / self.all_args.ifi)
			fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
			video_writer = cv2.VideoWriter(file_path, fourcc, fps, (frame_shape[1], frame_shape[0]))
		eval_log_file_name = str(self.gif_dir) + '/eval_log_' + scenario_name + '_num_agent' + str(self.all_args.num_agents) \
						+ '_landmark' + str(self.all_args.num_landmarks) \
						+ '_safety_' + str(self.all_args.use_safety_filter) \
						+ '_world_size' + str(self.all_args.world_size) + '_seed' + str(self.all_args.seed) + '.csv'
		eval_log_file = open(eval_log_file_name, mode='w', newline='', encoding='utf-8')
		eval_log_writer = None


		print("num_episodes: ", self.all_args.render_episodes)
		num_total_episode = int(self.all_args.num_env_steps) // self.all_args.episode_length // self.all_args.n_rollout_threads
		obs, agent_id, node_obs, adj, _ = envs.reset(num_total_episode-1)
		ep_info_list = []
		for episode in range(self.all_args.render_episodes):
			position_log_file_name = str(self.gif_dir) + '/log_position_' + scenario_name + '_num_agent' + str(self.all_args.num_agents) \
							+ '_landmark' + str(self.all_args.num_landmarks) \
							+ '_safety_' + str(self.all_args.use_safety_filter) \
							+ '_world_size' + str(self.all_args.world_size) \
							+ '_episode_' + str(episode) + '_seed' + str(self.all_args.seed) + '.csv'
			position_log_file = open(position_log_file_name, mode='w', newline='', encoding='utf-8')
			position_log_writer = None

			safety_log_file_name = str(self.gif_dir) + '/log_safety_' + scenario_name + '_num_agent' + str(self.all_args.num_agents) \
							+ '_landmark' + str(self.all_args.num_landmarks) \
							+ '_safety_' + str(self.all_args.use_safety_filter) \
							+ '_world_size' + str(self.all_args.world_size) \
							+ '_episode_' + str(episode) + '_seed' + str(self.all_args.seed) + '.csv'
			safety_log_file = open(safety_log_file_name, mode='w', newline='', encoding='utf-8')
			safety_log_writer = None
			
			min_distance_log_file_name = str(self.gif_dir) + '/log_min_distance_' + scenario_name + '_num_agent' + str(self.all_args.num_agents) \
							+ '_landmark' + str(self.all_args.num_landmarks) \
							+ '_safety_' + str(self.all_args.use_safety_filter) \
							+ '_world_size' + str(self.all_args.world_size) \
							+ '_episode_' + str(episode) + '_seed' + str(self.all_args.seed) + '.csv'
			min_distance_log_file = open(min_distance_log_file_name, mode='w', newline='', encoding='utf-8')
			min_distance_log_writer = None

			# print("episode", episode)
			if not get_metrics:
				if self.all_args.save_gifs:
					image = envs.render('rgb_array')[0][0]
					all_frames.append(image)
				else:
					envs.render('human')

			rnn_states = np.zeros((self.n_rollout_threads, 
									self.num_agents, 
									self.recurrent_N, 
									self.hidden_size), 
									dtype=np.float32)
			masks = np.ones((self.n_rollout_threads, 
							self.num_agents, 1), 
							dtype=np.float32)
			available_actions = np.ones((self.num_agents, self.envs.action_space[0].n), 
										dtype=np.float32)
			episode_rewards = []
			
			for step in range(self.episode_length):
				calc_start = time.time()

				zero_masks = masks[0] == 0

				if 	not zero_masks.all():
					available_actions = np.ones((self.num_agents, self.envs.action_space[0].n), 
										dtype=np.float32)
				# Broadcast the boolean mask to match the shape of available_actions
				broadcasted_zero_masks = np.broadcast_to(zero_masks, available_actions.shape)
				
				# TODO: This is a hack to make the stop action available when the agent is done
				stop_mask = np.zeros(self.envs.action_space[0].n)
				stop_mask[int(self.envs.action_space[0].n/2)] = 1
				available_actions[zero_masks[:,0]] = stop_mask
				self.trainer.prep_rollout()
		
				action, rnn_states, actor_features, _ = self.trainer.policy.act(
													np.concatenate(obs),
													np.concatenate(node_obs),
													np.concatenate(adj),
													np.concatenate(agent_id),
													np.concatenate(rnn_states),
													np.concatenate(masks),
													available_actions = available_actions,
													deterministic=True)

				actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
				rnn_states = np.array(np.split(_t2n(rnn_states), 
									self.n_rollout_threads))

				if self.action_space_type == 'MultiDiscrete':
					for i in range(envs.action_space[0].shape):
						uc_actions_env = np.eye(
								envs.action_space[0].high[i]+1)[actions[:, :, i]]
						if i == 0:
							actions_env = uc_actions_env
						else:
							actions_env = np.concatenate((actions_env, 
														uc_actions_env), 
														axis=2)
				elif self.action_space_type == 'Discrete':
					actions_env = self.map_action_index_to_one_hot_vector(actions, envs.action_space[0].n).squeeze(2)
				else:
					raise NotImplementedError

				# Obser reward and next obs
				obs, agent_id, node_obs, adj, \
					rewards,dones, infos, reset_count = envs.step(actions_env)
				# log positions
				agent_position_x_list = [0 for _ in range(self.num_agents)]
				agent_position_y_list = [0 for _ in range(self.num_agents)]
				# 0 if safe and not filtered, 1 if filtered, 2 if violated
				agent_safety_list = [0 for _ in range(self.num_agents)]
				min_distance_list = [np.inf for _ in range(self.num_agents)]
				for info in infos[0]:
					if 'position' in info:
						agent_position_x_list[info['id']] = info['position'][0]
						agent_position_y_list[info['id']] = info['position'][1]

					if ('Safety violated' in info) and info['Safety violated']:
						agent_safety_list[info['id']] = 2
					elif ('Safety filtered' in info) and info['Safety filtered']:
						agent_safety_list[info['id']] = 1

					if 'min_relative_distance' in info:
						min_distance_list[info['id']] = info['min_relative_distance']
				if step == 0:
					position_headers = ["step"]
					for i in range(self.num_agents):
						position_headers.append(f"x_{i}")
						position_headers.append(f"y_{i}")
					position_log_writer = csv.DictWriter(position_log_file, fieldnames=position_headers)
					position_log_writer.writeheader()

					safety_headers = ["step"]
					for i in range(self.num_agents):
						safety_headers.append(f"{i}")
					safety_log_writer = csv.DictWriter(safety_log_file, fieldnames=safety_headers)
					safety_log_writer.writeheader()

					min_distance_log_writer = csv.DictWriter(min_distance_log_file, fieldnames=safety_headers)
					min_distance_log_writer.writeheader()

				if position_log_writer is not None:
					position_log_writer.writerow({"step": step, **{f"x_{i}": agent_position_x_list[i] for i in range(self.num_agents)}, **{f"y_{i}": agent_position_y_list[i] for i in range(self.num_agents)}})
				if safety_log_writer is not None:
					safety_log_writer.writerow({"step": step, **{f"{i}": agent_safety_list[i] for i in range(self.num_agents)}})
				if min_distance_log_writer is not None:
					min_distance_log_writer.writerow({"step": step, **{f"{i}": min_distance_list[i] for i in range(self.num_agents)}})

				episode_rewards.append(rewards)
				dones_env = np.all(dones)
				rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
													self.recurrent_N, 
													self.hidden_size), 
													dtype=np.float32)
				masks = np.ones((self.n_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)
				masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
												dtype=np.float32)
				dones_env = np.all(dones, axis=1)
				masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
				if not get_metrics:
					if self.all_args.save_gifs:
						def _put_label(image, text, pos_x, pos_y, font_size=0.5, color=(255, 255, 255)):
							font = cv2.FONT_HERSHEY_DUPLEX
							fontScale  = font_size
							fontColor = color
							thickness = 1
							cv2.putText(image, text, 
								(pos_x, pos_y),
								font, 
								fontScale,
								fontColor,
								thickness,
								cv2.LINE_AA)
						image = envs.render('rgb_array')[0][0]
						label_x_pos = 800
						label_y_pos = 100
						label_y_offset = 22
						if put_label_in_video:
							_put_label(image, f"Simulation Time: {step :.1f}", label_x_pos, label_y_pos, font_size = 0.7)
							num_agents_done = np.sum(dones)
							agents_departed = [infos[0][i].get('Departed', False) for i in range(len(infos[0]))]
							num_agents_departed = agents_departed.count(True)
							agents_safety_filtered = [infos[0][i].get('Safety filtered', False) for i in range(len(infos[0]))]
							agents_safety_violated = [infos[0][i].get('Safety violated', False) for i in range(len(infos[0]))]
							# print("agents_safety_violated", agents_safety_violated)
							num_agents_safety_violated = agents_safety_violated.count(True)
							num_agents_safety_filtered = agents_safety_filtered.count(True)
							_put_label(image, f"Vehicle Departed: {num_agents_departed} / {self.num_agents}", label_x_pos, label_y_pos + label_y_offset + 11)
							_put_label(image, f"Vehicle Landed:   {num_agents_done} / {self.num_agents}", label_x_pos, label_y_pos + 2 * label_y_offset + 11)
							_put_label(image, f"Vehicle Safety Filtered: {num_agents_safety_filtered} / {self.num_agents}", label_x_pos, label_y_pos + 3 * label_y_offset + 11)
							text_color = (255, 100, 100) if num_agents_safety_violated > 0 else (255, 255, 255)
							_put_label(image, f"Vehicle Near Collision:  {num_agents_safety_violated} / {self.num_agents}", label_x_pos, label_y_pos + 4 * label_y_offset + 11, color=text_color)

						video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
						all_frames.append(image)
						calc_end = time.time()
						elapsed = calc_end - calc_start
						if elapsed < self.all_args.ifi:
							time.sleep(self.all_args.ifi - elapsed)
					else:
						envs.render('human')

				self.reset_number += reset_count
				if reset_count > 0:
					break
				if self.reset_number == self.all_args.render_episodes :
					break
				if np.all(dones_env):
					break
				# input("Press Enter to continue...")
			position_log_file.close()
			safety_log_file.close()
			min_distance_log_file.close()

			env_infos = self.process_infos(infos)

			num_collisions = self.get_collisions(env_infos)
			frac, success,time_taken = self.get_fraction_episodes(env_infos)

			if np.any(frac==1):
				frac_avg = 1.0
			else:
				frac_avg = np.max(frac)
			rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
			frac_episode_arr.append(frac_avg)
			success_rates_arr.append(success)
			num_collisions_arr.append(num_collisions)

			dist_mean = self.get_dist_mean(env_infos)
			dist_mean_arr.append(dist_mean[-1])
			time_mean = self.get_time_mean(env_infos)
			time_mean_arr.append(time_mean[-1])
	
			dists_traveled = self.get_dists_traveled(env_infos)
			dists_trav_list +=dists_traveled

			time_taken_list +=time_taken

			total_dists_traveled.append(np.sum(dists_traveled))
			total_time_taken.append(np.sum(time_taken))

			# RESET HERE.
			obs, agent_id, node_obs, adj, ep_info = envs.reset(num_total_episode-1)
			# ep_info returns stat of the previous episode.
			print("Episode summary: ", ep_info)
			ep_info_list.append(ep_info[0])
			if episode == 0:
				headers = ep_info[0].keys()
				writer = csv.DictWriter(eval_log_file, fieldnames=headers)
				writer.writeheader()
			writer.writerow(ep_info[0])

		# evaluate means in ep_info_list, it is a list of dictionary of same keys
		# Initialize a dictionary to accumulate sums
		accumulated_stats = defaultdict(float)

		for ep_info in ep_info_list:
			for key, value in ep_info.items():
				accumulated_stats[key] += value
		average_stats = {key: accumulated_stats[key] / len(ep_info_list) for key in accumulated_stats}

		# print("Success rates", success_rates_arr)
		# Convert boolean array to integers
		# success_rates_arr = success_rates_arr.astype(int)
		# success_rates_arr = [int(value) for value in success_rates_arr]
		# calculate statistics for success rates
		success_rates_median = np.median(success_rates_arr)
		success_rates_mean = np.mean(success_rates_arr)
		total_dists_traveled_median = np.median(total_dists_traveled)
		total_time_taken_median = np.median(total_time_taken)

		# np.set_printoptions(linewidth=400)
		# print("Rewards", np.mean(rewards_arr))
		# print("Frac of episode", np.mean(frac_episode_arr))

		# report the success rates statistics
		# print("Success rates mean", success_rates_mean)
		# print("Success rates median",success_rates_median)

		# print("Num collisions", np.mean(num_collisions_arr))

		# print("Dists traveled", dists_trav_list)
		# print("Time taken", time_taken_list)

		# rewards_mean = np.mean(rewards_arr)
		
		if not get_metrics and self.all_args.save_gifs:
			video_writer.release()
			print(f"Video saved to {file_path}")

		print("Average Stats over Episodes:", average_stats)
		eval_log_file.close()