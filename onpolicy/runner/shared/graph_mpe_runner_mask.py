import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import csv

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

	def run(self):
		self.warmup()   

		start = time.time()
		episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
		print("episodes", episodes)

		# This is where the episodes are actually run.
		for episode in range(episodes):


			if self.use_linear_lr_decay:
				self.trainer.policy.lr_decay(episode, episodes)
			flag = 0

			self.active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
									dtype=np.int32)

			for step in range(self.episode_length):
				# print("\nstep", step)
				# Sample actions
				if step == 0:
					# active_agents = None
					# active_mask = None
					# finished = None
					values, actions, action_log_probs, rnn_states, \
						rnn_states_critic, actions_env = self.collect(step)
					obs, agent_id, node_obs, adj, rewards, \
						dones, infos = self.envs.step(actions_env)
					finished,finished_list = self.get_finished(dones)
					self.active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)


					dones_env = np.all(dones, axis=1)
					self.active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
											self.num_agents, 1), dtype=np.float32)

					available_actions = np.ones((self.n_rollout_threads, self.num_agents, 25), dtype=np.float32)

					# # For no-Collab uncomment the line
					rewards = rewards[:,:, np.newaxis]
					data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
							dones, infos, values, actions, action_log_probs, 
							rnn_states, rnn_states_critic, available_actions)

					# insert data into buffer
					self.insert(data)
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
						dones, infos = self.envs.step(actions_env)

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
					self.insert(data, active_agents=self.active_agents,active_masks=self.active_masks,finished=finished)
					
					## if all agents are done, the env will be reset. don't forget to reset the active agents, masks and finished
					# input("check")


			# compute return and update network
			self.compute()
			train_infos = self.train()
			
			# post process
			total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
			
			# save model
			if (episode % self.save_interval == 0 or episode == episodes - 1):
				self.save()

			# log information
			if episode % self.log_interval == 0:
				end = time.time()

				env_infos = self.process_infos(infos)

				avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
				train_infos["average_episode_rewards"] = avg_ep_rew
				print(f"Average episode rewards is {avg_ep_rew:.3f} \t"
					f"Total timesteps: {total_num_steps} \t "
					f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}")
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



	def warmup(self):
		# reset env
		obs, agent_id, node_obs, adj = self.envs.reset()

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
						available_actions = np.zeros((25))
						available_actions[12] = 1
						flag= True
					else:
						available_actions=np.ones((25))
						
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
						available_actions = np.reshape(aa, (self.all_args.num_agents*self.n_rollout_threads, 25), order='C'))

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
					available_actions=np.ones((25))
					aa.append(available_actions)
				avail_actions_list.append(aa)

			
			avail_actions= np.asarray(avail_actions_list)

		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env, avail_actions)
	def async_compute_global_goal(self, active_agents,active_mask,finished):
		self.trainer.prep_rollout()        
		flag = False
		### look and see who is active and force the avilable actions to be limited

		all_actions=[]
		aaa=[]
		envs_aa=[]
		for env in range(self.n_rollout_threads): #len(active_mask)):
			aaa=[]
			for a in range(self.all_args.num_agents):
				# if active_mask[env][a]==True:
				if a in finished[env]:
					available_actions = np.zeros((25))
					available_actions[12] = 1
					flag= True
				else:
					available_actions=np.ones((25))
				aaa.append(available_actions)
			envs_aa.append(aaa)
		aa= np.asarray(envs_aa)

		concat_share_obs= np.stack([self.buffer.share_obs[step, e, a] for e, a, step in active_agents], axis=0)
		concat_obs= np.stack([self.buffer.obs[step, e, a] for e, a, step in active_agents], axis=0)
		if flag: ## if there are some finished agents:
			value, action, action_log_prob, rnn_states, rnn_states_critic \
					= self.trainer.policy.get_actions(concat_share_obs,
													concat_obs,
													np.stack([self.buffer.node_obs[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.adj[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.agent_id[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.share_agent_id[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.rnn_states[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.rnn_states_critic[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.masks[step, e, a] for e, a, step in active_agents], axis=0),
													available_actions = np.stack([aa[e,a] for e,a,step in active_agents],axis=0) )
		else:
			value, action, action_log_prob, rnn_states, rnn_states_critic \
					= self.trainer.policy.get_actions(concat_share_obs,
													concat_obs,
													np.stack([self.buffer.node_obs[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.adj[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.agent_id[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.share_agent_id[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.rnn_states[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.rnn_states_critic[step, e, a] for e, a, step in active_agents], axis=0),
													np.stack([self.buffer.masks[step, e, a] for e, a, step in active_agents], axis=0))
		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob),
												self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states),
									self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic),
												self.n_rollout_threads))
		# rearrange action
		if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
			for i in range(self.envs.action_space[0].shape):
				uc_actions_env = np.eye(self.envs.action_space[0].high[i] +
													1)[actions[:, :, i]]
				if i == 0:
					actions_env = uc_actions_env
				else:
					actions_env = np.concatenate((actions_env,
													uc_actions_env), axis=2)
		elif self.envs.action_space[0].__class__.__name__ == "Discrete":
			actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
			
		if flag:
			avail_actions = aa
		else:
			aaa=[]
			for env in range(len(active_mask)):
				aa=[]
				for a in range(len(active_mask[env])):
					available_actions=np.ones((25))
					aa.append(available_actions)
				aaa.append(aa)
			avail_actions= np.asarray(aaa)
		return values, actions, action_log_probs, rnn_states, rnn_states_critic,actions_env, avail_actions

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
		else:
			raise NotImplementedError

		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env)

	def insert(self, data,active_agents=None,active_masks = None, finished=None):
		obs, agent_id, node_obs, adj, agent_id, rewards, dones, \
			infos, values, actions, action_log_probs, \
			rnn_states, rnn_states_critic, available_actions = data
		# print("check ", dones.shape)

		dones_env = np.all(dones, axis=1)
		# print("dones in the env!!", dones_env.shape,dones_env)
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
		# print("masks in graphMpeRunner", masks.shape)

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
		# print("runner rewards", rewards.shape)
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

		print("num_episodes: ", self.all_args.render_episodes)
		for episode in range(self.all_args.render_episodes):
			# print("episode", episode)
			obs, agent_id, node_obs, adj = envs.reset()
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
			available_actions = np.ones((self.num_agents, 25), 
										dtype=np.float32)
			episode_rewards = []
			
			for step in range(self.episode_length):

				calc_start = time.time()
				# print("Masks", masks.shape, masks)
				# print("Available Actions", available_actions.shape, available_actions)

				zero_masks = masks[0] == 0
				# print("Zero Masks", zero_masks[:,0].shape, zero_masks[:,0])

				if 	not zero_masks.all():
					available_actions = np.ones((self.num_agents, 25), 
										dtype=np.float32)
				# Broadcast the boolean mask to match the shape of available_actions
				broadcasted_zero_masks = np.broadcast_to(zero_masks, available_actions.shape)

				
				available_actions[zero_masks[:,0]] = np.array([0, 0, 0, 0, 0, 0,
												                0, 0, 0, 0, 0, 0, 
																1, 0, 0, 0, 0, 0, 
																0, 0, 0, 0, 0, 0, 0])
				# print("Available Actions NEW", available_actions)
				self.trainer.prep_rollout()
				action, rnn_states = self.trainer.policy.act(
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

				if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
					for i in range(envs.action_space[0].shape):
						uc_actions_env = np.eye(
								envs.action_space[0].high[i]+1)[actions[:, :, i]]
						if i == 0:
							actions_env = uc_actions_env
						else:
							actions_env = np.concatenate((actions_env, 
														uc_actions_env), 
														axis=2)
				elif envs.action_space[0].__class__.__name__ == 'Discrete':
					actions_env = np.squeeze(np.eye(
											envs.action_space[0].n)[actions], 2)
				else:
					raise NotImplementedError

				# Obser reward and next obs
				obs, agent_id, node_obs, adj, \
					rewards,dones, infos, reset_count = envs.step(actions_env)

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
						image = envs.render('rgb_array')[0][0]

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
				# input("Press Enter to continue...")

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

		# print("Success rates", success_rates_arr)
		# Convert boolean array to integers
		# success_rates_arr = success_rates_arr.astype(int)
		# success_rates_arr = [int(value) for value in success_rates_arr]
		# calculate statistics for success rates

		success_rates_median = np.median(success_rates_arr)

		success_rates_mean = np.mean(success_rates_arr)
		total_dists_traveled_median = np.median(total_dists_traveled)
		total_time_taken_median = np.median(total_time_taken)

		np.set_printoptions(linewidth=400)
		print("Rewards", np.mean(rewards_arr))
		print("Frac of episode", np.mean(frac_episode_arr))

		# report the success rates statistics
		print("Success rates mean", success_rates_mean)
		print("Success rates median",success_rates_median)

		print("Num collisions", np.mean(num_collisions_arr))

		print("Dists traveled", dists_trav_list)
		print("Time taken", time_taken_list)

		rewards_mean = np.mean(rewards_arr)

		
		if not get_metrics:
			if self.all_args.save_gifs:
				file_name = '/safe_aam_' + 'num_agent' + str(self.all_args.num_agents) \
					+ '_safety_' + str(self.all_args.use_safety_filter) \
					+ '.gif'
				imageio.mimsave(str(self.gif_dir) + file_name, 
								all_frames, duration=self.all_args.ifi, loop=0)