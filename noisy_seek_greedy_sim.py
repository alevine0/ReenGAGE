from noisy_seek import NoisySeek
import numpy as np
env = NoisySeek(**{"max_goals":200,"timesteps":40, "center_sigma": 10., "noise_sigma": 1.})

tot_tot_reward = 0
for i in range(1000):
	done = False
	obs = env.reset()
	tot_reward = 0
	while (not done):
		goals = obs['desired_goal'].reshape((-1,3))
		goals =  goals[goals[:,0] == 1][:,1:]
		directions = goals - obs["observation"]
		dists = np.linalg.norm(directions,axis=1)
		amin = np.argmin(dists)
		opt_dir = directions[amin]
		dist = dists[amin]
		if (dist > 1):
			opt_dir = opt_dir/dist
		obs,reward,done,_= env.step(opt_dir)
		tot_reward += reward
	tot_tot_reward += tot_reward
	print(tot_tot_reward/(i+1))
print('final')
print(tot_tot_reward/(i+1))