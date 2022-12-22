Code and Supplementary Material for the paper "Goal-Conditioned Q-Learning as Knowledge Distillation" by Alexander Levine and Soheil Feizi, accepted in AAAI 2023. 

Non-Robotics Experiments:

See requirements.txt for installation requirements. Experiments can be reproduced using ``train_drive_seek.py'', ``train_noisy_seek.py'' and ``train_continuous_seek.py''. The flag ``--gradient_reg'' sets the value of the coefficent alpha (times the goal dimensionality d); setting this to zero will run the baseline experiments. We also include ``noisy_seek_greedy_sim.py'' to run the simulation of a ``greedy'' agent on NoisySeek.

Appendix experiments can be reproduced using:``train_drive_seek_no_sharing.py'' to reproduce the ablation study for Multi-ReenGAGE without parameter sharing, ``train_drive_seek_cnn.py'' for  Multi-ReenGAGE on DriveSeek using a CNN architecure, and ``train_continuous_seek_sac.py'' for ContinuousSeek with ReenGAGE+SAC+HER.

Robotics Experiments:

The ``baselines'' directory contains a fork of the OpenAI Baselines package, with additional code for ReenGAGE (here labeled as "gradher"). We also include the "Normalized" variant of ReenGAGE discussed in the appendix, as gradher_normalized. See the OpenAI baselines documentation for general setup instructions; an exammple of using ReenGAGE is provided below:

 mpirun -np 19   python -m baselines.run --alg=gradher --env=HandReach-v0 --num_timesteps=250000 --num_env 2 --seed 0 --log_path=./hand_0.0001_seed_0

We include our set-up as baselines_requirements.txt. Note that we used mujoco-py 2.1: while there are some documented issues with using this version of mujoco-py with openai gym, we do not believe that these issues affect the hand environments which we are using (see the following GitHub issues: https://github.com/openai/gym/issues/1711; https://github.com/openai/gym/issues/1541; https://github.com/openai/mujoco-py/pull/659; https://github.com/openai/gym/issues/2528; the hand environments do not apparently use the cacc, cfrc_ext or cfrc_int fields) and the baseline performance we observe seems consistent with (Plappert et al. 2018); nonetheless, we compare our method to reproduced baselines run in our environment for the sake of consistency.