# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs football_env on OpenAI's ppo2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as football_env
from gfootball.examples import models
import tensorflow as tf

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('level', 'academy_empty_goal_close',
                    'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(25e4),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; '
                     'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.27, 'Clip range')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')


def create_single_football_env(seed, level):
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name=level, stacked=('stacked' in 'extracted_stacked'),
      rewards='scoring',
      logdir=logger.get_dir(),
      enable_goal_videos=False,
      enable_full_episode_videos=False,
      render=False,
      dump_frequency=0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(seed)))
  return env


import gym

# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# multiprocess environment

def train():
    n_cpu = 7

    env = SubprocVecEnv([lambda _i = i: create_single_football_env(_i, 'academy_three_vs_one_right') for i in range(n_cpu)])
    print("Start LEARNING")

    model = PPO2(MlpPolicy,
                env,n_steps=FLAGS.nsteps,
                gamma=FLAGS.gamma,
                ent_coef=FLAGS.ent_coef,
                nminibatches=FLAGS.nminibatches,
                verbose=1,
                noptepochs=FLAGS.noptepochs,
                learning_rate=FLAGS.lr,
                tensorboard_log = 'gfootball/tensorboard/academy_three_vs_one_right',
                cliprange=FLAGS.cliprange)

    model.learn(total_timesteps=FLAGS.num_timesteps,
               log_interval=1,
               )
    print("DONE LEARNING From academy goal")
    model.save("3v1_player2")

    env = SubprocVecEnv([lambda _i = i: create_single_football_env(_i, 'academy_three_vs_one_mid') for i in range(n_cpu)])
    print("Start LEARNING")

    model = PPO2(MlpPolicy,
                env,n_steps=FLAGS.nsteps,
                gamma=FLAGS.gamma,
                ent_coef=FLAGS.ent_coef,
                nminibatches=FLAGS.nminibatches,
                verbose=1,
                noptepochs=FLAGS.noptepochs,
                learning_rate=FLAGS.lr,
                tensorboard_log = 'gfootball/tensorboard/academy_three_vs_one_mid',
                cliprange=FLAGS.cliprange)

    model.learn(total_timesteps=FLAGS.num_timesteps,
               log_interval=1,
               )
    print("DONE LEARNING From academy goal")
    model.save("3v1_player3")
    # n_cpu = 7
    # env = SubprocVecEnv([lambda _i = i: create_single_football_env(_i, 'academy_two_vs_one_right') for i in range(n_cpu)])
    # print("Start LEARNING")
    #
    # model = PPO2(MlpPolicy,
    #             env,n_steps=FLAGS.nsteps,
    #             gamma=FLAGS.gamma,
    #             ent_coef=FLAGS.ent_coef,
    #             nminibatches=FLAGS.nminibatches,
    #             verbose=1,
    #             noptepochs=FLAGS.noptepochs,
    #             learning_rate=FLAGS.lr,
    #             tensorboard_log = 'gfootball/tensorboard/academy_two_vs_one_right',
    #             cliprange=FLAGS.cliprange)
    #
    # model.learn(total_timesteps=FLAGS.num_timesteps,
    #            log_interval=1,
    #            )
    # print("DONE LEARNING From academy goal")
    # model.save("2v1_player2")



if __name__ == '__main__':
  train()
