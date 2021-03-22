import argparse
import sys
import os
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C, PPO2, SAC
from stable_baselines.common import make_vec_env

DEFAULT_ACTION_PORT = 5678
DEFAULT_OBSERVATION_PORT = 9001


def parse_args():
    parser = argparse.ArgumentParser(description='A driver script for a Godot agent')

    parser.add_argument('--id', type=int, required=True, help='the agent\'s identifier')
    parser.add_argument('--action_port', type=int, required=False, default=DEFAULT_ACTION_PORT,
                        help='the port number of the Godot action listener')
    parser.add_argument('--obs_port', type=int, required=False, default=DEFAULT_OBSERVATION_PORT,
                        help='the port number of the Godot observation publisher')

    # TODO: This should be created elsewhere
    parser.add_argument('--topic', metavar='TOPIC', required=False, help='the topics to subscribe', default='')
    parser.add_argument('--verbose', required=False, action="store_true",
                        help='increases verbosity (displays incoming and outgoing messages)')
    parser.add_argument('--verify', required=False, action="store_true",
                        help='verifies the env conforms to OpenAI Gym and baseline standards')
    parser.add_argument('--learn', required=False, action="store_true",
                        help='initiates RL learning')

    return parser.parse_args()


def choose_random_action(env):
    return env.action_space.sample()


def run():
    # process commandline arguments
    args = parse_args()

    # connect to Godot environment)
    env = gym.make('gym_godot:simple-animat-v0', args=args)

    if args.verify:
        env.verify()
        exit(0)

    # model_filepath = '../save/a2c.zip'
    #
    model = None
    # if os.path.exists(model_filepath):
    #     print(f'loading model {model_filepath}')
    #     model = A2C.load(model_filepath)
    #     model.set_env(env)
    # else:
    #     print(f'unable to find model {model_filepath}')

    if args.learn:
        if model is None:
            model = A2C(MlpPolicy, env, verbose=1)

        model.learn(total_timesteps=500)
        model.save('../save/sac')
        exit(0)

    if model is None:
        print("nope")

    obs = env.reset()

    done = False
    while not done:

        # TODO: This should be in an Agent class
        action = choose_random_action(env)
        # if args.verbose:
        #     print(f'executing action: {action}')

        observation, reward, done, info = env.step(action)

        if args.verbose:
            print(f'observation: {observation}')
            print(f'reward: {reward}')


if __name__ == "__main__":

    try:
        run()
    except KeyboardInterrupt:

        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
