import argparse
import sys
import os
import numpy as np
import gym
import pathlib


# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN, A2C, PPO2, SAC, ACER
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
    parser.add_argument('--model', metavar='FILEPATH', type=pathlib.Path, required=False,
                        help='a filepath to a model file (for loading and saving)')
    parser.add_argument('--purge', required=False, action="store_true",
                        help='removes any existing model at same model FILEPATH before learning')

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


    # For learning and policy execution a model file is required
    if not args.model:
        print('a model FILEPATH is required!')
        exit(1)

    # args.model = args.model.absolute()

    if args.purge and args.model.exists():
        print(f'removing previous model file {args.model}')
        args.model.unlink()

    algorithm = DQN
    model = None

    if args.model.exists():
        print(f'loading model {args.model}')
        model = algorithm.load(args.model.absolute(), env=env)

    if args.learn:
        if model is None:
            model = algorithm(MlpPolicy, env, exploration_fraction=1.0, verbose=1)

            # try preliminary save to catch file access issues
            model.save(args.model)

        model.learn(total_timesteps=25000)
        model.save(args.model)
        exit(0)
    else:
        if model is None:
            if args.model:
                print(f'unable to find agent model {args.model}')
            else:
                print(f'a MODEL filepath must be specified (see help)')

            exit(1)

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
