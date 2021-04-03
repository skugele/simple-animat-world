import argparse
import sys
import os
from time import time, gmtime, strftime
from pathlib import Path

import tensorflow as tf
import numpy as np
import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy as DpqLnMlpPolicy
from stable_baselines import DQN, A2C, PPO2, SAC, ACER
from stable_baselines.common import set_global_seeds

DEFAULT_ACTION_PORT = 5678
DEFAULT_OBSERVATION_PORT = 9001

BASE_MODEL_PATH = Path('save/stable-baselines')
DEFAULT_MODEL_FILE = Path('model.zip')

algorithm_params = {
    'DQN': {'impl': DQN, 'policy': DqnMlpPolicy, 'save_dir': BASE_MODEL_PATH / 'dqn'},
    'PPO2': {'impl': PPO2, 'policy': MlpLstmPolicy, 'save_dir': BASE_MODEL_PATH / 'ppo2'},
    'A2C': {'impl': A2C, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'a2c'},
    'ACER': {'impl': ACER, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'acer'}
}


def check_args(args):
    """ Check command-line arguments for issues and return an array of error messages.

    :param args: an argparse parser object containing command-line argument values
    :return: an array containing error messages if issues found; otherwise, and empty array.
    """
    error_msgs = []
    if args.algorithm not in algorithm_params.keys():
        error_msgs.append(
            f'Unsupported algorithm: {args.algorithm}. See help (--help -h) for list of supported algorithms.')

    return error_msgs


def parse_args():
    """ Parses users command-line arguments or displays help text.

    :return: an argparse parser containing the commandline arguments
    """

    parser = argparse.ArgumentParser(description='A driver script for a Godot agent')

    # TODO: I may need to rethink all of these with respect to vectorized environments.
    parser.add_argument('--n_agents', metavar='N', type=int, required=False, default=1,
                        help='the number of training agents to spawn')
    parser.add_argument('--action_port', metavar='PORT', type=int, required=False, default=DEFAULT_ACTION_PORT,
                        help='the port number of the Godot action listener')
    parser.add_argument('--obs_port', metavar='PORT', type=int, required=False, default=DEFAULT_OBSERVATION_PORT,
                        help='the port number of the Godot observation publisher')
    parser.add_argument('--topic', metavar='ID', required=False,
                        help='the topics to subscribe', default='')

    # algorithm parameters
    parser.add_argument('--algorithm', metavar='ID', type=str.upper, required=False, default='DQN',
                        help=f'the algorithm to execute. available algorithms: {",".join(algorithm_params.keys())}')
    parser.add_argument('--steps', metavar='N', type=int, required=False, default=10000,
                        help='the number of environment steps to execute')
    parser.add_argument('--steps_per_episode', metavar='N', type=int, required=False, default=np.inf,
                        help='the number of steps per episode')

    # saved model options
    parser.add_argument('--model', metavar='FILE', type=Path, required=False,
                        help='the saved model\'s filename (existing or new)', default=DEFAULT_MODEL_FILE)
    parser.add_argument('--purge', required=False, action="store_true",
                        help='removes previously saved model')

    # modes
    parser.add_argument('--verify', required=False, action="store_true",
                        help='verifies the environment conforms to OpenAI gym standards')
    parser.add_argument('--learn', required=False, action="store_true",
                        help='initiates RL learning session')
    parser.add_argument('--evaluate', required=False, action="store_true",
                        help='evaluates the model\'s quality')
    parser.add_argument('--run', required=False, action="store_true",
                        help='launches an agent using the model\'s policy (no learning occurs during execution)')

    # other
    parser.add_argument('--verbose', required=False, action="store_true",
                        help='increases verbosity')

    args = parser.parse_args()

    # display command-line argument errors (if any)
    errors = check_args(args)
    if errors:
        print('\n'.join(errors))
        exit(1)

    return args


def get_model_filepath(params, args):
    """ Gets the filepath to the saved model file.

    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: an argparse parser object containing command-line argument values
    :return:
    """
    return params['save_dir'] / args.model


def init_model(params, env, args):
    """ Initialize a stable-baselines model.

    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param env: an OpenAI gym environment
    :param args: command-line arguments (i.e., an argparse parser)

    :return: an instantiated stable-baselines model for the requested RL algorithm
    """
    algorithm, policy, saved_model = params['impl'], params['policy'], get_model_filepath(params, args)

    # Custom MLP policy of two layers of size 32 each with tanh activation function
    # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[8, 8])

    return algorithm.load(saved_model.absolute(), env=env) \
        if saved_model.exists() \
        else algorithm(policy, env,
                       # TODO: Move these into algorithm_params (different algos allow different args)
                       # exploration_fraction=0.8,
                       # exploration_final_eps=0.02,
                       # buffer_size=1000,
                       # learning_starts=200,
                       # target_network_update_freq=100,
                       # policy_kwargs=policy_kwargs,
                       verbose=args.verbose)


def make_godot_env(env_id, agent_id, args, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param agent_id: the agent identifier in Godot environment
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param args: command-line arguments (i.e., an argparse parser)
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        env = gym.make(env_id, agent_id=agent_id, args=args)
        env.seed(seed + agent_id)
        return env

    set_global_seeds(seed)
    return _init


def verify_env(args):
    """ Verifies that the environment conforms to OpenAI gym and stable-baseline standards.

    :param args: an argparse parser object containing command-line argument values
    :return: None
    """

    # verify does not work with vectorized environments, so this has to be created separately
    env = gym.make('gym_godot:simple-animat-v0', agent_id=1, args=args)
    env.verify()


def purge_model(params, args):
    """ Removes previously saved model file.

    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: an argparse parser object containing command-line argument values
    :return: None
    """
    saved_model = get_model_filepath(params, args)

    user_input = input(f'purge previous saved model {saved_model} (yes | no)?')

    if user_input.lower() in ['yes', 'y']:
        print(f'purge requested. removing previously saved model {saved_model}')
        try:
            saved_model.unlink()
        except FileNotFoundError as e:
            print(f'Error: file not found {saved_model}!')
            exit(1)
    else:
        print('aborting!')
        exit(1)


def learn(model, params, args):
    """ Executes an RL algorithm training loop on the environment for the specified number of steps.

    After training results are saved in the specified model file.

    :param model: a model object corresponding to a supported RL algorithm.
    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: command-line arguments (i.e., an argparse parser)
    :return: None
    """
    # begin training
    start = time()
    model.learn(total_timesteps=args.steps)
    end = time()

    print(f'elapsed time: {strftime("%H:%M:%S", gmtime(end - start))}')

    # save training results
    model.save(get_model_filepath(params, args))


def evaluate(model, env, args):
    """ Evaluates the goodness of a learned model's behavior policy.

    :param model: a model object corresponding to a supported RL algorithm.
    :param env: an OpenAI gym environment
    :param args: command-line arguments (i.e., an argparse parser)
    :return: tuple containing the mean and std. deviation of the agent's acquired rewards
    """

    mean, std_dev = evaluate_policy(model, env, n_eval_episodes=5)
    print(f'Policy evaluation results: mean (reward) = {mean}; std dev (reward) = {std_dev}')
    return mean, std_dev


def run(model, env, args):
    obs = env.reset()

    done = [False for _ in range(env.num_envs)]
    while not all(done):
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(f'done: {done}')


def main():
    args = parse_args()

    # TODO: Convert to a vectorized environment


    if args.verify:
        verify_env(args)

    env = DummyVecEnv([make_godot_env('gym_godot:simple-animat-v0', i, args, seed=i) for i in range(args.n_agents)])
    params = algorithm_params[args.algorithm]

    if args.purge:
        purge_model(params, args)

    model = init_model(params, env, args)

    if args.learn:
        learn(model, params, args)

    # if args.evaluate:
    #     env = DummyVecEnv([make_godot_env('gym_godot:simple-animat-v0', i, args) for i in range(1)])
    #     model = init_model(params, env, args)
    #     evaluate(model, env, args)

    if args.run:
        run(model, env, args)

    env.close()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:

        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
