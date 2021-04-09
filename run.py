import os
import sys
import re
import argparse
from collections import namedtuple, OrderedDict
from time import time, gmtime, strftime
from pathlib import Path
import traceback
from typing import Optional

import numpy as np
import optuna
import tensorflow as tf
import gym
import yaml
from optuna.integration import SkoptSampler
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.bench import Monitor

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines import DQN, A2C, PPO2, SAC, ACER, ACKTR
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

# SAC and DQNs have custom (non-common) policy implementations
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds

from eval import NonEpisodicEnvMonitor, CustomCheckPointCallback

# suppressing tensorflow's debug, info, and warning messages
from zoo.utils.hyperparams_opt import hyperparam_optimization, HYPERPARAMS_SAMPLER

tf.get_logger().setLevel('ERROR')

GodotInstance = namedtuple('GodotInstance', ['obs_port', 'action_port'])
GODOT_EVAL_INSTANCE = GodotInstance(9998, 9999)

RUNTIME_PATH = Path('tmp/')
BASE_MODEL_PATH = Path('save/stable-baselines')
DEFAULT_MODEL_FILE = Path('model.zip')

SAVE_FREQUENCY = 1000  # save freq. in training steps. note: for vectorized envs this will be n_envs * n_steps_per_env

algorithm_params = {
    'DQN': {'impl': DQN, 'policy': DqnMlpPolicy, 'save_dir': BASE_MODEL_PATH / 'dqn', 'hyper_params': {}},
    'PPO2': {'impl': PPO2, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'ppo2', 'hyper_params': {}},
    'A2C': {'impl': A2C, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'a2c', 'hyper_params': {}},
    'ACER': {'impl': ACER, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'acer', 'hyper_params': {}},
    'SAC': {'impl': SAC, 'policy': SacMlpPolicy, 'save_dir': BASE_MODEL_PATH / 'sac', 'hyper_params': {}},
    'ACKTR': {'impl': ACKTR, 'policy': MlpPolicy, 'save_dir': BASE_MODEL_PATH / 'acktr', 'hyper_params': {}},
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
    parser = argparse.ArgumentParser(description='A driver script for the training and execution of a Godot agent')

    parser.add_argument('--n_agents_per_env', metavar='N', type=int, required=False, default=1,
                        help='the number of training agents to spawn per godot environment')
    parser.add_argument('--n_godot_instances', metavar='N', type=int, required=False, default=1,
                        help='the number of available godot environments')

    # algorithm parameters
    parser.add_argument('--algorithm', metavar='ID', type=str.upper, required=False, default='DQN',
                        help=f'the algorithm to execute. available algorithms: {",".join(algorithm_params.keys())}')
    parser.add_argument('--max_steps_per_episode', metavar='N', type=int, required=False, default=1000,
                        help='the maximum number of environment steps to execute per episode')
    parser.add_argument('--steps', metavar='N', type=int, required=False, default=10000,
                        help='the number of environment steps to execute')
    parser.add_argument('--n_stack', metavar='N', type=int, required=False, default=1,
                        help='the number of observation frames to stack')

    # saved model options
    parser.add_argument('--model', metavar='FILE', type=Path, required=False,
                        help='the saved model\'s filename (existing or new)', default=DEFAULT_MODEL_FILE)
    parser.add_argument('--purge', required=False, action="store_true",
                        help='removes previously saved model')

    # hyper-parameter optimization settings
    parser.add_argument('--sampler', help='sampler to use when optimizing hyper-parameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='pruner to use when optimizing hyper-parameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])

    # modes
    parser.add_argument('--verify', required=False, action="store_true",
                        help='verifies the environment conforms to OpenAI gym standards')
    parser.add_argument('--learn', required=False, action="store_true",
                        help='initiates RL learning session')
    parser.add_argument('--optimize', required=False, action="store_true",
                        help='runs a hyper-parameter optimization study')
    parser.add_argument('--evaluate', required=False, action="store_true",
                        help='evaluates the model\'s quality')
    parser.add_argument('--run', required=False, action="store_true",
                        help='launches an agent using the model\'s policy (no learning occurs during execution)')

    # other
    parser.add_argument('--verbose', required=False, action="store_true",
                        help='increases verbosity')
    parser.add_argument('--session_id', metavar='ID', type=str, required=False,
                        help='a session id to use (existing or new)', default=None)

    args = parser.parse_args()

    # display command-line argument errors (if any)
    errors = check_args(args)
    if errors:
        print('\n'.join(errors))
        exit(1)

    return args


def get_model_filepath(params, args, filename):
    """ Gets the filepath to the saved model file.

    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: an argparse parser object containing command-line argument values
    :return:
    """
    return params['save_dir'] / filename


def get_stats_filepath(session_path, agent_id, eval=False):
    """ Gets the filepath to the run statistics file."""

    stats_dir = session_path / 'monitor'
    stats_dir.mkdir(exist_ok=True)

    if eval:
        return stats_dir / 'eval.csv'
    else:
        return stats_dir / f'agent_{agent_id}.csv'


def get_tensorboard_path(session_path):
    tf_dir = session_path / 'tensorboard'
    if not tf_dir.exists():
        tf_dir.mkdir()

    return tf_dir


def init_model(session_path, params, env, args, eval=False):
    """ Initialize a stable-baselines model.

    :param eval:
    :param session_path: (str) the session ID
    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param env: an OpenAI gym environment
    :param args: command-line arguments (i.e., an argparse parser)

    :return: an instantiated stable-baselines model for the requested RL algorithm
    """
    algorithm, policy, saved_model = params['impl'], params['policy'], get_model_filepath(params, args,
                                                                                          filename=args.model)

    # Custom MLP policy of two layers of size 32 each with tanh activation function
    # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[8, 8])

    if saved_model.exists():
        return algorithm.load(saved_model.absolute(),
                              env=env,
                              tensorboard_log=get_tensorboard_path(session_path))
    else:
        if eval:
            raise ValueError('evaluation mode requires a saved model.')

        return algorithm(policy, env, **params['hyper_params'],
                         verbose=args.verbose,
                         tensorboard_log=get_tensorboard_path(session_path))


def make_godot_env(env_id, agent_id, obs_port, action_port, args, session_path, eval=False, seed=0):
    """ Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param agent_id: the agent identifier in Godot environment
    :param action_port: port number for Godot action server
    :param obs_port: port number for Godot observation publisher
    :param args: command-line arguments (i.e., an argparse parser)
    :param session_path: (str) the session ID
    :param seed: (int) the initial seed for RNG
    """

    def _init():
        env = gym.make(env_id, agent_id=agent_id, obs_port=obs_port, action_port=action_port, args=args)
        env = NonEpisodicEnvMonitor(env, filename=get_stats_filepath(session_path, agent_id, eval), freq=100)
        env.seed(seed + agent_id)
        return env

    set_global_seeds(seed)
    return _init


def create_env(args, env_id, godot_instances, params, session_path, eval=False):
    n = 1 if eval else args.n_agents_per_env
    env = SubprocVecEnv([make_godot_env(env_id, i, obs_port, action_port, args, session_path, eval, seed=i)
                         for i in range(n) for obs_port, action_port in godot_instances])

    env_stats_path = get_model_filepath(params, args, filename='vec_normalize.pkl')
    if env_stats_path.exists():
        env = VecNormalize.load(env_stats_path, env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0, clip_reward=100.0)

    if args.n_stack > 1:
        env = VecFrameStack(env, n_stack=args.n_stack)

    return env


def verify_env(env_id, args):
    """ Verifies that the environment conforms to OpenAI gym and stable-baseline standards.

    :param env_id: (str) the environment ID
    :param args: an argparse parser object containing command-line argument values
    :return: None
    """

    # verify does not work with vectorized environments, so this has to be created separately
    env = gym.make(env_id, agent_id=1, args=args)
    env.verify()


def purge_model(params, args, interactive=True):
    """ Removes previously saved model file.

    :param interactive:
    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: an argparse parser object containing command-line argument values
    :return: None
    """
    saved_model = get_model_filepath(params, args, filename=args.model)
    saved_stats = get_model_filepath(params, args, filename='vec_normalize.pkl')

    if saved_model.exists():
        if interactive:
            user_input = input(f'purge previous saved model {saved_model} (yes | no)?')
        else:
            user_input = 'yes'

        if user_input.lower() in ['yes', 'y']:
            print(f'purge requested. removing previously saved model {saved_model} and stats')
            try:
                saved_model.unlink()
                saved_stats.unlink()
            except FileNotFoundError as e:
                print(f'Error: file not found {saved_model}!')
                exit(1)
        else:
            print('aborting!')
            exit(1)


def learn(env, model, params, args):
    """ Executes an RL algorithm training loop on the environment for the specified number of steps.

    After training results are saved in the specified model file.

    :param model: a model object corresponding to a supported RL algorithm.
    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: command-line arguments (i.e., an argparse parser)
    :return: None
    """
    # add callbacks
    cp_callback = CustomCheckPointCallback(save_path=params['save_dir'], model_filename='model.zip', verbose=1)

    # begin training
    start = time()
    model.learn(total_timesteps=args.steps, callback=cp_callback)
    end = time()

    print(f'elapsed time: {strftime("%H:%M:%S", gmtime(end - start))}')

    # save training results
    model.save(get_model_filepath(params, args, filename=args.model))
    env.save(get_model_filepath(params, args, filename='vec_normalize.pkl'))


def optimize(env_id, params, args, session_path):
    n_trials = 25
    n_startup_trials = 5  # prevents pruning until some number of trails have occurred
    n_episodes_per_eval = 10

    seed = int(time())

    if args.sampler == 'random':
        sampler = RandomSampler(seed=seed)
    elif args.sampler == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif args.sampler == 'skopt':
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(args.sampler))

    if args.pruner == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif args.pruner == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials)
    elif args.pruner == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials)
    else:
        raise ValueError('Unknown pruner: {}'.format(args.pruner))

    study = optuna.create_study(sampler=sampler, pruner=pruner)

    # the objective function called by optuna during each trial
    def objective(trial):
        # copy to preserve original params
        _params = params.copy()

        _params['hyper_params'] = HYPERPARAMS_SAMPLER[args.algorithm.lower()](trial)
        _params['save_dir'] = _params['save_dir'] / 'optimizer'

        try:
            # purge any previously saved models
            purge_model(_params, args, interactive=False)

            ######################################################
            # learning phase - on possibly multiple environments #
            ######################################################
            godot_instances = [GodotInstance(o_port, a_port) for o_port, a_port in
                               get_godot_instances(args.n_godot_instances)]
            env = create_env(args, env_id, godot_instances, _params, session_path)

            # learn and save model
            model = init_model(session_path, _params, env, args)
            learn(env, model, _params, args)
            env.close()

            ##########################################################################
            # evaluation phase - single environment (deterministic action selection) #
            ##########################################################################
            env = create_env(args, env_id, [GODOT_EVAL_INSTANCE], _params, session_path, eval=True)

            # loaded previously learned model and evaluate
            model = init_model(session_path, _params, env, args, eval=True)
            mean_reward, _ = evaluate(model, env, args, n_episodes=n_episodes_per_eval)
            env.close()

        # TODO: I may need to implement some kind of callback to deal with NaNs.
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            raise optuna.exceptions.TrialPruned()

        # optuna minimizes the objective by default, so we need to flip the sign to maximize
        cost = -1 * mean_reward
        return cost

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def evaluate(model, env, args, n_episodes=5):
    """ Evaluates the goodness of a learned model's behavior policy.

    :param model: a model object corresponding to a supported RL algorithm.
    :param env: an OpenAI gym environment
    :param args: command-line arguments (i.e., an argparse parser)
    :param n_episodes: the number of episodes to use for policy evaluation

    :return: tuple containing the mean and std. deviation of the agent's acquired rewards over episodes
    """
    mean, std_dev = evaluate_policy(model, env, n_eval_episodes=n_episodes)

    if args.verbose:
        print(
            f'Policy evaluation results: mean (reward) per episode = {mean}; std dev (reward) per episode = {std_dev}')

    return mean, std_dev


def run(model, env, args):
    obs = env.reset()

    done = [False for _ in range(env.num_envs)]
    while not all(done):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)


def sanitize(string):
    """ Cleanses a string so it is safe to use in a file path """
    return re.sub(r'[:/\\<>|?*\'\"]', '_', string)


def create_session_id(env_id, params, args):
    return sanitize(f'{env_id}_{args.algorithm}_{int(time())}')


def init_session(env_id, params, args):
    session_id = sanitize(args.session_id) if args.session_id else create_session_id(env_id, params, args)
    session_path = RUNTIME_PATH / session_id
    if not session_path.exists():
        session_path.mkdir()

    return session_id, session_path


def get_godot_instances(n, starting_obs_port=10001, starting_action_port=10002):
    return map(lambda i: (starting_obs_port + i * 2, starting_action_port + i * 2), range(n))


def main(env_id):
    args = parse_args()
    params = algorithm_params[args.algorithm]

    if args.verify:
        verify_env(env_id, args)

    if args.purge:
        purge_model(params, args)

    session_id, session_path = init_session(env_id, params, args)

    if args.optimize:

        data_frame = optimize(env_id, params, args, session_path)

        report_name = f'optimizer_results_{args.algorithm}_{int(time())}-{args.sampler}-{args.pruner}.csv'
        report_path = session_path / report_name

        if args.verbose:
            print("Writing report to {}".format(report_path))

        data_frame.to_csv(report_path)

    elif args.learn:
        godot_instances = [GodotInstance(o_port, a_port) for o_port, a_port in
                           get_godot_instances(args.n_godot_instances)]
        env = create_env(args, env_id, godot_instances, params, session_path)
        model = init_model(session_path, params, env, args)
        learn(env, model, params, args)
        env.close()

    elif args.evaluate:
        env = create_env(args, env_id, [GODOT_EVAL_INSTANCE], params, session_path, eval=True)
        model = init_model(session_path, params, env, args)
        evaluate(model, env, args)
        env.close()


if __name__ == "__main__":

    try:
        main(env_id='gym_godot:simple-animat-v0')
    except Exception as e:
        print(f'execution terminated due to the following exception:\n\"{e}\"', file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
