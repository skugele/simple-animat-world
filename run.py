import sys
import re
import argparse
from time import time, gmtime, strftime
from pathlib import Path
import traceback

import numpy as np
import tensorflow as tf
import gym

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines import DQN, A2C, PPO2, SAC, ACER, ACKTR
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

# SAC and DQNs have custom (non-common) policy implementations
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds

# from stable_baselines import results_plotter
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.noise import AdaptiveParamNoiseSpec
# from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.callbacks import CheckpointCallback, CallbackList, EveryNTimesteps

from eval import NonEpisodicEnvMonitor

# suppressing tensorflow's debug, info, and warning messages
tf.get_logger().setLevel('ERROR')

DEFAULT_ACTION_PORT = 5678
DEFAULT_OBSERVATION_PORT = 9001

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

    parser = argparse.ArgumentParser(description='A driver script for a Godot agent')

    parser.add_argument('--n_agents', metavar='N', type=int, required=False, default=1,
                        help='the number of training agents to spawn')

    # TODO: I may need to rethink all of these with respect to vectorized environments.
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


def get_stats_filepath(session_path, agent_id):
    """ Gets the filepath to the run statistics file. """

    stats_dir = session_path / 'monitor'
    if not stats_dir.exists():
        stats_dir.mkdir()

    # TODO: if the filename ends in ".gz" then numpy will automatically use compression.
    return stats_dir / f'agent_{agent_id}.csv'


def get_tensorboard_path(session_path):
    tf_dir = session_path / 'tensorboard'
    if not tf_dir.exists():
        tf_dir.mkdir()

    return tf_dir


def init_model(session_path, params, env, args):
    """ Initialize a stable-baselines model.

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
        return algorithm(policy, env, **params['hyper_params'],
                         verbose=args.verbose,
                         tensorboard_log=get_tensorboard_path(session_path))


def make_godot_env(session_path, env_id, agent_id, args, seed=0):
    """
    Utility function for multiprocessed env.

    :param session_path: (str) the session ID
    :param env_id: (str) the environment ID
    :param agent_id: the agent identifier in Godot environment
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param args: command-line arguments (i.e., an argparse parser)
    :param seed: (int) the initial seed for RNG
    """

    def _init():
        env = gym.make(env_id, agent_id=agent_id, args=args)
        env = NonEpisodicEnvMonitor(env, filename=get_stats_filepath(session_path, agent_id), freq=100)
        env.seed(seed + agent_id)
        return env

    set_global_seeds(seed)
    return _init


def verify_env(env_id, args):
    """ Verifies that the environment conforms to OpenAI gym and stable-baseline standards.

    :param env_id: (str) the environment ID
    :param args: an argparse parser object containing command-line argument values
    :return: None
    """

    # verify does not work with vectorized environments, so this has to be created separately
    env = gym.make(env_id, agent_id=1, args=args)
    env.verify()


def purge_model(params, args):
    """ Removes previously saved model file.

    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: an argparse parser object containing command-line argument values
    :return: None
    """
    saved_model = get_model_filepath(params, args, filename=args.model)

    # TODO: need to also remove saved VecNormalize stats

    if saved_model.exists():
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


def learn(env, model, params, args):
    """ Executes an RL algorithm training loop on the environment for the specified number of steps.

    After training results are saved in the specified model file.

    :param model: a model object corresponding to a supported RL algorithm.
    :param params: algorithm parameters for a supported stable-baselines algorithm.
    :param args: command-line arguments (i.e., an argparse parser)
    :return: None
    """
    # add callbacks
    cp_callback = CheckpointCallback(save_freq=1000, save_path=params['save_dir'], name_prefix='model')

    # begin training
    start = time()
    model.learn(total_timesteps=args.steps, callback=cp_callback)
    end = time()

    print(f'elapsed time: {strftime("%H:%M:%S", gmtime(end - start))}')

    # save training results
    model.save(get_model_filepath(params, args, filename=args.model))
    env.save(get_model_filepath(params, args, filename='vec_normalize.pkl'))


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
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        print(f'done: {done}')


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


def main(env_id):
    args = parse_args()
    params = algorithm_params[args.algorithm]

    if args.verify:
        verify_env(env_id, args)

    session_id, session_path = init_session(env_id, params, args)

    env = DummyVecEnv([make_godot_env(session_path, env_id, i, args, seed=i) for i in range(args.n_agents)])

    env_stats_path = get_model_filepath(params, args, filename='vec_normalize.pkl')
    if env_stats_path.exists():
        env = VecNormalize.load(env_stats_path, env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0, clip_reward=100.0)

    # TODO: Can we automate hyper-parameter optimization?

    # TODO: Add callbacks to monitor the training process

    # TODO: Add feature and reward normalization using VecNormalize
    if args.purge:
        purge_model(params, args)

    model = init_model(session_path, params, env, args)

    if args.learn:
        learn(env, model, params, args)

    if args.evaluate:
        env = DummyVecEnv([make_godot_env(session_path, env_id, i, args) for i in range(1)])
        model = init_model(session_path, params, env, args)
        evaluate(model, env, args)

    if args.run:
        run(model, env, args)

    env.close()


if __name__ == "__main__":

    try:
        main(env_id='gym_godot:simple-animat-v0')
    except KeyboardInterrupt:
        print('session terminated by user!')
        sys.exit(1)
    except Exception as e:
        print(f'runtime exception raised: {e}!')
        print(traceback.format_exc())
        sys.exit(1)

    sys.exit(0)
