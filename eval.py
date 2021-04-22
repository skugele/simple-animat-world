import argparse
import os

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import gym
import numpy as np

import matplotlib.pyplot as plt

from time import time

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter

from stable_baselines.common.callbacks import BaseCallback


# TODO: Move the callbacks into a separate "callbacks.py" file
#############
# Callbacks #
#############
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class CustomCheckPointCallback(BaseCallback):
    def __init__(self, save_path, save_freq=1000, model_filename='model.zip', verbose=0):
        super(CustomCheckPointCallback, self).__init__(verbose)

        CHECKPOINT_PREFIX = '.'

        self.model_filename = model_filename
        self.model_path = save_path / self.model_filename

        self.save_freq = save_freq

        self.chkpt_dir = Path(save_path)
        self.chkpt_filename = CHECKPOINT_PREFIX + self.model_filename
        self.chkpt_filepath = self.chkpt_dir / self.chkpt_filename

    def _init_callback(self) -> None:
        # create folder if needed
        if self.chkpt_dir is not None:
            self.chkpt_dir.mkdir(parents=True, exist_ok=True)

        # remove previous temporary model files
        if self.chkpt_filepath.exists():
            if self.verbose:
                print(f'removing temporary file {self.chkpt_filepath}')

            self.chkpt_filepath.unlink()

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._update_checkpoint()

        return True

    def _on_training_end(self) -> None:
        # one final checkpoint to make sure everything is saved
        self._update_checkpoint()

        if self.verbose:
            print(
                f'renaming checkpoint file \'{self.chkpt_filename}\' to permanent model file \'{self.model_filename}\'')

        try:
            # workaround for WinError 138 that occurs when the file already exists
            if self.model_path.exists():
                self.model_path.unlink()

            # move temporary training file to permanent model file
            self.chkpt_filepath.rename(self.model_path)
        except Exception as e:
            print(f'rename of checkpoint file failed. cause: {e}')

    def _update_checkpoint(self) -> None:
        if self.verbose:
            print("saving model checkpoint to {}".format(self.chkpt_filepath))

        self.model.save(self.chkpt_filepath.absolute())


class EpisodicEnvMonitor(gym.Wrapper):
    """ A monitor wrapper for episodic Gym environments.

    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param freq: (Optional[int]) the number of time steps between file writes
    """

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str],
                 freq: Optional[int]):
        super(EpisodicEnvMonitor, self).__init__(env=env)

        self.filename = filename
        self.save_freq = freq

        self.needs_reset = True

        self.rewards = []

        self.steps_total = 0
        self.steps_since_save = 0

    def reset(self, **kwargs) -> np.ndarray:
        """ Resets the Gym environment.

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
        """
        if not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done.")

        self.rewards = []
        self.steps_since_save = 0
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        if done:
            self.needs_reset = True
            if len(self.rewards) > 0:
                r = np.array(self.rewards)

                # calculate statistics on rewards batch
                r_n, r_cumm, r_min, r_max = r.shape[0], np.sum(r), np.min(r), np.max(r)

                print(f'*** {time()}:\t[n = {r_n}, cumm. = {r_cumm:.3f}; min. = {r_min:.3f}; max. = {r_max:.3f}]',
                      flush=True)

                # save results to file
                if self.filename:
                    with open(self.filename, "a") as f:
                        f.write(f'{r_n}, {r_cumm}, {r_min}, {r_max}\n')

                self.rewards = []

        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        self.env.close()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(x, y, window_size=50, title='', x_label='', y_label='', show=True):
    y = moving_average(y, window=window_size)

    # truncate x following data reduction in moving average
    x = x[x.shape[0] - y.shape[0]:]

    _fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if show:
        plt.show()


def get_data_files_for_series(args):
    """ parse key/value pairs from command line into a dictionary """

    # {key:'series name', value:'array of file paths'}
    series_data_files_dict = {}

    if args.data:
        for item in args.data:
            key, value = map(lambda v: v.strip(), item.split('='))
            series_data_files_dict[key] = Path(value)

    return series_data_files_dict


def summary(args):
    raise NotImplementedError('Summary is not yet implemented')


def get_n_episodes(cumms_per_file):
    max_episodes = 0
    for episodes in cumms_per_file:
        max_episodes = max(max_episodes, len(episodes))

    return max_episodes


def plot(args):
    # {key:'series name', value:'array of episode-ordered values containing average episode rewards'}
    avg_per_episode = {}

    # TODO: refactor this to use pandas data frames or np arrays
    for series, path in get_data_files_for_series(args).items():
        avg_per_episode[series] = []

        episodes_per_instance = []
        for file in path.glob('*.csv'):
            episodes_per_instance.append([])
            with file.open(mode='r') as fp:

                # CSV fields in files are expected to be formatted as:
                #    (1) episode length
                #    (2) avg cumm reward
                #    (3) min reward per episode
                #    (4) max reward per episode
                for line in fp.readlines():
                    _n, cumm, _min, _max = map(lambda v: v.strip(), line.split(','))
                    # episodes_per_instance[-1].append(cumm)
                    episodes_per_instance[-1].append(_n)

                # print(f'cumms for episodes in file {file}')
                # print(episodes_per_instance[-1])

        n_agents = len(episodes_per_instance)
        n_episodes = get_n_episodes(episodes_per_instance)

        # this should have length n_episodes. Each index should contain an array of cumms (1 per agent)
        cumm_rewards_per_episode = []

        for episode_ndx in range(n_episodes):
            cumm_rewards_per_episode.append([])
            for instance_ndx in range(n_agents):
                # add cumms for this agent (if it reach that episode count)
                if episode_ndx < len(episodes_per_instance[instance_ndx]):
                    cumm_rewards_per_episode[episode_ndx].append(episodes_per_instance[instance_ndx][episode_ndx])

            # should be at most 75 values per episode (one for each agent that reached that episode number)
            # print(f'n values for episode {episode_ndx}: {len(cumm_rewards_per_episode[episode_ndx])}')
            # print(f'len: {len(cumm_rewards_per_episode[episode_ndx])}')
            if len(cumm_rewards_per_episode[episode_ndx]) < n_agents:
                break

            array_of_cumms = np.array(cumm_rewards_per_episode[episode_ndx], dtype=np.float64)
            avg_per_episode[series].append(np.average(array_of_cumms))

        ##################
        # begin plotting #
        ##################
        x = np.arange(len(avg_per_episode[series]))

        # generate avg episode length plot
        plot_results(x, y=avg_per_episode[series],
                     title=f'Average Episode Length (algorithm={series})',
                     x_label='Episode Number',
                     y_label='Avg. Episode Length (in steps)')

        # generate avg cumm reward per episode plot
        plot_results(x, y=avg_per_episode[series],
                     title=f'Average Cumulative Reward Per Episode (algorithm={series})',
                     x_label='Episode Number',
                     y_label='Avg. Cumulative Reward Per Episode')

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A driver script for the evaluation of an RL model')

    parser.add_argument('--data', metavar='KEY=VALUE', nargs='+', required=True,
                        help='a set of key value pairs. keys represent the names of the data series, and values'
                             'represent the file paths to the corresponding series data file(s) stored as CSV')

    # modes
    parser.add_argument('--summary', required=False, action="store_true",
                        help='provides summary statistics of the supplied data')
    parser.add_argument('--plot', required=False, action="store_true",
                        help='generates plot(s) of the supplied data ')

    args = parser.parse_args()

    if args.summary:
        summary(args)

    if args.plot:
        plot(args)

        # files = Path(f'tmp/{args.session_id}/monitor').glob('*.csv')
        # for file in files:
        #     steps, rewards = [a.flatten() for a in np.split(np.loadtxt(file, delimiter=','), 2, axis=1)]
        #     plot_results(x=steps, y=rewards)
