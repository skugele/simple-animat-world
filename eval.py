import argparse
import os

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import gym
import numpy as np
import pandas as pd

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


def plot_results(df, x_selector='x', y_selector='y', series_selector='series', window_size=50, title='', x_label='',
                 y_label='', show=True):
    _fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    legend_labels, legend_handles = [], []
    for series in df[series_selector].unique():
        series_df = df[df[series_selector] == series]

        x = series_df[x_selector]
        y = moving_average(series_df[y_selector], window=window_size)

        # truncate x following data reduction in moving average
        x = x[x.shape[0] - y.shape[0]:]

        hndl, = plt.plot(x, y, label=series)

        legend_labels.append(series)
        legend_handles.append(hndl)

    plt.legend(legend_handles, legend_labels)
    if show:
        plt.show()

    return plt


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

    # a combined data frame containing stats for all episodes, algorithms, and agents
    df = None

    for series, path in get_data_files_for_series(args).items():
        for agent_id, file in enumerate(path.glob('*.csv')):
            file_df = pd.read_csv(file, names=['ep_length', 'cumm_reward', 'min_reward', 'max_reward'])

            file_df['ep_number'] = np.arange(len(file_df))
            file_df['agent_id'] = [agent_id] * len(file_df)
            file_df['series'] = [series] * len(file_df)

            df = file_df if df is None else df.append(file_df)

    plot_df = df.groupby(['ep_number', 'series'], as_index=False)[['cumm_reward', 'ep_length']].mean()

    plot_results(plot_df, x_selector='ep_number', y_selector='cumm_reward', title='Average Cumulative Reward',
                 x_label='Episode', y_label='Cumulative Reward')
    plot_results(plot_df, x_selector='ep_number', y_selector='ep_length', title='Average Episode Length (in steps)',
                 x_label='Episode', y_label='Number of Steps')


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
