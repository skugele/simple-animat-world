import os

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import gym
import numpy as np

import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter

from stable_baselines.common.callbacks import BaseCallback


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
            print(f'renaming checkpoint file \'{self.chkpt_filename}\' to permanent model file \'{self.model_filename}\'')

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


class NonEpisodicEnvMonitor(gym.Wrapper):
    """ A monitor wrapper for non-episodic Gym environments.

    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param freq: (Optional[int]) the number of time steps between file writes
    """

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str],
                 freq: Optional[int]):
        super(NonEpisodicEnvMonitor, self).__init__(env=env)

        # TODO: if the filename already exists, need to read its contents to get the starting timestep
        # TODO: this will occur with the user wants to continue a previous session
        self.filename = filename
        self.save_freq = freq

        # TODO: Can we dump the model parameters as well?

        self.needs_reset = True

        # TODO: This is extremely inefficient. Make it better!

        # Output data
        self.steps = []
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

        self.steps = []
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
        if done:
            self.needs_reset = True

        self.steps_total += 1
        self.steps.append(self.steps_total)

        self.rewards.append(reward)

        self.steps_since_save += 1
        if self.steps_since_save >= self.save_freq:
            out = np.column_stack((np.array(self.steps), np.array(self.rewards)))

            # save results to file
            with open(self.filename, "ab") as f:
                f.write(b'\n')
                np.savetxt(f, out, delimiter=",")

            # stats for watching enjoyment (FIXME: control this with a VERBOSE flag)
            print(
                f'*** steps: {self.steps_total}; reward: [cumm = {np.sum(out[:, 1])}; avg. = {np.average(out[:, 1])}; std. = {np.std(out[:, 1])}]',
                flush=True)

            self.steps_since_save = 0
            self.steps = []
            self.rewards = []

        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        self.env.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.steps_total

    def stats(self):
        pass


# Helper from the library
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DDPG LunarLander")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
