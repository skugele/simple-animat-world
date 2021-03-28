import zmq
import numpy as np
import json
import gym

from time import sleep

from stable_baselines.common.env_checker import check_env

DEFAULT_TIMEOUT = 5000  # in milliseconds

STEP_REPEAT_COUNT = 10

DIM_OBSERVATIONS = 8
DIM_ACTIONS = 4

# Keys for connection dictionary
CONN_KEY_OBSERVATIONS = 'OBS'
CONN_KEY_ACTIONS = 'ACTIONS'

# TODO: Move this
def split(m):
    """ Splits message into separate topic and content strings.
    :param m: a ZeroMq message containing the topic string and JSON content
    :return: a tuple containing the topic and JSON content
    """
    ndx = m.find('{')
    return m[0:ndx - 1], m[ndx:]


# OpenAI documentation on creating custom environments:
#         https://github.com/openai/gym/blob/master/docs/creating-environments.md

# TODO: Split this off into another base class that this one inherits from that contains more general Godot concerns,
#       like connection management
class SimpleAnimatWorld(gym.Env):
    _args = None

    # Godot environment connections (using ZeroMQ)
    _connections = None

    def __init__(self, args=None):
        self._args = args

        self.action_space = gym.spaces.Discrete(2 ** DIM_ACTIONS - 1)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(STEP_REPEAT_COUNT * DIM_OBSERVATIONS,))

        # ZeroMQ connection context - shared by all network sockets
        context = zmq.Context()

        # establish connections (and fail fast if Godot process not running)
        self._connections = {
            CONN_KEY_OBSERVATIONS: self._establish_obs_conn(context, args),
            CONN_KEY_ACTIONS: self._establish_action_conn(context, args)
        }

    def step(self, action):
        print(f'taking step with action {action}', flush=True)

        observations = []  # a set of agent observations from Godot
        reward = 0.0  # always 0.0 -> this is determined in agent code
        done = False  # always False -> this is determined in agent code
        info = {}  # metadata about agent's observations

        # multiple executions and observations possible per call
        for i in range(STEP_REPEAT_COUNT):
            self._send_action_to_godot(action)
            meta, obs = self._receive_observation_from_godot()

            # there will be a set of observations and info if repeat count > 1
            observations.append([obs])
            info[i] = meta

        reward = self._calculate_reward(observations)
        print(f'reward: {reward}', flush=True)

        return np.concatenate(observations, axis=1)[0, :], reward, done, info

    def reset(self):
        """ respawns the agent within a running Godot environment and returns an initial observation """

        # TODO: Implement this! Current it returns an observation, but does not reset anything!
        # (1) Depends on respawn functionality in Godot environment and
        # (2) Mechanism to indicate a respawn is needed (perhaps in an action header?)

        observations = []  # a set of agent observations from Godot
        reward = 0.0  # always 0.0 -> this is determined in agent code
        done = False  # always False -> this is determined in agent code
        info = []  # metadata about agent's observations

        # multiple executions and observations possible per call
        for i in range(STEP_REPEAT_COUNT):
            meta, obs = self._receive_observation_from_godot()

            # there will be a set of observations and info if repeat count > 1
            observations.append([obs])
            info.append(meta)

        # return np.concatenate(observations, axis=0), reward, done, info
        return np.concatenate(observations, axis=1)[0, :]

    def render(self, mode='noop'):
        """ this is a noop. rendering is done in the Godot engine. """
        pass

    def close(self):
        # explicitly release ZeroMQ socket connections
        for conn in self._connections.values():
            conn.close()

    def verify(self):
        """ perform sanity checks on the environment """
        check_env(self, warn=True)

    def _establish_action_conn(self, context, args):
        socket = context.socket(zmq.REQ)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(args.action_port)
        if args.verbose:
            print('establishing Godot action client using URL ', conn_str)

        socket.connect(conn_str)

        # configure timeout - without a timeout on receive the process can hang indefinitely
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

        return socket

    def _establish_obs_conn(self, context, args):
        # establish subscriber connection
        socket = context.socket(zmq.SUB)

        # filters messages by topic
        socket.setsockopt_string(zmq.SUBSCRIBE, str(args.topic))

        # configure timeout
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(args.obs_port)
        if args.verbose:
            print('establishing Godot sensors subscriber using URL ', conn_str)
            if args.topic:
                print('filtering messages for topic: ', str(args.topic))

        socket.connect(conn_str)

        return socket

    def _send_action_to_godot(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()

        connection = self._connections[CONN_KEY_ACTIONS]

        request = {'id': self._args.id, 'action': int(action)}
        request_encoded = json.dumps(request)

        connection.send_string(request_encoded)
        reply = connection.recv_json()

    def _receive_observation_from_godot(self):
        connection = self._connections[CONN_KEY_OBSERVATIONS]

        # receive observation message (encoded as TOPIC + [SPACE] + json_encoded(PAYLOAD))
        topic, payload_enc = split(connection.recv_string())

        # unmarshal JSON message content into a dictionary
        payload = json.loads(payload_enc)
        return payload['header'], self._parse_observation(payload['data'])

    def _parse_observation(self, data):
        # print(f'data: {data}')

        obs = None
        try:
            data_as_list = data['SMELL'] + data['SOMATOSENSORY'] + [data['TOUCH']]
            obs = np.array(data_as_list) / 100.0
        except KeyError as e:
            print(f'exception {e} occurred in observation message {data}')

        return obs

    def _calculate_reward(self, obs):
        smell = np.concatenate(obs, axis=0)[:, 0]
        satiety = np.concatenate(obs, axis=0)[:, 6]
        # health = np.concatenate(obs, axis=0)[:, 4]

        satiety_multiplier = 10000
        # health_multiplier = 10
        smell_multiplier = 3000

        reward = 0.0

        smell_delta = smell[-1] - smell[0]
        satiety_delta = satiety[-1] - satiety[0]
        # health_delta = health[-1] - health[0]

        # reward stronger smells if did not eat
        if satiety_delta < 0:
            reward += smell_multiplier * smell_delta

        reward += round(satiety_multiplier * satiety_delta, 4)
        return reward
