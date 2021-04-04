import zmq
import numpy as np
import json
import gym

from time import sleep

from stable_baselines.common.env_checker import check_env

DEFAULT_TIMEOUT = 5000  # in milliseconds

STEP_REPEAT_COUNT = 1

# TODO: Set this from an observation from Godot
# DIM_OBSERVATIONS = 6 # SMELL -> 1 & 2, SOMATOSENSORY -> 3, TOUCH -> 4, VELOCITY -> 5 & 6
DIM_OBSERVATIONS = 2 # SMELL -> 1 & 2
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

    def __init__(self, agent_id, args=None):
        self._agent_id = agent_id
        self._args = args

        self._reset_history()

        self.action_space = gym.spaces.Discrete(2 ** DIM_ACTIONS - 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(DIM_OBSERVATIONS,))

        # ZeroMQ connection context - shared by all network sockets
        context = zmq.Context()

        # establish connections (and fail fast if Godot process not running)
        self._connections = {
            CONN_KEY_OBSERVATIONS: self._establish_obs_conn(context, args),
            CONN_KEY_ACTIONS: self._establish_action_conn(context, args)
        }

    def step(self, action):

        self._curr_step += 1

        if self._last_obs is None:
            _, self._last_obs = self._receive_observation_from_godot()

        # observations = []  # a set of agent observations from Godot
        info = {}  # metadata about agent's observations

        # for i in range(STEP_REPEAT_COUNT):
        #     meta, obs = self._receive_observation_from_godot()
        #     observations.append([obs])
        #     info[i] = meta

        self._send_action_to_godot(action)

        meta, obs = self._receive_observation_from_godot()

        wait_count = 0
        while self._last_obs_action < self._action_seqno and wait_count < 100:
            # print(f'waiting for godot -> {wait_count}', flush=True)
            meta, obs = self._receive_observation_from_godot()
            wait_count += 1

        info['godot_info'] = meta

        # TODO: This should be determined in the agent's code
        reward = self._calculate_reward(obs)
        done = self._check_done()

        # return np.concatenate(observations, axis=1)[0, :], reward, done, info


        self._last_obs = obs

        # FIXME: remove this
        obs = obs[:2]

        # if self._args.verbose:
        #     print('agent: {}, step: {}, action: {}, reward: {}, obs: {}'.format(
        #         self._agent_id, self._curr_step, action, reward, obs), flush=True)

        return obs, reward, done, info

    def reset(self):
        """ respawns the agent within a running Godot environment and returns an initial observation """

        self._reset_history()

        obs, _, _, _ = self.step(0)

        # TODO: Implement this! Currently it returns an observation, but does not reset anything!
        # (1) Depends on respawn functionality in Godot environment and
        # (2) Mechanism to indicate a respawn is needed (perhaps in an action header?)

        return obs

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

    def _reset_history(self):
        self._curr_step = 0
        self._max_step = self._args.steps_per_episode
        self._last_obs = None
        self._action_seqno = 0
        self._last_obs_action = -1

    def _check_done(self):
        if self._max_step == np.inf:
            return False

        return (self._max_step - self._curr_step) <= 0

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

    def _get_topic(self):
        return f'/agents/{self._agent_id}'

    def _establish_obs_conn(self, context, args):
        # establish subscriber connection
        socket = context.socket(zmq.SUB)

        # filters messages by topic
        socket.setsockopt_string(zmq.SUBSCRIBE, self._get_topic())

        # configure timeout
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(args.obs_port)
        if args.verbose:
            print('establishing Godot sensors subscriber using URL ', conn_str)

        socket.connect(conn_str)

        return socket

    def _create_action_message(self, action):
        header = {'id': self._agent_id, 'seqno': self._action_seqno}
        data = {'action': int(action)}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _send_action_to_godot(self, action):
        self._action_seqno += 1

        if isinstance(action, np.ndarray):
            action = action.tolist()

        connection = self._connections[CONN_KEY_ACTIONS]

        request_encoded = self._create_action_message(action)
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
            data_as_list = data['SMELL'] + [data['SOMATOSENSORY']] + [data['TOUCH']] + data['VELOCITY']

            # the last action executed
            self._last_obs_action = data['LAST_ACTION']

            obs = np.round(np.array(data_as_list), decimals=6)
        except KeyError as e:
            print(f'exception {e} occurred in observation message {data}')

        return obs

    def _calculate_reward(self, obs):
        # print(f'obs: {obs}, last_obs: {self._last_obs}')
        new_smell_intensity = obs[0] + obs[1]
        new_satiety = obs[2]
        new_touch = obs[3]

        old_smell_intensity = self._last_obs[0] + self._last_obs[1]
        old_satiety = self._last_obs[2]
        old_touch = self._last_obs[3]

        # health = np.concatenate(obs, axis=0)[:, 4]

        # satiety_multiplier = 10000
        # health_multiplier = 10
        # smell_multiplier = 100

        smell_delta = new_smell_intensity - old_smell_intensity
        # print(f'smell delta: {smell_delta}')

        # TODO: This should be non-linear (use a sigmoid function or something of that nature)
        satiety_delta = new_satiety - old_satiety
        # health_delta = health[-1] - health[0]

        reward = 0.0

        # reward += smell_delta
        reward += 1000  * smell_delta
        reward += 100 * satiety_delta

        # TODO: agent death?
        if new_satiety == 0:
            pass

        # reward stronger smells if did not eat
        # if satiety_delta <= 0:
        #     reward +=  smell_delta * 100 if smell_delta < 0 else smell_delta * 5

        return reward
