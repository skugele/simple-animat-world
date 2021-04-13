import sys

import zmq
import numpy as np
import json
import gym

from stable_baselines.common.env_checker import check_env

DEFAULT_TIMEOUT = 5  # in milliseconds

# TODO: Set this from an observation from Godot
DIM_OBSERVATIONS = 6  # SMELL -> 1 & 2, SOMATOSENSORY -> 3, TOUCH -> 4, VELOCITY -> 5 & 6
# DIM_OBSERVATIONS = 2 # SMELL -> 1 & 2
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

    def __init__(self, agent_id, obs_port, action_port, args=None):
        print(f'debug setting: {args.debug}')
        self._agent_id = agent_id
        self._obs_port = obs_port
        self._action_port = action_port
        self._args = args

        self._curr_step = 0
        self._max_step = self._args.max_steps_per_episode
        self._curr_action_seqno = 0

        self._last_obs = None
        self._last_obs_action_seqno = -1

        self.action_space = gym.spaces.Discrete(2 ** DIM_ACTIONS - 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(DIM_OBSERVATIONS,))

        # ZeroMQ connection context - shared by all network sockets
        self._context = zmq.Context()

        # establish connections (and fail fast if Godot process not running)
        self._connections = {
            CONN_KEY_OBSERVATIONS: self._establish_obs_conn(args),
            CONN_KEY_ACTIONS: self._establish_action_conn(args)
        }

        self._joined_world = False
        self._terminal_state_reached = True

    def step(self, action):
        if self._check_done():
            print('error: attempting to step a \'done\' environment!', flush=True)
            return

        self._curr_step += 1

        self._send_action_to_godot(action)

        # wait for corresponding observation
        wait_count = 0
        meta, obs = self._receive_observation_from_godot()
        while self._last_obs_action_seqno < self._curr_action_seqno:
            meta, obs = self._receive_observation_from_godot()
            wait_count += 1

            # assume action message was lost and resend to Godot
            if wait_count % 5 == 0:
                print(f'agent {self._agent_id} resending action {action}!')
                self._send_action_to_godot(action)

        # remaining return values
        reward = self._calculate_reward(obs)
        done = self._check_done()
        info = {'godot_info': meta}  # metadata about agent's observations

        if self._args.debug:
            print(f'last_obs: {self._last_obs}\nobs: {obs}\nreward: {reward}\ndone: {done}\n')

        # this must be set after the call to _calculate_reward!
        self._last_obs = obs

        return obs, reward, done, info

    def reset(self):
        """ respawns the agent within a running Godot environment and returns an initial observation """

        if self._joined_world:
            self._send_quit_to_godot()

        self._send_join_to_godot()
        self._reset_history()

        # wait until observations start flowing for this agent id
        _, self._last_obs = self._receive_observation_from_godot()

        return self._last_obs

    def render(self, mode='noop'):
        """ this is a noop. rendering is done in the Godot engine. """
        pass

    def close(self):
        if self._joined_world:
            self._send_quit_to_godot()

        # explicitly release ZeroMQ socket connections
        for conn in self._connections.values():
            conn.close()

    def verify(self):
        """ perform sanity checks on the environment """
        check_env(self, warn=True)

    def _reset_history(self):
        self._curr_step = 0
        self._last_obs = None
        self._curr_action_seqno = 0
        self._last_obs_action_seqno = -1
        self._terminal_state_reached = False

    def _check_done(self):
        # check: terminal state
        if self._terminal_state_reached:
            return True

        # check: step based termination
        if (self._max_step - self._curr_step) < 0:
            return True

        return False

    def _establish_action_conn(self, args):
        socket = self._context.socket(zmq.REQ)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(self._action_port)

        if self._args.debug:
            print('establishing Godot action client using URL ', conn_str)

        socket.connect(conn_str)

        # configure timeout - without a timeout on receive the process can hang indefinitely
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

        return socket

    def _get_topic(self):
        return f'/agents/{self._agent_id}'

    def _establish_obs_conn(self, args):
        # establish subscriber connection
        socket = self._context.socket(zmq.SUB)

        # filters messages by topic
        socket.setsockopt_string(zmq.SUBSCRIBE, self._get_topic())

        # configure timeout
        socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

        # TODO: The hostname needs to be generalized to allow remote connections
        conn_str = 'tcp://localhost:' + str(self._obs_port)

        if self._args.debug:
            print('establishing Godot sensors subscriber using URL ', conn_str)

        socket.connect(conn_str)

        return socket

    def _create_action_message(self, action):
        header = {'type': 'action', 'id': self._agent_id, 'seqno': self._curr_action_seqno}
        data = {'action': int(action)}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _create_quit_message(self):
        header = {'type': 'quit', 'id': self._agent_id}
        data = {}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _create_join_message(self):
        header = {'type': 'join', 'id': self._agent_id}
        data = {}

        request = {'header': header, 'data': data}
        request_encoded = json.dumps(request)
        return request_encoded

    def _send(self, connection, message, max_tries=np.inf):
        wait_count = 0

        while wait_count < max_tries:
            try:
                connection.send_string(message)
            except zmq.error.Again:
                if self._args.debug:
                    print(f'Received EAGAIN: Godot was unavailable during send. Retrying.')

                wait_count += 1
            else:
                break

    def _receive_response(self, connection, as_json=False, timeout=DEFAULT_TIMEOUT, max_tries_before_reconnect=5):
        wait_count = 0

        message = None
        while wait_count < max_tries_before_reconnect:
            if (connection.poll(timeout)) & zmq.POLLIN != 0:
                message = connection.recv_json() if as_json else connection.recv_string()
            else:
                wait_count += 1

        return message

    def _reconnect(self, connection_type):
        print(f'reconnecting to server!', flush=True)

        connection = None
        if connection_type == CONN_KEY_OBSERVATIONS:
            connection = self._connections[connection_type] = self._establish_obs_conn(self._args)
        elif connection_type == CONN_KEY_ACTIONS:
            connection = self._connections[connection_type] = self._establish_action_conn(self._args)

        return connection

    def _receive(self, connection, as_json=False, max_tries=np.inf):
        wait_count = 0

        message = None
        while wait_count < max_tries:
            try:
                message = connection.recv_json() if as_json else connection.recv_string()
            except zmq.error.Again:
                if self._args.debug:
                    print(f'Received EAGAIN: Godot was unavailable during receive. Retrying.')

                wait_count += 1
            else:
                break

        return message

    def _send_message_to_action_server(self, message, timeout=DEFAULT_TIMEOUT, max_tries = np.inf):
        connection = self._connections[CONN_KEY_ACTIONS]

        retry_count = 0
        server_reply = None
        while server_reply is None and retry_count < max_tries:
            self._send(connection, message)
            server_reply = self._receive_response(connection, timeout=timeout, as_json=True)

            # server failed to send reply, attempt a reconnect
            if not server_reply:
                connection = self._reconnect(CONN_KEY_ACTIONS)
                retry_count += 1

        return server_reply is not None

    def _send_action_to_godot(self, action, max_tries=np.inf):
        self._curr_action_seqno += 1

        if isinstance(action, np.ndarray):
            action = action.tolist()

        message = self._create_action_message(action)
        if not self._send_message_to_action_server(message, timeout=10):
            raise RuntimeError(f'agent {self._agent_id} was unable to send action messsage to Godot! aborting.')

    def _send_quit_to_godot(self, max_tries=np.inf):
        if self._args.debug:
            print(f'agent {self._agent_id} is attempting to leave the world!', flush=True)

        message = self._create_quit_message()
        if not self._send_message_to_action_server(message, timeout=100):
            raise RuntimeError(f'agent {self._agent_id} was unable to send quit messsage to Godot! aborting.')

        # wait until observations stop flowing for this agent id
        _, obs = self._receive_observation_from_godot()
        while obs is not None:
            _, obs = self._receive_observation_from_godot()

        self._joined_world = False

        if self._args.debug:
            print(f'agent {self._agent_id} has left the world!', flush=True)

    def _send_join_to_godot(self, max_tries=np.inf):
        if self._args.debug:
            print(f'agent {self._agent_id} is attempting to join the world!', flush=True)

        message = self._create_join_message()
        if not self._send_message_to_action_server(message, timeout=100):
            raise RuntimeError(f'agent {self._agent_id} was unable to send join messsage to Godot! aborting.')

        self._joined_world = True

        if self._args.debug:
            print(f'agent {self._agent_id} has joined the world!', flush=True)

    def _receive_observation_from_godot(self, max_tries=np.inf):
        connection = self._connections[CONN_KEY_OBSERVATIONS]

        meta, obs = None, None

        # receive observation message (encoded as TOPIC + [SPACE] + json_encoded(PAYLOAD))
        message = self._receive(connection, max_tries=max_tries)
        if message:
            _, payload_enc = split(message)
            payload = json.loads(payload_enc)
            meta, obs = payload['header'], self._parse_observation(payload['data'])

        return meta, obs

    def _parse_observation(self, data):
        obs = None
        try:
            data_as_list = data['SMELL'] + [data['SOMATOSENSORY']] + [data['TOUCH']] + data['VELOCITY']

            # the last action executed
            self._last_obs_action_seqno = data['LAST_ACTION']

            obs = np.round(np.array(data_as_list), decimals=6)
        except KeyError as e:
            print(f'exception {e} occurred in observation message {data}', flush=True)

        return obs

    def _calculate_reward(self, obs):
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
        reward += 1000 * smell_delta
        reward += 100 * satiety_delta

        # TODO: agent death?
        if new_satiety == 0:
            reward -= 1000
            self._terminal_state_reached = True

        # reward stronger smells if did not eat
        # if satiety_delta <= 0:
        #     reward +=  smell_delta * 100 if smell_delta < 0 else smell_delta * 5

        return reward
