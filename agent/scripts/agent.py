import zmq
import json
import argparse
import sys
import os
import random
from time import sleep

DEFAULT_TIMEOUT = 5000  # in milliseconds

DEFAULT_ACTION_PORT = 5678
DEFAULT_SENSORS_PORT = 9001

N_ACTIONS = 4
ACTION_VALUES = [True, False]
ACTION_REPEAT_COUNT = 20


def parse_args():
    parser = argparse.ArgumentParser(description='A Network Client for Requesting Agent Actions in Godot')

    parser.add_argument('--id', type=int, required=True, help='the agent\'s identifier')
    parser.add_argument('--action_port', type=int, required=False, default=DEFAULT_ACTION_PORT,
                        help='the port number of the Godot action message listener')
    parser.add_argument('--sensors_port', type=int, required=False, default=DEFAULT_SENSORS_PORT,
                        help='the port number of the Godot sensor message publisher')
    parser.add_argument('--topic', metavar='TOPIC', required=False,
                        help='the topics to subscribe', default='')
    parser.add_argument('--verbose', required=False, action="store_true",
                        help='increases verbosity (displays requests & replies)')

    return parser.parse_args()


def establish_connection(args):
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:' + str(args.port))

    # without a timeout on receive the process can hang indefinitely
    socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

    return socket


def send_request(connection, request):
    request_as_json = json.dumps(request)
    if args.verbose:
        print("request: ", request_as_json)

    connection.send_string(request_as_json)


def receive_reply(connection, args):
    reply = connection.recv_json()
    if args.verbose:
        print("reply: ", reply)

    return reply


def build_action_message(id, actions):
    msg = dict()
    msg['agent_id'] = id
    msg['actions'] = actions

    return msg


def select_random_actions(n_actions):
    return random.choices(ACTION_VALUES, k=n_actions)


def split(m):
    """ Splits message into separate topic and content strings.
    :param m: a ZeroMq message containing the topic string and JSON content
    :return: a tuple containing the topic and JSON content
    """
    ndx = m.find('{')
    return m[0:ndx - 1], m[ndx:]


def establish_action_conn(context, args):
    socket = context.socket(zmq.REQ)

    conn_str = 'tcp://localhost:' + str(args.action_port)
    if args.verbose:
        print('establishing Godot action client using URL ', conn_str)

    socket.connect(conn_str)

    # configure timeout - without a timeout on receive the process can hang indefinitely
    socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

    return socket


def establish_sensor_conn(context, args):
    # establish subscriber connection
    socket = context.socket(zmq.SUB)

    # filters messages by topic
    socket.setsockopt_string(zmq.SUBSCRIBE, str(args.topic))

    # configure timeout
    socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)

    conn_str = 'tcp://localhost:' + str(args.sensors_port)
    if args.verbose:
        print('establishing Godot sensors subscriber using URL ', conn_str)
        if args.topic:
            print('filtering messages for topic: ', str(args.topic))

    socket.connect(conn_str)

    return socket


def establish_connections(args):
    context = zmq.Context()

    connections = {}
    connections['sensors'] = establish_sensor_conn(context, args)
    connections['actions'] = establish_action_conn(context, args)

    return connections


if __name__ == "__main__":
    try:
        # parse command line arguments
        args = parse_args()

        # establish connection to Godot action server
        connections = establish_connections(args)

        # create action request
        while True:

            # send actions
            request = build_action_message(id=args.id,
                                           actions=select_random_actions(N_ACTIONS))

            for i in range(ACTION_REPEAT_COUNT):
                send_request(connections['actions'], request)
                reply = receive_reply(connections['actions'], args)

                # receive sensor updates
                msg = connections['sensors'].recv_string()

                topic, content = split(msg)

                # unmarshal JSON message content
                obj = json.loads(content)

                if args.verbose:
                    print('new message on topic: ', topic)
                    print('header: ', obj['header'])
                    print('data: ', obj['data'])
                    print('')
                else:
                    print(content)

                # sleep(0.1)

    except KeyboardInterrupt:

        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
