import zmq
import json
import argparse
import sys
import os
import random
from time import sleep

DEFAULT_TIMEOUT = 5000  # in milliseconds
DEFAULT_PORT = 5678

N_ACTIONS = 4
ACTION_VALUES = [True, False]


def parse_args():
    parser = argparse.ArgumentParser(description='A Network Client for Requesting Agent Actions in Godot')

    parser.add_argument('--id', type=int, required=True, help='the agent\'s identifier')
    parser.add_argument('--count', type=int, required=False, default=1,
                        help='the number of action requests to send')
    parser.add_argument('--port', type=int, required=False, default=DEFAULT_PORT,
                        help='the port number of the Godot action listener')
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


if __name__ == "__main__":
    try:
        # parse command line arguments
        args = parse_args()

        # establish connection to Godot action server
        connection = establish_connection(args)

        # create action request
        while True:
            request = build_action_message(id=args.id,
                                           actions=select_random_actions(N_ACTIONS))
            send_request(connection, request)
            reply = receive_reply(connection, args)
            sleep(0.05)

    except KeyboardInterrupt:

        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
