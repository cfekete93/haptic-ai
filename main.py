#!/usr/bin/env python3

import argparse
import sys


def main(args: argparse.Namespace):
    command = getattr(args, 'command')
    match command:
        case 'build':
            build_ic()
        case 'test':
            _test(args)
        case _:
            print(f"error: unknown command '{command}'!")
            sys.exit(1)


def _test(args: argparse.Namespace):
    test_command = getattr(args, 'test-command')
    match test_command:
        case 'regular':
            test_ic()
        case 'api':
            test_sc()
        case default:
            print(f"error: unknown command '{test_command}'!")
            sys.exit(1)


def setup_argparse() -> argparse.ArgumentParser:
    # Setup parser
    parser = argparse.ArgumentParser(description='Examples on how to use HapticsDAO Intent Classifier')
    subparsers = parser.add_subparsers(dest='command', help='list of commands to run')
    subparsers.required = True

    # LOGIN command
    subparser = subparsers.add_parser('build', help='build intent classifier')

    # TEST command
    subparser = subparsers.add_parser('test', help='test intent classifier')
    testparsers = subparser.add_subparsers(dest='test-command', help='list of test commands to run')
    testparsers.required = True

    ## TEST - REGULAR command - run test on ic
    testparser = testparsers.add_parser('regular', help='run test directly against the underlying intent classifier')

    ## TEST - API command - run test on ic using web API
    testparser = testparsers.add_parser('api', help='run tests using the intent classifier through the REST API')

    return parser


def build_ic():
    print('test')
    return
    import intent_classifier as ic
    ic.generate_intent_classifier()


def test_ic():
    import intent_classifier as ic
    nlu = ic.IntentClassifier.load_intent_classifier()
    print(nlu.get_intent("is it cold in India right now"))
    print(nlu.get_intent("I would like buy bitcoin"))


def test_sc():
    from service_classifier import service_classifier as sc

    service = sc.parse_request("I would like buy bitcoin")
    print(service)
    service = sc.parse_request("I would like buy 100 bitcoin")
    print(service)
    service = sc.parse_request("I would like sell 37 ethereum")
    print(service)
    service = sc.parse_request("I would like see bitcoin")
    print(service)
    service = sc.parse_request("Show me dogecoin")
    print(service)
    service = sc.parse_request("What is btc doing")
    print(service)
    service = sc.parse_request("How are you today?")
    print(service)
    service = sc.parse_request("is it cold in India right now?")
    print(service)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = setup_argparse().parse_args()
    main(args)
