#!/usr/bin/env python3
import argparse
import os
import sys

from parser_lib import config
from parser_lib.benchmarks import generate_output
from parser_lib.util import get_benchmarks_by_name


class WebExtract:
    def __init__(self):
        """Main routine to run parser"""
        parser = argparse.ArgumentParser(
            description='WebExtract Parser',
            usage='''
                    {} <command> [<args>]
                    Available commands are:
                        parse           Parse benchmarks into intermediate format
                    '''.format('python run_parsing.py'))

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Not an available command')           
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def parse(self):
        """Parse benchmarks into intermediate format"""
        # Setup argparse
        parser = argparse.ArgumentParser(
            description=self.parse.__doc__)
        parser.add_argument(
            '--benchmarks')
        parser.add_argument('--benchmark-name', default=config.BENCHMARK_NAME)
        parser.add_argument(
            '--benchmark-folder', default=config.BENCHMARK_FOLDER)
        parser.add_argument(
            '--benchmark-output', default=config.BENCHMARK_OUTPUT)
        parser.add_argument(
            '--debug', action='store_true', default=False)

        # Parse arguments
        args = parser.parse_args(sys.argv[2:])
        benchmark_folder = os.path.join(os.getcwd(), args.benchmark_name, args.benchmark_folder)
        benchmark_output = os.path.join(os.getcwd(), args.benchmark_name, args.benchmark_output)

        if args.debug:
            config.DEBUG_MODE = True

        # Run desired benchmark
        to_run = get_benchmarks_by_name(benchmark_folder, args.benchmarks)
        for link in to_run:
            name = os.path.splitext(os.path.basename(link))[0]
            config.CURR_DOMAIN = name.split("_")[0]
            output = os.path.join(benchmark_output, '{}.json'.format(name))
            generate_output(link, output=output)


def main():
    WebExtract()

if __name__ == '__main__':
    main()
