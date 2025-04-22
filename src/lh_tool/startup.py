import os
import time
import psutil
import argparse


def startup(pid, period, cmd):
    while psutil.pid_exists(pid):
        time.sleep(period)

    print(f"Start up: {cmd}")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pid", type=int, help="pid of waited process")
    parser.add_argument("-c", "--cmd", type=str, help="cmd to be executed")
    parser.add_argument("-p", "--period", type=int, default=600, help="monitoring period")
    opts = parser.parse_args()
    print(opts)

    startup(pid=opts.pid, period=opts.period, cmd=opts.cmd)


if __name__ == "__main__":
    main()
