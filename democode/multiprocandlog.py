"""
Description
___________

Multi-processing example which demonstrates several very important functionalities.
One is the ability to use multiprocessing. Two is the ability to log events instead of replying on print statements.
The last is the ability to demonstrate logging while multiprocessing.

log levels are in order from lowest to highest priority as follows
Notset -> debug -> info -> warning -> error -> critical

:Author: Cody (carobers@asu.edu)
:Date: 2023-07-26
:Copyright: 2023 Arizona State University
"""
import concurrent.futures as cf
import time
import logging

### Logging ###
# Configures the logger such that it prints to a screen and file including the format
__LOGFMT = "%(asctime)s|%(levelname)s|%(filename)s|%(lineno)d|%(funcName)s|%(message)s"

logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)
logger = logging.getLogger(__name__)
__logh = logging.FileHandler("./app.log")
logger.addHandler(__logh)
logger.log(100, __LOGFMT)
__logh.flush()
__logh.setFormatter(logging.Formatter(__LOGFMT))


def mp_func(iters: int, name: str):
    """
    my_func writes to log, and prints the ith iteration of a for loop. It then sleeps between each iteration before closing.

    :param iters: number of iterations to run in the for loop
    :type iters: int
    :param name: a string to print out
    :type name: str
    :return: Does not return anything
    """
    log = logger.getChild(__name__)
    log.debug(f"starting run {name} with iters={iters}")

    for i in range(iters):
        log.debug(f"{name}->i={i}")
        time.sleep(1)

    log.debug(f"return from mp_func {name}")


def main():
    log = logger.getChild(__name__)
    log.info("Beginning Test")
    with cf.ProcessPoolExecutor() as executor:
        executor.submit(mp_func, 5, "cat")
        executor.submit(mp_func, 2, "dog")
        executor.submit(mp_func, 3, "burrito")

    log.info("finished execution")
    log.warning("This is a warning")
    log.error("This is an error")
    log.critical("this is critical, the world may be on fire!")


if __name__ == "__main__":
    main()
