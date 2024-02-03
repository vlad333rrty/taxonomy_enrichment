import time


def measure(function):
    """

    :param function: executable
    :return: (delta, result)
    """
    start = time.time()
    res = function()
    end = time.time()
    return end - start, res
