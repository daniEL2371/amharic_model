import numpy as np
import argparse

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128)
    parser.add_argument('-e' '--epochs', dest='epoches', type=float, default=1)
    parser.add_argument('-s', '--save_on', dest='save_on', default=1, type=int)
    parser.add_argument('-i', '--input', dest='input_file')
    parser.add_argument('-t', '--tag', dest='tag_name')
    args = parser.parse_args()
    return args

def one_hot_encode(val, length):
    a = np.zeros((length,))
    a[val] = 1
    return a

def highpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)