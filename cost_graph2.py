import numpy as np
from check_point_manager import CheckpointManager
import matplotlib.pyplot as plt
import argparse

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='name', type=str)
    parser.add_argument('-t' '--model_tag', dest='tag', type=str)
    args = parser.parse_args()
    return args

def pretty_time(seconds):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    return "{0}d, {1}h, {2}m, {3:.2f}s".format(days, hrs, mins, secs)

args = parse_training_args()

checkpoint = CheckpointManager()
s = "{0}_{1}".format(args.name, args.tag)
checkpoint.prepare(s)
def get_info():
        rows = checkpoint.get_states()
        info = {}
        time_taken = 0
        for row in rows:
            vals = row[1].split(',')
            t = float(vals[1].split(':')[1])
            for r in vals[2:]:
                if r == '':
                    break
                k, v = r.split(':')
                if k not in info:
                    info[k] = [v]
                else:
                    info[k].append(v)
            time_taken += t
        return info, time_taken

info, time_taken = get_info()
print(pretty_time(time_taken))
plt.subplot(121)
for key in info:
    if 'loss' in key:
        row = [round(float(x), 2) for x in info[key]]
        x = range(len(row))
        plt.ylabel('Loss')
        plt.xlabel('Epoches')
        plt.plot(x, row, label=key)
        plt.legend()
plt.subplot(122)
for key in info:
    if 'acc' in key:
        row = [round(float(x), 2) for x in info[key]]
        x = range(len(row))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoches')
        plt.plot(x, row, label=key)
        plt.legend()


plt.show()

