import numpy as np
from check_point_manager import CheckpointManager
import matplotlib.pyplot as plt
import argparse

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='name', type=str)
    parser.add_argument('-t' '--model_tag', dest='tag', type=str)
    parser.add_argument('-s' '--samples', dest='n_samples', type=int, default=-1)
    args = parser.parse_args()
    return args

args = parse_training_args()

checkpoint = CheckpointManager()
s = "{0}_{1}".format(args.name, args.tag)
checkpoint.prepare(s)
def get_costs():
        rows = checkpoint.get_states()
        costs = []
        for row in rows:
            vals = row[1].split(',')
            cost = float(vals[3].split(': ')[1])
            costs.append(cost)
        return np.array(costs)

costs = get_costs()
s = int(len(costs)/args.n_samples)
if args.n_samples != -1:
    indexes = np.arange(0, len(costs), s)
    costs = costs[indexes]
print(len(costs))
plt.plot(range(len(costs)), costs)
plt.show()
