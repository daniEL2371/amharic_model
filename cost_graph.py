import numpy as np
from check_point_manager import CheckpointManager
import matplotlib.pyplot as plt

checkpoint = CheckpointManager()
s = "{0}_{1}".format("whole_model", "whole_train_256_2")
checkpoint.prepare(s)
def get_costs():
        rows = checkpoint.get_states()
        costs = []
        for row in rows:
            vals = row[1].split(',')
            cost = float(vals[3].split(': ')[1])
            costs.append(cost)
        return costs

costs = get_costs()

plt.plot(range(len(costs)), costs)
plt.show()