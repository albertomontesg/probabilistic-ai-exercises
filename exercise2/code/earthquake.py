import matplotlib.pyplot as plt

import core


def define_graph():
    g = core.BayesNet()
    g.add_nodes_from(['Earthquake', 'Burglar', 'Radio', 'Alarm', 'Phone'])
    g.add_edges_from([('Earthquake', 'Radio'), ('Earthquake', 'Alarm'), ('Burglar', 'Alarm'), ('Alarm', 'Phone')])
    return g

g = define_graph()
g.get_reachable('Radio', plot=True)
plt.show()
g.get_reachable('Radio', ['Phone'], plot=True)
plt.show()
g.get_reachable('Radio', ['Phone', 'Earthquake'], plot=True)
plt.show()
