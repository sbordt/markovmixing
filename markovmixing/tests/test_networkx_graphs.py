import markovmixing as mkm
import networkx as nx


def test_networkx_graphs():
	G = nx.path_graph(10)
	mkm.nx_graph_analyze_lazy_srw(G)

	G = nx.hypercube_graph(10)
	print nx.number_of_nodes(G)
	mkm.nx_graph_analyze_lazy_srw(G)

if __name__=="__main__":
    test_networkx_graphs()