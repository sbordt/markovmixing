import markovmixing as mkm
import networkx as nx


def test_networkx_graphs():
	G = nx.path_graph(10)
	mc_srw = mkm.nx_graph_srw(G)
	mc_lswr = mkm.nx_graph_lazy_srw(G)

	G = nx.hypercube_graph(10)
	print nx.number_of_nodes(G)
	mkm.nx_graph_lazy_srw(G)

	G = nx.complete_graph(50)
	print mkm.nx_graph_nbrw(G).get_n()

if __name__=="__main__":
    test_networkx_graphs()