import markovmixing as mkm
import networkx as nx


def test_networkx_graphs():
	G = nx.path_graph(1000)
	mkm.nx_graph_analyze_srw(G)
	




if __name__=="__main__":
    test_networkx_graphs()