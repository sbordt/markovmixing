# MarkovMixing
Python package to determine the mixing behaviour of small Markov chains. Markov chains are represented explicitly by their transition matrices, using [scipy sparse matrices](https://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html). The mixing behaviour is determined by explicitly multiplying distributions on the state place with the transition matrix (many times). 

The package supports general Markov chains with the class MarkovChain. However so far the focus in on random walks on graphs. There is direct support for the [networkx](https://networkx.github.io/) graph package.

And now some examples!
    
## Examples
This example shows hot to explicitly use the package with a networkx graph.



    import markovmixing as mkm
    import networkx as nx
    import matplotlib.pyplot as plt

    # load a random 6-regular graph with 50.000 nodes from file
    G_6_regular = nx.read_sparse6('6_regular.s6')

    # get the adjacency matrix
    A = nx.to_scipy_sparse_matrix(G_6_regular)

    # transition matrix for SRW on the graph from the adjacency matrix 
    P = mkm.graph_srw_transition_matrix(A)

    # Markov chain with the transition marix
    mc = mkm.MarkovChain(P)

    # stationary distribution of SRW on a graph is deg(x)/2*|E|

    # add a random starting position to the Markov chain
    mc.add_distributions(mkm.random_delta_distributions(mc.get_n(),1))

    # determine the mixing in total variation 
    mc.compute_tv_mixing()

    # plot the mixing
    (x,tv) = mc.distribution_tv_mixing(0)
    plt.plot(x, tv)
    plt.xlabel("t")
    plt.ylabel("Distance to stationary distribution in total variation")
    plt.show()

## References

   
["Markov Chains and Mixing Times" by Levin, Peres and Wilmer](http://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) is a good book.