# MarkovMixing

Python package to determine the mixing behaviour of small Markov chains. Markov chains are represented explicitly by their transition matrices, using [scipy sparse matrices](https://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html). The mixing behaviour is determined by explicitly multiplying distributions on the state place with the transition matrix (many times). 

The package supports general Markov chains with the class MarkovChain. However so far the focus in on random walks on graphs. There is direct support for the [networkx](https://networkx.github.io/) graph package.

And now some examples!
    
## Examples


## References

   
["Markov Chains and Mixing Times" by Levin, Peres and Wilmer](http://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) is a good book.