Fragility-of-transport-networks-to-path-fragmentation

This repository contains the Julia code used to analyze the Fragility of transport networks to path fragmentation.

The code is written in Julia and uses the following packages: Graphs, StatsBase, Plots, GraphRecipes, Random, NetworkLayout,
SimpleWeightedGraphs, Printf, DelimitedFiles, JLD2, Statistics.

The model generates transport networks by depositing edge-disjoint shortest paths on a two-dimensional square lattice. 
The resulting routes are analyzed using two representations: Network effectiveness Q(p), and network connectivity C(p).
The code includes six attack protocols: Random Edge Fragmentation, Random Line Removal, Length-based removal, Degree-based removal, 
Betweenness-centrality removal, and Transfer station removal.

The main functions are organized as follows:

1.Pathfinding and route deposition functions generate edge-disjoint shortest-path routes on the square lattice and store each route as an ordered sequence of lattice nodes.

2.Route-intersection functions build the line network, where each route is represented as one node and edges indicate shared physical stations between routes.

3.Structural-statistics functions compute the route length distribution, line network degree distribution, length-degree correlations, and deposition-time diagnostics used to characterize the growth of the network.

4.Compute the network effectiveness Q(p) from global efficiency.

6.Compute the network connectivity C(p) , defined as the largest connected component in the line network.

6.Attack-protocol functions implement the six disruption strategies and record the resulting Q(p) and C(p)curves.

7.Data-processing and plotting functions save simulation outputs in .jld2 format and generate the figures used in the manuscript.
