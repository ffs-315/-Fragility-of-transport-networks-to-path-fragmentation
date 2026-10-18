# Fragility of transport networks to path fragmentation

This repository contains Julia code for generating route-based transport networks and analyzing their responses to disruption.

## Code organization

- **`route_network_structure.jl`**: network generation, route-intersection graphs, structural statistics, and growth diagnostics for Figures 1–2.
- **`route_network_resilience.jl`**: disruption simulations and plotting functions for Figures 4–5 and Supplementary Figures S1–S2. This file automatically includes `route_network_structure.jl`.

Keep both files in the same directory.

## Requirements

Install the required packages in Julia:

```julia
using Pkg
Pkg.add(["Graphs", "StatsBase", "Plots", "NetworkLayout",
         "SimpleWeightedGraphs", "JLD2"])
