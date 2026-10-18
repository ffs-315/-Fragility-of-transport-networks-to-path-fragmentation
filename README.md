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

## usage
include("route_network_resilience.jl")
mkpath("results")

# Example seeds; not verified manuscript seeds.
network_seed = 1234
attack_seed = 5678

Random.seed!(network_seed)
routes = generate_network_to_fixed_phi(10, 0.25)

result = run_resilience_case(
    routes, 10, 0.25;
    runs=2, seed=attack_seed,
    outfile="results/example.jld2"
)

JLD2.jldsave(
    "results/example_network.jld2";
    routes=routes, L=10, phi=0.25, network_seed=network_seed
)

cases = Dict((10, 0.25) => result)
for metric in (:Q, :C)
    plot_resilience_grid(
        cases; metric=metric,
        L_values=[10], phi_values=[0.25],
        outfile="results/example_$(metric).pdf"
    )
end
