using Graphs, StatsBase, Plots, Random, SimpleWeightedGraphs, Statistics, JLD2
include(joinpath(@__DIR__, "route_network_structure.jl"))

# ---------- Routing graph and effectiveness ----------
function build_multiplex_routing_graph(fragments::Vector{Vector{Int}}, L::Int)
    num_physical_nodes = (L + 1)^2
    num_fragments = length(fragments)
    total_nodes = num_physical_nodes + num_fragments
    G_routing = SimpleWeightedDiGraph(total_nodes)
    for u in 1:num_physical_nodes
        ux, uy = (u - 1) % (L + 1), (u - 1) ÷ (L + 1)
        if ux < L
            v = u + 1
            add_edge!(G_routing, u, v, 1.0)
            add_edge!(G_routing, v, u, 1.0)
        end
        if uy < L
            v = u + (L + 1)
            add_edge!(G_routing, u, v, 1.0)
            add_edge!(G_routing, v, u, 1.0)
        end
    end
    alight_cost = 1e-8
    for (frag_idx, frag_nodes) in enumerate(fragments)
        frag_node_id = num_physical_nodes + frag_idx
        for station in frag_nodes
            add_edge!(G_routing, station, frag_node_id, 1.0)
            add_edge!(G_routing, frag_node_id, station, alight_cost)
        end
    end
    return G_routing, num_physical_nodes
end

function multiplex_global_efficiency(G_routing::SimpleWeightedDiGraph, num_physical_nodes::Int)
    eff = 0.0
    w = Graphs.weights(G_routing)
    for i in 1:num_physical_nodes
        ds = dijkstra_shortest_paths(G_routing, i, w).dists
        for j in (i+1):num_physical_nodes
            d = ds[j]
            if d > 0 && d < typemax(Float64)
                eff += 1.0 / d
            end
        end
    end
    return eff / (num_physical_nodes * (num_physical_nodes - 1) / 2)
end

# ---------- Structural connectivity and CCDF ----------
function line_graph_lcc_mass(fragments::Vector{Vector{Int}})
    if isempty(fragments)
        return 0.0
    end
    G = build_intersection_graph(fragments)
    if nv(G) == 0
        return 0.0
    end
    frag_lengths = [length(frag) - 1 for frag in fragments]
    comps = connected_components(G)
    return maximum(sum(frag_lengths[c]) for c in comps)
end

function line_graph_lcc_ratio(fragments::Vector{Vector{Int}}, lcc0_mass::Real)
    if lcc0_mass <= 0
        return 0.0
    end
    return line_graph_lcc_mass(fragments) / lcc0_mass
end

function compute_ccdf(degrees)
    counts = countmap(degrees)
    sorted_k = sort(collect(keys(counts)))
    total = length(degrees)
    ccdf_vals = Float64[]
    cum_count = total
    for k in sorted_k
        push!(ccdf_vals, cum_count / total)
        cum_count -= counts[k]
    end
    return sorted_k, ccdf_vals
end

function generate_log_p_targets()
    p1 = 10 .^ range(log10(0.01), log10(0.1), length=10)
    p2 = 10 .^ range(log10(0.1), log10(1.0), length=10)[2:end]
    return vcat([0.0], p1, p2)
end

function protocol5_default_p_targets()
    small_targets = exp10.(range(log10(0.01), log10(0.1), length=10))
    large_targets = collect(0.15:0.05:1.0)
    return sort(unique(vcat([0.0], small_targets, large_targets)))
end

# ---------- Fragment reconstruction ----------
function random_edge_fragments_at_p_minlen2(routes, removed; ell_min::Int=2)
    surviving_fragments = Vector{Vector{Int}}()
    for (i, he) in enumerate(routes)
        current_frag = [he[1]]
        for j in 1:(length(he)-1)
            if (i, j) in removed
                if length(current_frag) - 1 >= ell_min
                    push!(surviving_fragments, current_frag)
                end
                current_frag = [he[j+1]]
            else
                push!(current_frag, he[j+1])
            end
        end
        if length(current_frag) - 1 >= ell_min
            push!(surviving_fragments, current_frag)
        end
    end
    return surviving_fragments
end

function node_to_internal_segments(routes::Vector{Vector{Int}})
    node_segments = Dict{Int,Vector{Tuple{Int,Int}}}()
    for (h_id, he) in enumerate(routes)
        for j in 1:(length(he)-1)
            u, v = he[j], he[j+1]
            seg = (h_id, j)
            push!(get!(node_segments, u, Tuple{Int,Int}[]), seg)
            push!(get!(node_segments, v, Tuple{Int,Int}[]), seg)
        end
    end
    return node_segments
end

function fragments_after_node_attack(routes::Vector{Vector{Int}},
                                     removed_nodes::Set{Int}; ell_min::Int=2)
    surviving_fragments = Vector{Vector{Int}}()
    for line in routes
        current_fragment = Int[]
        for node in line
            if node in removed_nodes
                if length(current_fragment) - 1 >= ell_min
                    push!(surviving_fragments, current_fragment)
                end
                current_fragment = Int[]
            else
                push!(current_fragment, node)
            end
        end
        if length(current_fragment) - 1 >= ell_min
            push!(surviving_fragments, current_fragment)
        end
    end
    return surviving_fragments
end

# ---------- Protocol 1: random edge removal ----------
function execute_random_edge_attack(routes::Vector{Vector{Int}},
    p_targets::Vector{Float64}, L::Int, E_0::Float64,
    E_grid::Float64, total_edges::Int; ell_min::Int=2)

    num_phys = (L + 1)^2
    p_real, Q_vals, S1_vals = Float64[], Float64[], Float64[]
    lcc0_mass = line_graph_lcc_mass(routes)
    all_segments = Tuple{Int,Int}[]
    for (line_index, line) in enumerate(routes)
        for segment_index in 1:(length(line)-1)
            push!(all_segments, (line_index, segment_index))
        end
    end
    shuffled_segments = shuffle(all_segments)
    for target_p in p_targets
        if target_p == 0.0
            push!(p_real, 0.0)
            push!(Q_vals, 1.0)
            push!(S1_vals, 1.0)
            continue
        end
        number_directly_removed = min(round(Int, target_p * total_edges), total_edges)
        removed_set = Set(shuffled_segments[1:number_directly_removed])
        surviving_fragments = random_edge_fragments_at_p_minlen2(
            routes, removed_set; ell_min=ell_min)
        surviving_edges = sum(length(fragment) - 1
            for fragment in surviving_fragments; init=0)
        actual_p = (total_edges - surviving_edges) / total_edges
        G_damaged, _ = build_multiplex_routing_graph(surviving_fragments, L)
        E_p = multiplex_global_efficiency(G_damaged, num_phys)
        denominator = E_0 - E_grid
        Q_p = denominator > 0 ? max(0.0, (E_p - E_grid) / denominator) : 0.0
        S1_p = line_graph_lcc_ratio(surviving_fragments, lcc0_mass)
        push!(p_real, actual_p)
        push!(Q_vals, Q_p)
        push!(S1_vals, S1_p)
    end
    return p_real, Q_vals, S1_vals
end

# ---------- Protocols 2–5: full-route removal ----------
function execute_line_attack(routes::Vector{Vector{Int}}, sorted_indices::Vector{Int},
    p_targets::Vector{Float64}, L::Int, E_0::Float64,
    E_grid::Float64, total_edges::Int)

    num_phys = (L + 1)^2
    p_real, Q_vals, S1_vals = Float64[], Float64[], Float64[]
    lcc0_mass = line_graph_lcc_mass(routes)
    removed_flags = falses(length(routes))
    current_removed_edges = 0
    sort_idx = 1
    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(Q_vals, 1.0)
            push!(S1_vals, 1.0)
            continue
        end
        while current_removed_edges / total_edges < p_t &&
              sort_idx <= length(sorted_indices)
            he_idx = sorted_indices[sort_idx]
            removed_flags[he_idx] = true
            current_removed_edges += length(routes[he_idx]) - 1
            sort_idx += 1
        end
        actual_p = current_removed_edges / total_edges
        surviving_lines = routes[.!removed_flags]
        G_damaged, _ = build_multiplex_routing_graph(surviving_lines, L)
        E_p = multiplex_global_efficiency(G_damaged, num_phys)
        denominator = E_0 - E_grid
        Q_p = denominator > 0 ? max(0.0, (E_p - E_grid) / denominator) : 0.0
        S1_p = line_graph_lcc_ratio(surviving_lines, lcc0_mass)
        push!(p_real, actual_p)
        push!(Q_vals, Q_p)
        push!(S1_vals, S1_p)
    end
    return p_real, Q_vals, S1_vals
end

function random_longest_line_order(line_lengths::AbstractVector{<:Integer}, rng::AbstractRNG)
    removal_order = Int[]
    for line_length in sort(unique(line_lengths); rev=true)
        tied_lines = findall(==(line_length), line_lengths)
        shuffle!(rng, tied_lines)
        append!(removal_order, tied_lines)
    end
    return removal_order
end

# Adaptive degree: update after every removal.
function adaptive_degree_order(G_line::SimpleGraph, rng::AbstractRNG)
    number_lines = nv(G_line)
    initial_degrees = collect(degree(G_line))
    current_degrees = copy(initial_degrees)
    active = trues(number_lines)
    removal_order, degree_at_removal, tie_sizes = Int[], Int[], Int[]
    while length(removal_order) < number_lines
        active_indices = findall(active)
        isempty(active_indices) && break
        maximum_degree = maximum(current_degrees[active_indices])
        tied_lines = [line_index for line_index in active_indices
                      if current_degrees[line_index] == maximum_degree]
        chosen_line = rand(rng, tied_lines)
        push!(removal_order, chosen_line)
        push!(degree_at_removal, maximum_degree)
        push!(tie_sizes, length(tied_lines))
        active[chosen_line] = false
        for neighbor_line in neighbors(G_line, chosen_line)
            if active[neighbor_line]
                current_degrees[neighbor_line] -= 1
            end
        end
        current_degrees[chosen_line] = -1
    end
    return (order=removal_order, initial_degrees=initial_degrees,
        degree_at_removal=degree_at_removal, tie_sizes=tie_sizes)
end

function protocol5_active_subgraph(G_line::SimpleGraph, active_indices::Vector{Int})
    G_active = SimpleGraph(length(active_indices))
    global_to_local = zeros(Int, nv(G_line))
    for (local_i, global_i) in enumerate(active_indices)
        global_to_local[global_i] = local_i
    end
    for global_u in active_indices
        local_u = global_to_local[global_u]
        for global_v in neighbors(G_line, global_u)
            local_v = global_to_local[global_v]
            if local_v != 0 && local_u < local_v
                add_edge!(G_active, local_u, local_v)
            end
        end
    end
    return G_active
end

# Adaptive betweenness: recompute after every removal.
function adaptive_betweenness_order(G_line::SimpleGraph, rng::AbstractRNG;
                                    atol::Float64=1e-12, rtol::Float64=1e-10)
    number_lines = nv(G_line)
    initial_betweenness = betweenness_centrality(G_line)
    active = trues(number_lines)
    removal_order = Int[]
    betweenness_at_removal = Float64[]
    tie_sizes, active_counts_before_removal = Int[], Int[]
    while length(removal_order) < number_lines
        active_indices = findall(active)
        isempty(active_indices) && break
        if length(active_indices) <= 2
            bc_active = zeros(Float64, length(active_indices))
        else
            G_active = protocol5_active_subgraph(G_line, active_indices)
            bc_active = betweenness_centrality(G_active)
        end
        maximum_betweenness = maximum(bc_active)
        tied_local_indices = findall(
            x -> isapprox(x, maximum_betweenness; atol=atol, rtol=rtol), bc_active)
        chosen_local_index = rand(rng, tied_local_indices)
        chosen_line = active_indices[chosen_local_index]
        push!(removal_order, chosen_line)
        push!(betweenness_at_removal, maximum_betweenness)
        push!(tie_sizes, length(tied_local_indices))
        push!(active_counts_before_removal, length(active_indices))
        active[chosen_line] = false
    end
    return (order=removal_order, initial_betweenness=initial_betweenness,
        betweenness_at_removal=betweenness_at_removal, tie_sizes=tie_sizes,
        active_counts_before_removal=active_counts_before_removal)
end

# ---------- Protocol 6: initial occupancy ranking, random ties ----------
function random_shared_node_order(routes::Vector{Vector{Int}}, rng::AbstractRNG)
    node_to_routes = Dict{Int,Set{Int}}()
    for (h_id, he) in enumerate(routes)
        for node in unique(he)
            push!(get!(node_to_routes, node, Set{Int}()), h_id)
        end
    end
    occupancy_groups = Dict{Int,Vector{Int}}()
    for (node, route_ids) in node_to_routes
        shared_count = length(route_ids)
        push!(get!(occupancy_groups, shared_count, Int[]), node)
    end
    node_order = Int[]
    for shared_count in sort(collect(keys(occupancy_groups)); rev=true)
        tied_nodes = copy(occupancy_groups[shared_count])
        shuffle!(rng, tied_nodes)
        append!(node_order, tied_nodes)
    end
    return node_order
end

function execute_shared_node_order_QC(routes::Vector{Vector{Int}},
    node_order::Vector{Int}, p_targets::Vector{Float64}, L::Int,
    E_0::Float64, E_grid::Float64, total_edges::Int; ell_min::Int=2)

        issorted(p_targets) || error("p_targets must be sorted in ascending order")
    num_phys = (L + 1)^2
    node_segments = node_to_internal_segments(routes)
    removed_nodes = Set{Int}()
    removed_segments = Set{Tuple{Int,Int}}()
    actual_p, direct_p, Q_values, C_values = Float64[], Float64[], Float64[], Float64[]
    removed_node_counts = Int[]
    lcc0_mass = line_graph_lcc_mass(routes)
    node_index = 1
    for p_target in p_targets
        while length(removed_segments) / total_edges < p_target &&
              node_index <= length(node_order)
            node = node_order[node_index]
            push!(removed_nodes, node)
            for segment in get(node_segments, node, Tuple{Int,Int}[])
                push!(removed_segments, segment)
            end
            node_index += 1
        end
        current_direct_p = length(removed_segments) / total_edges
        surviving_fragments = fragments_after_node_attack(
            routes, removed_nodes; ell_min=ell_min)
        surviving_edges = sum(length(fragment) - 1
            for fragment in surviving_fragments; init=0)
        current_actual_p = (total_edges - surviving_edges) / total_edges
        if isempty(removed_nodes)
            Q_p, C_p = 1.0, 1.0
        else
            C_p = line_graph_lcc_ratio(surviving_fragments, lcc0_mass)
            G_damaged, _ = build_multiplex_routing_graph(surviving_fragments, L)
            E_p = multiplex_global_efficiency(G_damaged, num_phys)
            denominator = E_0 - E_grid
            Q_p = denominator > 0 ? max(0.0, (E_p - E_grid) / denominator) : 0.0
        end
        push!(actual_p, current_actual_p)
        push!(direct_p, current_direct_p)
        push!(Q_values, Q_p)
        push!(C_values, C_p)
        push!(removed_node_counts, length(removed_nodes))
    end
    return (actual_p=actual_p, direct_p=direct_p, Q=Q_values,
        C=C_values, removed_node_counts=removed_node_counts)
end

function interpolate_shared_node_curve(p_values::AbstractVector,
                                      y_values::AbstractVector, p_grid::AbstractVector)
    unique_p, unique_y = Float64[], Float64[]
    for index in eachindex(p_values)
        current_p = Float64(p_values[index])
        current_y = Float64(y_values[index])
        if isempty(unique_p) || current_p > unique_p[end] + 1e-12
            push!(unique_p, current_p)
            push!(unique_y, current_y)
        else
            unique_y[end] = current_y
        end
    end
    interpolated = Float64[]
    for query_p in p_grid
        if query_p <= unique_p[1]
            push!(interpolated, unique_y[1])
        elseif query_p >= unique_p[end]
            push!(interpolated, unique_y[end])
        else
            left_index = searchsortedlast(unique_p, query_p)
            p1, p2 = unique_p[left_index], unique_p[left_index+1]
            y1, y2 = unique_y[left_index], unique_y[left_index+1]
            weight = (query_p - p1) / (p2 - p1)
            push!(interpolated, y1 + weight * (y2 - y1))
        end
    end
    return interpolated
end

# ---------- Degree snapshots for S1 ----------
function line_attack_degree_snapshots(routes, removal_order, p_targets, total_edges)
    removed = falses(length(routes))
    removed_edges, order_index = 0, 1
    actual_p = zeros(length(p_targets))
    degree_vectors = Vector{Vector{Int}}(undef, length(p_targets))
    remaining_lengths = Vector{Vector{Int}}(undef, length(p_targets))
    remaining_line_counts = zeros(Int, length(p_targets))
    for (p_index, p_target) in enumerate(p_targets)
        while removed_edges / total_edges < p_target &&
              order_index <= length(removal_order)
            line_index = removal_order[order_index]
            removed[line_index] = true
            removed_edges += length(routes[line_index]) - 1
            order_index += 1
        end
        surviving_lines = routes[.!removed]
        G = build_intersection_graph(surviving_lines)
        actual_p[p_index] = removed_edges / total_edges
        degree_vectors[p_index] = collect(degree(G))
        remaining_lengths[p_index] = [length(line) - 1 for line in surviving_lines]
        remaining_line_counts[p_index] = length(surviving_lines)
    end
    return (actual_p=actual_p, degrees=degree_vectors,
        lengths=remaining_lengths, line_counts=remaining_line_counts)
end

function execute_shared_node_order_CCDF(routes::Vector{Vector{Int}},
    node_order::Vector{Int}, p_ccdf_targets::Vector{Float64},
    total_edges::Int; ell_min::Int=2)

        issorted(p_ccdf_targets) || error("p_ccdf_targets must be sorted in ascending order")
    node_segments = node_to_internal_segments(routes)
    removed_nodes = Set{Int}()
    removed_segments = Set{Tuple{Int,Int}}()
    node_index = 1
    actual_p_values, direct_p_values = Float64[], Float64[]
    removed_node_counts = Int[]
    degree_data, fragment_length_data = Vector{Vector{Int}}(), Vector{Vector{Int}}()
    for p_target in p_ccdf_targets
        surviving_fragments = fragments_after_node_attack(
            routes, removed_nodes; ell_min=ell_min)
        surviving_edges = sum(length(fragment) - 1
            for fragment in surviving_fragments; init=0)
        current_actual_p = (total_edges - surviving_edges) / total_edges
        while current_actual_p < p_target && node_index <= length(node_order)
            node = node_order[node_index]
            push!(removed_nodes, node)
            for segment in get(node_segments, node, Tuple{Int,Int}[])
                push!(removed_segments, segment)
            end
            node_index += 1
            surviving_fragments = fragments_after_node_attack(
                routes, removed_nodes; ell_min=ell_min)
            surviving_edges = sum(length(fragment) - 1
                for fragment in surviving_fragments; init=0)
            current_actual_p = (total_edges - surviving_edges) / total_edges
        end
        current_direct_p = length(removed_segments) / total_edges
        G_damaged = build_intersection_graph(surviving_fragments)
        current_degrees = collect(degree(G_damaged))
        current_lengths = [length(fragment) - 1 for fragment in surviving_fragments]
        push!(actual_p_values, current_actual_p)
        push!(direct_p_values, current_direct_p)
        push!(removed_node_counts, length(removed_nodes))
        push!(degree_data, current_degrees)
        push!(fragment_length_data, current_lengths)
    end
    return (actual_p=actual_p_values, direct_p=direct_p_values,
        removed_node_counts=removed_node_counts,
        degrees=degree_data, fragment_lengths=fragment_length_data)
end

# Reuse the same shuffled edge order as the Q/C run.
function random_edge_degree_snapshots(routes, shuffled_segments, p_targets,
                                     total_edges; ell_min::Int=2)
    actual_p = Float64[]
    degrees = Vector{Vector{Int}}()
    for p in p_targets
        n_remove = min(round(Int, p * total_edges), total_edges)
        removed = Set(shuffled_segments[1:n_remove])
        fragments = random_edge_fragments_at_p_minlen2(routes, removed; ell_min=ell_min)
        surviving_edges = sum(length(fragment) - 1 for fragment in fragments; init=0)
        push!(actual_p, (total_edges - surviving_edges) / total_edges)
        push!(degrees, collect(degree(build_intersection_graph(fragments))))
    end
    return (actual_p=actual_p, degrees=degrees)
end

# ---------- Compact experiment and summary wrappers ----------
function summarize_QC(p_runs, Q_runs, C_runs)
    runs = size(Q_runs, 1)
    p_mean = vec(mean(p_runs, dims=1))
    Q_mean = vec(mean(Q_runs, dims=1))
    C_mean = vec(mean(C_runs, dims=1))
    Q_sd = vec(std(Q_runs, dims=1))
    C_sd = vec(std(C_runs, dims=1))
    return (p=p_mean, Q=Q_mean, C=C_mean,
        Q_sd=Q_sd, C_sd=C_sd, Q_sem=Q_sd ./ sqrt(runs), C_sem=C_sd ./ sqrt(runs))
end

function run_resilience_case(routes::Vector{Vector{Int}}, L::Int, phi::Real;
    runs::Int=10, seed::Int=1234, ell_min::Int=2,
    p_targets=protocol5_default_p_targets(),
    shared_p_targets=generate_log_p_targets(),
    p_ccdf_targets=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95],
    outfile=nothing)

        runs >= 2 || error("At least two runs are required to calculate SD and SEM.")
    p_targets = Float64.(p_targets)
    shared_p_targets = Float64.(shared_p_targets)
    p_ccdf_targets = Float64.(p_ccdf_targets)
    issorted(p_targets) || error("p_targets must be sorted in ascending order")
    issorted(shared_p_targets) || error("shared_p_targets must be sorted in ascending order")
    issorted(p_ccdf_targets) || error("p_ccdf_targets must be sorted in ascending order")

    total_edges = sum(length(route) - 1 for route in routes)
    total_edges > 0 || error("routes must contain at least one edge")
    num_phys = (L + 1)^2
    G_line = build_intersection_graph(routes)
    G_0, _ = build_multiplex_routing_graph(routes, L)
    G_grid, _ = build_multiplex_routing_graph(Vector{Vector{Int}}(), L)
    E_0 = multiplex_global_efficiency(G_0, num_phys)
    E_grid = multiplex_global_efficiency(G_grid, num_phys)
    line_lengths = [length(route) - 1 for route in routes]
    protocols = [:random_edge, :random_line, :longest, :degree, :betweenness, :shared_node]
    results = Dict{Symbol,Any}()

    for protocol in protocols
        println("L=$L, phi=$phi, protocol=$protocol")
        targets = protocol == :shared_node ? shared_p_targets : p_targets
        p_runs = zeros(runs, length(targets))
        Q_runs = zeros(runs, length(targets))
        C_runs = zeros(runs, length(targets))
        raw_runs, degree_runs, orders = Any[], Any[], Any[]
        seeds = Int[]

        for run in 1:runs
            if protocol == :random_edge
                run_seed = seed + 100_000 * L + run
            elseif protocol == :random_line
                run_seed = seed + 200_000 * L + run
            elseif protocol == :longest
                run_seed = seed + 300_000 * L + 1_000 * round(Int, 100 * phi) + run
            elseif protocol == :degree
                run_seed = seed + 400_000 * L + 1_000 * round(Int, 100 * phi) + run
            elseif protocol == :betweenness
                run_seed = seed + 500_000 * L + 1_000 * round(Int, 100 * phi) + run
            else
                run_seed = seed + 100_000 * L + 1_000 * round(Int, 100 * phi) + run
            end
            push!(seeds, run_seed)
            rng = MersenneTwister(run_seed)

            if protocol == :random_edge
                Random.seed!(run_seed)
                all_segments = Tuple{Int,Int}[]
                for (i, route) in enumerate(routes)
                    for j in 1:(length(route)-1)
                        push!(all_segments, (i, j))
                    end
                end
                order = shuffle(all_segments)
                Random.seed!(run_seed)
                p, Q, C = execute_random_edge_attack(
                    routes, targets, L, E_0, E_grid, total_edges; ell_min=ell_min)
                snapshot = random_edge_degree_snapshots(
                    routes, order, p_ccdf_targets, total_edges; ell_min=ell_min)

            elseif protocol == :shared_node
                order = random_shared_node_order(routes, rng)
                qc = execute_shared_node_order_QC(
                    routes, order, targets, L, E_0, E_grid, total_edges; ell_min=ell_min)
                p, Q, C = qc.actual_p, qc.Q, qc.C
                snapshot = execute_shared_node_order_CCDF(
                    routes, order, p_ccdf_targets, total_edges; ell_min=ell_min)

            else
                if protocol == :random_line
                    Random.seed!(run_seed)
                    order = shuffle(collect(1:length(routes)))
                elseif protocol == :longest
                    order = random_longest_line_order(line_lengths, rng)
                elseif protocol == :degree
                    order = adaptive_degree_order(G_line, rng).order
                else
                    order = adaptive_betweenness_order(G_line, rng).order
                end
                p, Q, C = execute_line_attack(
                    routes, order, targets, L, E_0, E_grid, total_edges)
                snapshot = line_attack_degree_snapshots(
                    routes, order, p_ccdf_targets, total_edges)
            end

            push!(orders, order)
            push!(degree_runs, snapshot)
            push!(raw_runs, (p=p, Q=Q, C=C))
            if protocol == :shared_node
                p_runs[run, :] .= targets
                Q_runs[run, :] .= interpolate_shared_node_curve(p, Q, targets)
                C_runs[run, :] .= interpolate_shared_node_curve(p, C, targets)
            else
                p_runs[run, :] .= p
                Q_runs[run, :] .= Q
                C_runs[run, :] .= C
            end
        end

        results[protocol] = (
            summary=summarize_QC(p_runs, Q_runs, C_runs),
            p_targets=targets, p_runs=p_runs, Q_runs=Q_runs, C_runs=C_runs,
            raw_runs=raw_runs, degree_runs=degree_runs, orders=orders, seeds=seeds)
    end

    result = (L=L, phi=Float64(phi), routes=routes, total_edges=total_edges,
        E_0=E_0, E_grid=E_grid, runs=runs, seed=seed, ell_min=ell_min,
        p_ccdf_targets=p_ccdf_targets, protocols=results)
    if outfile !== nothing
        mkpath(dirname(abspath(outfile)))
        JLD2.jldsave(outfile; result=result)
    end
    return result
end

# ---------- Common plot styles ----------
function resilience_styles()
    return [
        (name=:random_edge, label="Random Edge", color=:gray, marker=:circle),
        (name=:random_line, label="Random Line", color=:black, marker=:cross),
        (name=:longest, label="Longest Line", color=:blue, marker=:square),
        (name=:degree, label="Adaptive Degree", color=:orange, marker=:diamond),
        (name=:betweenness, label="Adaptive Betweenness", color=:red, marker=:utriangle),
        (name=:shared_node, label="Shared Nodes", color=:green, marker=:star5)
    ]
end

# ---------- Figures 4 and 5: Q(p) and C(p) ----------
function plot_resilience_grid(cases; metric::Symbol=:Q,
    L_values=[25, 50, 75, 100], phi_values=[0.25, 0.50, 0.75, 0.95],
    error_type::Symbol=:sem, outfile=nothing)

    metric in (:Q, :C) || error("metric must be :Q or :C")
    error_type in (:sem, :sd) || error("error_type must be :sem or :sd")
    panels = []
    for (i, L) in enumerate(L_values)
        for (j, phi) in enumerate(phi_values)
            result = cases[(L, Float64(phi))]
            p = plot(xlims=(0, 1), ylims=(-0.02, 1.02),
                title=i == 1 ? "φ = $phi" : "",
                xlabel=i == length(L_values) ? "p" : "",
                ylabel=j == 1 ? "$(metric)(p), L=$L" : "",
                legend=(i == 1 && j == 1) ? :bottomleft : false,
                grid=true, framestyle=:box, legendfontsize=5)
            for style in resilience_styles()
                s = result.protocols[style.name].summary
                y = getproperty(s, metric)
                err = getproperty(s, Symbol(string(metric), "_", string(error_type)))
                plot!(p, s.p, y; yerror=err, label=style.label,
                    color=style.color, marker=style.marker,
                    markersize=2.5, linewidth=1.2)
            end
            push!(panels, p)
        end
    end
    fig = plot(panels...; layout=(length(L_values), length(phi_values)),
        size=(320 * length(phi_values), 260 * length(L_values)), margin=3Plots.mm)
    if outfile !== nothing
        mkpath(dirname(abspath(outfile)))
        savefig(fig, outfile)
    end
    display(fig)
    return fig
end

# ---------- Figure S1: degree CCDF after attacks ----------
# One selected run, with the actual loss at every snapshot retained in the data.
function plot_attack_ccdf_grid(cases;
    case_order=[(25, 0.25), (25, 0.95), (100, 0.25), (100, 0.95)],
    run::Int=1, outfile=nothing)

    panels = []
    for (row, style) in enumerate(resilience_styles())
        for (column, key) in enumerate(case_order)
            result = cases[key]
            snapshot = result.protocols[style.name].degree_runs[run]
            colors = palette(:tab10, length(result.p_ccdf_targets))
            p = plot(yaxis=:log10, grid=true, framestyle=:box,
                title=row == 1 ? "L=$(key[1]), φ=$(key[2])" : "",
                xlabel=row == 6 ? "Degree k" : "",
                ylabel=column == 1 ? "$(style.label)\nCCDF" : "",
                legend=(row == 1 && column == 1) ? :bottomleft : false,
                legendfontsize=5)
            for index in eachindex(result.p_ccdf_targets)
                degrees = snapshot.degrees[index]
                if isempty(degrees)
                    continue
                end
                k, ccdf = compute_ccdf(degrees)
                actual_p = snapshot.actual_p[index]
                plot!(p, k, ccdf; color=colors[index],
                    label="p=$(round(actual_p, digits=2))",
                    linewidth=1, marker=:circle, markersize=1.5)
            end
            push!(panels, p)
        end
    end
    fig = plot(panels...; layout=(6, length(case_order)),
        size=(300 * length(case_order), 1350), margin=3Plots.mm)
    if outfile !== nothing
        mkpath(dirname(abspath(outfile)))
        savefig(fig, outfile)
    end
    display(fig)
    return fig
end

# ---------- Figure S2: Q versus C ----------
function plot_Q_vs_C(cases; L::Int=100, phi_values=[0.25, 0.95], outfile=nothing)
    panels = []
    for (i, phi) in enumerate(phi_values)
        result = cases[(L, Float64(phi))]
        p = plot(xlabel="Structural connectivity C(p)",
            ylabel=i == 1 ? "Network effectiveness Q(p)" : "",
            title="φ = $phi", xlims=(-0.02, 1.02), ylims=(-0.02, 1.02),
            legend=i == 1 ? :bottomright : false, grid=true,
            framestyle=:box, legendfontsize=5)
        plot!(p, [0.0, 1.0], [0.0, 1.0]; color=:black,
            linestyle=:dash, linewidth=1, label="Q = C")
        for style in resilience_styles()
            s = result.protocols[style.name].summary
            plot!(p, s.C, s.Q; label=style.label, color=style.color,
                marker=style.marker, markersize=3, linewidth=1.4)
        end
        push!(panels, p)
    end
    fig = plot(panels...; layout=(1, length(phi_values)),
        size=(360 * length(phi_values), 320), margin=3Plots.mm)
    if outfile !== nothing
        mkpath(dirname(abspath(outfile)))
        savefig(fig, outfile)
    end
    display(fig)
    return fig
end