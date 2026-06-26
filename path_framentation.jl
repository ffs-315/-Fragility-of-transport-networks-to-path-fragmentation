using Graphs
using StatsBase
using Plots
using GraphRecipes
using Random
using NetworkLayout
using SimpleWeightedGraphs
using Printf
using DelimitedFiles
using JLD2
using Statistics

# --- 1. Pathfinding Algorithms ---

function manhattan_dist(u::Int, v::Int, L_nodes::Int)
    ux, uy = (u - 1) % L_nodes, (u - 1) ÷ L_nodes
    vx, vy = (v - 1) % L_nodes, (v - 1) ÷ L_nodes
    return abs(ux - vx) + abs(uy - vy)
end

function constrained_closest_path(L_R::SimpleGraph, A::Int, B::Int, min_len::Int, max_len::Int, L_nodes::Int)
    visited = Dict{Int,Int}(A => A)
    depths = Dict{Int,Int}(A => 0)
    queue = [A]

    best_node = -1
    min_dist = typemax(Int)

    while !isempty(queue)
        curr = popfirst!(queue)

        if depths[curr] >= min_len
            d = manhattan_dist(curr, B, L_nodes)
            if d < min_dist
                min_dist = d
                best_node = curr
            end

            if curr == B
                break
            end
        end

        if depths[curr] == max_len
            continue
        end

        for nbr in neighbors(L_R, curr)
            if !haskey(visited, nbr)
                visited[nbr] = curr
                depths[nbr] = depths[curr] + 1
                push!(queue, nbr)
            end
        end
    end

    path_nodes = Int[]
    path_edges = Tuple{Int,Int}[]

    if best_node != -1
        curr = best_node
        pushfirst!(path_nodes, curr)
        while curr != A
            p = visited[curr]
            pushfirst!(path_edges, (p, curr))
            pushfirst!(path_nodes, p)
            curr = p
        end
    end

    return path_nodes, path_edges
end

# --- 2. Network Generation ---

function generate_continuous_networks(L::Int, phi_targets::AbstractVector)
    L_nodes = L + 1
    L_R = Graphs.grid([L_nodes, L_nodes])

    hyperedges = Vector{Vector{Int}}()
    total_length = 0
    denominator = 2 * L * (L + 1)

    max_consecutive_fails = 10000
    fails = 0

    results = Dict{Float64,Vector{Vector{Int}}}()

    sorted_targets = sort(collect(phi_targets))
    target_idx = 1

    while target_idx <= length(sorted_targets)
        target_phi = sorted_targets[target_idx]
        current_phi = total_length / denominator

        if current_phi >= target_phi
            results[target_phi] = copy(hyperedges)
            target_idx += 1
            continue
        end

        A, B = rand(1:nv(L_R)), rand(1:nv(L_R))
        if A == B
            continue
        end

        path_nodes, path_edges = constrained_closest_path(L_R, A, B, 2, L, L_nodes)

        if isempty(path_edges)
            fails += 1
            if fails > max_consecutive_fails
                @warn "Jamming limit reached for L=$L. Simulation stuck at phi = $(round(current_phi, digits=3))"
                break
            end
            continue
        end

        fails = 0

        for (u, v) in path_edges
            rem_edge!(L_R, u, v)
        end

        push!(hyperedges, path_nodes)
        total_length += length(path_edges)
    end

    return results
end


function build_intersection_graph(hyperedges::Vector{Vector{Int}})
    N = length(hyperedges)
    G_line = SimpleGraph(N)

    # Precompute sets for fast intersection testing
    station_sets = [Set(he) for he in hyperedges]

    for i in 1:N
        for j in (i+1):N
            if !isdisjoint(station_sets[i], station_sets[j])
                add_edge!(G_line, i, j)
            end
        end
    end
    return G_line
end
# --- 3. Analysis & Visualization ---

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

function run_ensemble_ccdf(L::Int, phi_targets::AbstractVector, realizations::Int)
    println("Running $realizations realizations for L=$L...")

    # Initialize dictionary to accumulate all node degrees across realizations
    aggregated_degrees = Dict{Float64,Vector{Int}}()
    for phi in phi_targets
        aggregated_degrees[phi] = Int[]
    end

    for r in 1:realizations
        if r % 10 == 0
            println("Completed $r / $realizations realizations...")
        end

        snapshots = generate_continuous_networks(L, phi_targets)

        for phi in phi_targets
            if haskey(snapshots, phi)
                G = build_intersection_graph(snapshots[phi])
                # Append degrees from this realization to the master list
                append!(aggregated_degrees[phi], degree(G))
            end
        end
    end

    plt = plot(
        #title="Ensemble Degree CCDF (L=$L, Runs=$realizations)",
        #xlabel="Degree (k)",
        #ylabel="P(K >= k)",
        xaxis=:log10,  # <-- Added x-axis log scale
        yaxis=:log10,
        legend=:bottomleft,
        grid=true
    )

    for phi in sort(collect(phi_targets))
        degs = aggregated_degrees[phi]
        if isempty(degs)
            println("Warning: No data for phi = $phi. Jamming limit reached early across all runs.")
            continue
        end

        k_vals, ccdf_vals = compute_ccdf(degs)

        # <-- Filter out k=0 to prevent log(0) domain errors
        pos_idx = k_vals .> 0
        k_vals = k_vals[pos_idx]
        ccdf_vals = ccdf_vals[pos_idx]

        if isempty(k_vals)
            continue
        end

        plot!(plt, k_vals, ccdf_vals, label="phi = $(round(phi, digits=2))", marker=:circle, markersize=3, linewidth=1.5)
    end

    display(plt)
    return plt
end

function plot_ccdf(L::Int, phi_targets::AbstractVector)
    println("Generating networks for CCDF (L=$L)...")
    snapshots = generate_continuous_networks(L, phi_targets)

    plt = plot(
        title="Degree CCDF of Intersection Graph (L=$L, min_len=2)",
        xlabel="Degree (k)",
        ylabel="P(K >= k)",
        yaxis=:log10,
        legend=:bottomleft,
        grid=true
    )

    for phi in phi_targets
        if !haskey(snapshots, phi)
            continue
        end

        G = build_intersection_graph(snapshots[phi])
        degs = degree(G)

        if !isempty(degs)
            k_vals, ccdf_vals = compute_ccdf(degs)
            plot!(plt, k_vals, ccdf_vals, label="phi = $(round(phi, digits=2))", marker=:circle, markersize=3, linewidth=1.5)
        end
    end

    display(plt)
    return snapshots, plt
end

function generate_evolution_gif(snapshots, phi_targets=collect(0.25:0.05:0.90))

    #println("Running continuous deposition simulation (L=$L)...")
    #snapshots = generate_continuous_networks(L, phi_targets)

    println("Rendering GIF frames natively...")

    anim = @animate for phi in phi_targets
        if !haskey(snapshots, phi)
            continue
        end

        current_hyperedges = snapshots[phi]
        G = build_intersection_graph(current_hyperedges)

        if nv(G) == 0
            continue
        end

        # 1. Compute Layout directly (FIXED)
        pos = NetworkLayout.spring(G)
        x = [p[1] for p in pos]
        y = [p[2] for p in pos]

        # 2. Build edge coordinates with NaN separators
        edge_x = Float64[]
        edge_y = Float64[]
        for e in edges(G)
            push!(edge_x, x[src(e)], x[dst(e)], NaN)
            push!(edge_y, y[src(e)], y[dst(e)], NaN)
        end

        # 3. Logarithmic scaling for better visual balance
        node_sizes = [log(max(ℯ, length(he) - 1)) * 1.5 for he in current_hyperedges]

        # 4. Construct the plot manually
        plt = plot(
            title="Space R Evolution\nDensity (phi) = $(round(phi, digits=3))",
            framestyle=:none,
            legend=false,
            size=(600, 600)
        )

        # Draw edges
        if !isempty(edge_x)
            plot!(plt, edge_x, edge_y, linecolor=:gray, linewidth=0.5, linealpha=0.6)
        end

        # Draw nodes directly over the edges
        scatter!(plt, x, y,
            markersize=node_sizes,
            markercolor=:steelblue,
            markerstrokewidth=0.5,
            markerstrokecolor=:white
        )
    end

    output_file = "space_r_evolution.gif"
    gif(anim, output_file, fps=5)
    println("GIF successfully saved to $output_file")
end

# --- 4. Main Execution Block ---

function main(L)
    ccdf_phi_targets = collect(0.3:0.05:0.95)
    snapshots, final_plot = plot_ccdf(L, ccdf_phi_targets)
    generate_evolution_gif(snapshots, ccdf_phi_targets)
    return final_plot
end

function main_ensemble(L_size=50, num_realizations=100)
    # Define coarser targets for speed
    phi_targets = [0.25, 0.4, 0.55, 0.7, 0.85, 0.95]

    final_plot = run_ensemble_ccdf(L_size, phi_targets, num_realizations)
    return final_plot
end


# --- New Generation Routine for Fixed Phi ---

function generate_network_to_fixed_phi(L::Int, target_phi::Float64)
    L_nodes = L + 1
    L_R = Graphs.grid([L_nodes, L_nodes])

    hyperedges = Vector{Vector{Int}}()
    total_length = 0
    denominator = 2 * L * (L + 1)
    fails = 0

    while (total_length / denominator) < target_phi
        A, B = rand(1:nv(L_R)), rand(1:nv(L_R))
        if A == B
            continue
        end

        path_nodes, path_edges = constrained_closest_path(L_R, A, B, 2, L, L_nodes)

        if isempty(path_edges)
            fails += 1
            if fails > 5000
                @warn "Jamming limit reached before target phi for L=$L."
                break
            end
            continue
        end

        fails = 0
        for (u, v) in path_edges
            rem_edge!(L_R, u, v)
        end

        push!(hyperedges, path_nodes)
        total_length += length(path_edges)
    end

    return hyperedges
end

# --- New Analysis & Plotting Routines ---

function fit_exponential_ccdf(k_vals, ccdf_vals; min_prob=0.1)
    # Filter out the finite-size cutoff tail to isolate the linear regime
    valid_idx = ccdf_vals .>= min_prob
    x = k_vals[valid_idx]
    y = log.(ccdf_vals[valid_idx])

    if length(x) < 2
        return zeros(length(k_vals)), 0.0
    end

    # OLS Linear regression: ln(CCDF) = a - lambda * k
    A = hcat(ones(length(x)), x)
    coeffs = A \ y
    intercept = coeffs[1]
    lambda = -coeffs[2]

    # Generate the fitted line across the ENTIRE original k range so it 
    # draws correctly on the plot, extending past the cutoff point
    fit_y = exp.(intercept .- lambda .* k_vals)
    return fit_y, lambda
end


################ Fragmentation Resilience Analysis #######################
# --- 2. Multiplex Attack & Routing Mechanics ---

function fracture_hyperedges(hyperedges::Vector{Vector{Int}}, p::Float64)
    surviving_fragments = Vector{Vector{Int}}()

    for he in hyperedges
        current_frag = [he[1]]

        for i in 1:(length(he)-1)
            u, v = he[i], he[i+1]

            if rand() >= p
                push!(current_frag, v)
            else
                if length(current_frag) > 1
                    push!(surviving_fragments, current_frag)
                end
                current_frag = [v]
            end
        end

        if length(current_frag) > 1
            push!(surviving_fragments, current_frag)
        end
    end

    return surviving_fragments
end


########################## NEW FUNCTIONS ########################


function track_deposition_dynamics(L::Int, phi_targets::AbstractVector)
    L_nodes = L + 1
    L_R = Graphs.grid([L_nodes, L_nodes])

    hyperedges = Vector{Vector{Int}}()
    total_length = 0
    denominator = 2 * L * (L + 1)
    fails = 0

    sorted_targets = sort(collect(phi_targets))
    target_idx = 1

    # Data tracking
    birth_phis = Float64[]
    lengths = Int[]
    degrees_at_phi = Dict{Float64,Vector{Int}}()

    # NEW: Array to track the tortuosity ratio
    tortuosity_ratios = Float64[]
    #@warn "Max path length modified to L = L**2"
    while target_idx <= length(sorted_targets)
        target_phi = sorted_targets[target_idx]
        current_phi = total_length / denominator

        if current_phi >= target_phi
            G_line = build_intersection_graph(hyperedges)
            degrees_at_phi[target_phi] = degree(G_line)
            target_idx += 1
            continue
        end

        A, B = rand(1:nv(L_R)), rand(1:nv(L_R))
        if A == B
            continue
        end
        path_nodes, path_edges = constrained_closest_path(L_R, A, B, 2, L, L_nodes)

        if isempty(path_edges)
            fails += 1
            if fails > 10000
                @warn "Jamming limit reached at phi = $(round(current_phi, digits=3))"
                break
            end
            continue
        end

        fails = 0
        for (u, v) in path_edges
            rem_edge!(L_R, u, v)
        end

        # NEW: Calculate and store the tortuosity ratio
        ideal_len = manhattan_dist(A, B, L_nodes)
        actual_len = length(path_edges)
        push!(tortuosity_ratios, actual_len / ideal_len)

        push!(hyperedges, path_nodes)
        push!(lengths, actual_len)
        push!(birth_phis, current_phi)
        total_length += actual_len
    end

    # MODIFIED: Return the new array
    return birth_phis, lengths, degrees_at_phi, tortuosity_ratios
end

function plot_age_degree_scatter(L::Int)
    phi_targets = [0.25, 0.50, 0.75, 0.95]
    birth_phis, lengths, degrees_at_phi = track_deposition_dynamics(L, phi_targets)

    plots = []

    # We want a fixed color range so the colors mean the same thing across all subplots
    clims = (0.0, maximum(phi_targets))

    for phi in phi_targets
        if !haskey(degrees_at_phi, phi)
            continue
        end

        current_degrees = degrees_at_phi[phi]
        n_lines = length(current_degrees)

        # Slice the arrays to only include lines that existed at this target_phi
        current_lengths = lengths[1:n_lines]
        current_births = birth_phis[1:n_lines]

        p = scatter(
            current_lengths,
            current_degrees,
            zcolor=current_births,
            clims=clims,
            cmap=:viridis,
            markerstrokewidth=0,
            markersize=6,
            alpha=0.7,
            title="Phi = $phi",
            xlabel="Line Length",
            ylabel="Degree (Intersections)",
            legend=false,
            ylims=(0, 100),
            xlims=(0, 100),
            grid=true,
            aspect_ratio=1
        )
        push!(plots, p)
    end

    layout_dim = max(1, ceil(Int, length(plots) / 2))
    h1 = plot(plots..., layout=(layout_dim, 2), size=(900, 450 * layout_dim), margin=5Plots.mm)
    display(h1)
    savefig(h1, "age_degree_scatter_L$(L).pdf")

    return h1
end

# --- New Plotting Utilities for Confidence Bands ---

function interp1d(x::Vector{Float64}, y::Vector{Float64}, x_new::AbstractVector{Float64})
    y_new = zeros(length(x_new))
    if isempty(x) || isempty(y)
        return y_new
    end
    for (i, xn) in enumerate(x_new)
        if xn <= x[1]
            y_new[i] = y[1]
        elseif xn >= x[end]
            y_new[i] = y[end]
        else
            idx = searchsortedlast(x, xn)
            if idx == 0
                y_new[i] = y[1]
            elseif idx == length(x)
                y_new[i] = y[end]
            else
                t = (xn - x[idx]) / (x[idx+1] - x[idx])
                y_new[i] = y[idx] + t * (y[idx+1] - y[idx])
            end
        end
    end
    return y_new
end

function plot_confidence_bands!(p, x::AbstractVector{Float64}, Y_vecs::Vector{Vector{Float64}}, color::Symbol)
    if isempty(Y_vecs)
        return
    end

    # Convert Vector{Vector} to Matrix where each row is a realization
    Y = reduce(hcat, Y_vecs)'

    # Calculate statistics column by column
    mean_Y = Float64[]
    p025 = Float64[]
    p25 = Float64[]
    p75 = Float64[]
    p975 = Float64[]

    for i in 1:size(Y, 2)
        col = Y[:, i]
        push!(mean_Y, sum(col) / length(col))
        push!(p025, percentile(col, 2.5))
        push!(p25, percentile(col, 25.0))
        push!(p75, percentile(col, 75.0))
        push!(p975, percentile(col, 97.5))
    end

    # Plot 95% band (2.5th to 97.5th percentile)
    plot!(p, x, p975, fillrange=p025, color=color, alpha=0.2, linewidth=0, label="")
    # Plot 50% band (25th to 75th percentile)
    plot!(p, x, p75, fillrange=p25, color=color, alpha=0.4, linewidth=0, label="")
    # Plot Mean
    plot!(p, x, mean_Y, color=color, linewidth=2, label="")
end

function plot_moving_average_scaling(L_values::AbstractVector{Int}=[25, 100, 250, 500], realizations::Int=10, phi_target::Float64=0.95; save_file::String="moving_average_data.jld2")
    plots_len = []
    plots_tort = []
    all_data = Dict{Int,Vector{Dict{String,Any}}}()

    for L in L_values
        println("Running simulations for L = $L...")
        all_data[L] = Vector{Dict{String,Any}}()

        p_len = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Mean Length / L",
            legend=false,
            grid=true,
            ylim=(0.0, 1.0)
        )

        p_tort = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Tortuosity Ratio (Actual / Manhattan)",
            legend=false,
            grid=true,
            ylim=(0.0, 2.5)
        )

        for r in 1:realizations
            # MODIFIED: Unpack the new tortuosity_ratios array
            birth_phis, lengths, degrees_dict, tortuosity_ratios = track_deposition_dynamics(L, [phi_target])

            ma_tortuosity = Float64[]
            ma_lengths = Float64[]
            ma_phis = Float64[]

            if !isempty(tortuosity_ratios)
                w = min(max(1, length(tortuosity_ratios) ÷ 20), 100)

                if length(tortuosity_ratios) >= w
                    ma_lengths = Float64[sum(lengths[i:(i+w-1)]) / (w * L) for i in 1:(length(lengths)-w+1)]
                    ma_tortuosity = Float64[sum(tortuosity_ratios[i:(i+w-1)]) / w for i in 1:(length(tortuosity_ratios)-w+1)]
                    ma_phis = Float64[sum(birth_phis[i:(i+w-1)]) / w for i in 1:(length(birth_phis)-w+1)]

                    plot!(p_len, ma_phis, ma_lengths, linewidth=1.5, alpha=0.5, color=:steelblue, ylim=(0.0, 1.0))
                    plot!(p_tort, ma_phis, ma_tortuosity, linewidth=1.5, alpha=0.5, color=:darkorange, ylim=(0.0, 2.5))
                end
            end

            push!(all_data[L], Dict(
                "birth_phis" => birth_phis,
                "lengths" => lengths,
                "degrees" => get(degrees_dict, phi_target, Int[]),
                "ma_phis" => ma_phis,
                "ma_lengths" => ma_lengths,
                "ma_tortuosity" => ma_tortuosity # MODIFIED: Save the new data
            ))
        end

        hline!(p_len, [2 / 3], color=:black, linestyle=:dash, linewidth=2)
        hline!(p_tort, [1.0], color=:black, linestyle=:dash, linewidth=2)
        vline!(p_len, [0.5], color=:red, linestyle=:dot, linewidth=2)
        vline!(p_tort, [0.5], color=:red, linestyle=:dot, linewidth=2)

        push!(plots_len, p_len)
        push!(plots_tort, p_tort)
    end

    jldopen(save_file, "w") do file
        file["all_data"] = all_data
    end
    println("Data successfully saved to $save_file")

    all_plots = vcat(plots_len, plots_tort)
    final_plot = plot(all_plots..., layout=(2, length(L_values)), size=(400 * length(L_values), 800), margin=5Plots.mm)
    display(final_plot)
    pdf_filename = splitext(save_file)[1] * ".pdf"
    savefig(final_plot, pdf_filename)
    return final_plot, all_data
end

# --- NEW Function: Moving Average Scaling with Confidence Bands ---
function plot_moving_average_scaling_bands(L_values::AbstractVector{Int}=[25, 100, 250], realizations::Int=10, phi_target::Float64=0.95; save_file::String="moving_average_data_bands.jld2")
    plots_len = []
    plots_tort = []
    all_data = Dict{Int,Vector{Dict{String,Any}}}()

    # Create a uniform x-axis to interpolate individual paths onto
    phi_grid = collect(range(0.01, stop=phi_target, length=200))

    for L in L_values
        println("Running simulations for L = $L...")
        all_data[L] = Vector{Dict{String,Any}}()

        p_len = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Mean Length / L",
            legend=false,
            grid=true,
            ylim=(0.0, 1.0)
        )

        p_tort = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Tortuosity Ratio (Actual / Manhattan)",
            legend=false,
            grid=true,
            ylim=(0.0, 2.5)
        )

        lengths_interp = Vector{Vector{Float64}}()
        tort_interp = Vector{Vector{Float64}}()

        for r in 1:realizations
            birth_phis, lengths, degrees_dict, tortuosity_ratios = track_deposition_dynamics(L, [phi_target])

            ma_tortuosity = Float64[]
            ma_lengths = Float64[]
            ma_phis = Float64[]

            if !isempty(tortuosity_ratios)
                w = min(max(1, length(tortuosity_ratios) ÷ 20), 100)

                if length(tortuosity_ratios) >= w
                    ma_lengths = Float64[sum(lengths[i:(i+w-1)]) / (w * L) for i in 1:(length(lengths)-w+1)]
                    ma_tortuosity = Float64[sum(tortuosity_ratios[i:(i+w-1)]) / w for i in 1:(length(tortuosity_ratios)-w+1)]
                    ma_phis = Float64[sum(birth_phis[i:(i+w-1)]) / w for i in 1:(length(birth_phis)-w+1)]

                    if length(ma_phis) > 1
                        push!(lengths_interp, interp1d(ma_phis, ma_lengths, phi_grid))
                        push!(tort_interp, interp1d(ma_phis, ma_tortuosity, phi_grid))
                    end
                end
            end

            push!(all_data[L], Dict(
                "birth_phis" => birth_phis,
                "lengths" => lengths,
                "degrees" => get(degrees_dict, phi_target, Int[]),
                "ma_phis" => ma_phis,
                "ma_lengths" => ma_lengths,
                "ma_tortuosity" => ma_tortuosity
            ))
        end

        # Plot aggregated bands ONLY (No individual messy lines)
        plot_confidence_bands!(p_len, phi_grid, lengths_interp, :steelblue)
        plot_confidence_bands!(p_tort, phi_grid, tort_interp, :darkorange)

        hline!(p_len, [2 / 3], color=:black, linestyle=:dash, linewidth=2)
        hline!(p_tort, [1.0], color=:black, linestyle=:dash, linewidth=2)
        vline!(p_len, [0.5], color=:red, linestyle=:dot, linewidth=2)
        vline!(p_tort, [0.5], color=:red, linestyle=:dot, linewidth=2)

        push!(plots_len, p_len)
        push!(plots_tort, p_tort)
    end

    jldopen(save_file, "w") do file
        file["all_data"] = all_data
    end
    println("Data successfully saved to $save_file")

    all_plots = vcat(plots_len, plots_tort)
    final_plot = plot(all_plots..., layout=(2, length(L_values)), size=(400 * length(L_values), 800), margin=5Plots.mm)
    display(final_plot)
    pdf_filename = splitext(save_file)[1] * ".pdf"
    savefig(final_plot, pdf_filename)
    return final_plot, all_data
end

# --- New Function: Replot Saved Data ---

function replot_saved_data(filename::String="moving_average_data.jld2")
    println("Loading data from $filename...")
    all_data = JLD2.load(filename, "all_data")

    L_values = sort(collect(keys(all_data)))
    if isempty(L_values)
        println("No data found in $filename.")
        return nothing
    end

    plots_len = []
    plots_tort = []

    for L in L_values
        println("Replotting for L = $L...")

        p_len = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Mean Length / L",
            legend=false,
            grid=true,
            ylim=(0.0, 1.0)
        )

        p_tort = plot(
            title="L = $L",
            xlabel="Birth Phi",
            ylabel="Tortuosity Ratio (Actual / Manhattan)",
            legend=false,
            grid=true,
            ylim=(0.0, 2.5)
        )

        for realization_data in all_data[L]
            ma_phis = get(realization_data, "ma_phis", Float64[])
            ma_lengths = get(realization_data, "ma_lengths", Float64[])
            ma_tortuosity = get(realization_data, "ma_tortuosity", Float64[])

            # Backwards compatibility in case ma_lengths wasn't natively saved in older run
            if isempty(ma_lengths) && !isempty(ma_phis) && haskey(realization_data, "lengths")
                lengths = realization_data["lengths"]
                w = min(max(1, length(lengths) ÷ 20), 100)
                if length(lengths) >= w
                    ma_lengths = Float64[sum(lengths[i:(i+w-1)]) / (w * L) for i in 1:(length(lengths)-w+1)]
                end
            end

            # Backwards compatibility in case ma_tortuosity wasn't natively saved in older run
            if isempty(ma_tortuosity) && !isempty(ma_phis) && haskey(realization_data, "tortuosity_ratios")
                tr = realization_data["tortuosity_ratios"]
                w = min(max(1, length(tr) ÷ 20), 100)
                if length(tr) >= w
                    ma_tortuosity = Float64[sum(tr[i:(i+w-1)]) / w for i in 1:(length(tr)-w+1)]
                end
            end

            if !isempty(ma_phis) && length(ma_phis) > 1
                if !isempty(ma_lengths)
                    plot!(p_len, ma_phis, ma_lengths, linewidth=1.5, alpha=0.5, color=:steelblue, ylim=(0.0, 1.0))
                end
                if !isempty(ma_tortuosity)
                    plot!(p_tort, ma_phis, ma_tortuosity, linewidth=1.5, alpha=0.5, color=:darkorange, ylim=(0.0, 2.5))
                end
            end
        end

        hline!(p_len, [2 / 3], color=:black, linestyle=:dash, linewidth=2)
        hline!(p_tort, [1.0], color=:black, linestyle=:dash, linewidth=2)
        vline!(p_len, [0.5], color=:red, linestyle=:dot, linewidth=2)
        vline!(p_tort, [0.5], color=:red, linestyle=:dot, linewidth=2)

        push!(plots_len, p_len)
        push!(plots_tort, p_tort)
    end

    all_plots = vcat(plots_len, plots_tort)
    final_plot = plot(all_plots..., layout=(2, length(L_values)), size=(400 * length(L_values), 800), margin=5Plots.mm)
    display(final_plot)
    pdf_filename = splitext(filename)[1] * "_replot.pdf"
    savefig(final_plot, pdf_filename)
    return final_plot
end

# --- 2. Multiplex Attack & Routing Mechanics (Corrected) ---

function build_multiplex_routing_graph(fragments::Vector{Vector{Int}}, L::Int)
    num_physical_nodes = (L + 1)^2
    num_fragments = length(fragments)
    total_nodes = num_physical_nodes + num_fragments

    G_routing = SimpleWeightedDiGraph(total_nodes)

    # 1. Add Physical Layer (Walking costs 1)
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

    # 2. Add express Layer
    # Use a tiny epsilon for alighting so the sparse matrix doesn't delete the edge
    alight_cost = 1e-8

    for (frag_idx, frag_nodes) in enumerate(fragments)
        frag_node_id = num_physical_nodes + frag_idx

        for station in frag_nodes
            # Boarding
            add_edge!(G_routing, station, frag_node_id, 1.0)
            # Alighting 
            add_edge!(G_routing, frag_node_id, station, alight_cost)
        end
    end

    return G_routing, num_physical_nodes
end

function multiplex_global_efficiency(G_routing::SimpleWeightedDiGraph, num_physical_nodes::Int)
    eff = 0.0

    # Explicitly call the Graphs module to avoid the StatsBase collision
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
# --- 3. Experiment Execution ---

function run_fragmentation_resilience(L, target_phi)

    println("Generating master network at phi = $target_phi...")
    master_hyperedges = generate_network_to_fixed_phi(L, target_phi)
    num_phys = (L + 1)^2

    # 1. Baseline efficiency (p = 0)
    G_baseline, _ = build_multiplex_routing_graph(master_hyperedges, L)
    E_0 = multiplex_global_efficiency(G_baseline, num_phys)

    # 2. Worst-case efficiency (p = 1, physical grid only)
    # We simulate this by passing an empty array of fragments
    empty_fragments = Vector{Vector{Int}}()
    G_grid, _ = build_multiplex_routing_graph(empty_fragments, L)
    E_grid = multiplex_global_efficiency(G_grid, num_phys)

    println("Baseline Multiplex Efficiency E(0) = $(round(E_0, digits=4))")
    println("Physical Grid Efficiency E_grid = $(round(E_grid, digits=4))")

    p_vals = collect(0.0:0.05:1.0)
    Q_vals = Float64[]
    attack_iterations = 20

    println("Running hyperedge fragmentation simulations...")
    for p in p_vals
        current_E_sum = 0.0

        for _ in 1:attack_iterations
            surviving_fragments = fracture_hyperedges(master_hyperedges, p)
            G_damaged, _ = build_multiplex_routing_graph(surviving_fragments, L)
            current_E_sum += multiplex_global_efficiency(G_damaged, num_phys)
        end

        E_p = current_E_sum / attack_iterations

        # Bounded Normalization: Q(p) goes from 1.0 down to 0.0
        denominator = E_0 - E_grid
        if denominator > 0
            Q_p = (E_p - E_grid) / denominator
        else
            Q_p = 0.0 # Safety fallback if transit adds no value initially
        end

        # Clamp to 0 just in case floating point arithmetic drops it to -1e-16 at p=1
        push!(Q_vals, max(0.0, Q_p))
    end

    plt = plot(
        p_vals, Q_vals,
        title="Topological Resilience to Hyperedge Fragmentation\n(L=$L, phi=$target_phi)",
        xlabel="Fraction of Internal Edges Destroyed (p)",
        ylabel="Network Effectiveness Q(p)",
        marker=:diamond,
        markersize=4,
        linewidth=2,
        color=:purple,
        legend=false,
        grid=true
    )

    display(plt)
    return plt
end

# --- 2. Attack Simulators ---

# --- 1. Line Graph & Target Generation ---

function generate_log_p_targets()
    # 10 points logspaced from 0.01 (1%) to 0.1 (10%)
    p1 = 10 .^ range(log10(0.01), log10(0.1), length=10)
    # 9 points logspaced from 0.1 to 1.0 (excluding the duplicate 0.1)
    p2 = 10 .^ range(log10(0.1), log10(1.0), length=10)[2:end]

    # Prepend 0.0
    return vcat([0.0], p1, p2)
end


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

#Protocol 1: True Random Edge Removal (Fragmentation)

function execute_random_edge_attack(
    hyperedges::Vector{Vector{Int}},
    p_targets::Vector{Float64},
    L::Int,
    E_0::Float64,
    E_grid::Float64,
    total_edges::Int;
    ell_min::Int=2
)
    num_phys = (L + 1)^2

    p_real = Float64[]
    Q_vals = Float64[]
    S1_vals = Float64[]

    lcc0_mass = line_graph_lcc_mass(hyperedges)

    all_segments = Tuple{Int,Int}[]

    for (line_index, line) in enumerate(hyperedges)
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

        number_directly_removed = min(
            round(Int, target_p * total_edges),
            total_edges
        )

        removed_set = Set(
            shuffled_segments[1:number_directly_removed]
        )

        surviving_fragments = Vector{Vector{Int}}()

        for (line_index, line) in enumerate(hyperedges)
            current_fragment = [line[1]]

            for segment_index in 1:(length(line)-1)
                next_node = line[segment_index+1]

                if (line_index, segment_index) in removed_set

                    if length(current_fragment) - 1 >= ell_min
                        push!(surviving_fragments, current_fragment)
                    end

                    current_fragment = [next_node]
                else
                    push!(current_fragment, next_node)
                end
            end


            if length(current_fragment) - 1 >= ell_min
                push!(surviving_fragments, current_fragment)
            end
        end

        surviving_edges = sum(
            length(fragment) - 1
            for fragment in surviving_fragments;
            init=0
        )

        actual_p = (total_edges - surviving_edges) / total_edges

        G_damaged, _ = build_multiplex_routing_graph(
            surviving_fragments,
            L
        )

        E_p = multiplex_global_efficiency(G_damaged, num_phys)

        denominator = E_0 - E_grid

        Q_p = denominator > 0 ?
              max(0.0, (E_p - E_grid) / denominator) :
              0.0

        S1_p = line_graph_lcc_ratio(
            surviving_fragments,
            lcc0_mass
        )

        push!(p_real, actual_p)
        push!(Q_vals, Q_p)
        push!(S1_vals, S1_p)
    end

    return p_real, Q_vals, S1_vals
end

# Protocols 2-4: Targeted Full-Line Removal
function execute_line_attack(hyperedges::Vector{Vector{Int}}, sorted_indices::Vector{Int}, p_targets::Vector{Float64}, L::Int, E_0::Float64, E_grid::Float64, total_edges::Int)
    num_phys = (L + 1)^2
    p_real = Float64[]
    Q_vals = Float64[]
    S1_vals = Float64[]
    lcc0_mass = line_graph_lcc_mass(hyperedges)

    removed_flags = falses(length(hyperedges))
    current_removed_edges = 0
    sort_idx = 1

    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(Q_vals, 1.0)
            push!(S1_vals, 1.0)
            continue
        end

        # Remove lines in the sorted order until we hit or overshoot the target p
        while (current_removed_edges / total_edges) < p_t && sort_idx <= length(sorted_indices)
            he_idx = sorted_indices[sort_idx]
            removed_flags[he_idx] = true
            current_removed_edges += length(hyperedges[he_idx]) - 1
            sort_idx += 1
        end

        actual_p = current_removed_edges / total_edges
        surviving_lines = hyperedges[.!removed_flags]

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

# Protocol 5: Targeted shared lattice-node attack
# Attack lattice nodes passed by the largest number of hyperedges.

function sorted_lattice_nodes_by_hyperedge_occupancy(hyperedges::Vector{Vector{Int}})
    node_to_hyperedges = Dict{Int,Set{Int}}()

    for (h_id, he) in enumerate(hyperedges)
        for v in unique(he)
            s = get!(node_to_hyperedges, v, Set{Int}())
            push!(s, h_id)
        end
    end

    nodes = collect(keys(node_to_hyperedges))

    sorted_nodes = sort(
        nodes;
        by=v -> (length(node_to_hyperedges[v]), v),
        rev=true
    )

    return sorted_nodes, node_to_hyperedges
end

function node_to_internal_segments(hyperedges::Vector{Vector{Int}})
    node_segments = Dict{Int,Vector{Tuple{Int,Int}}}()

    for (h_id, he) in enumerate(hyperedges)
        for j in 1:(length(he)-1)
            u = he[j]
            v = he[j+1]
            seg = (h_id, j)

            push!(get!(node_segments, u, Tuple{Int,Int}[]), seg)
            push!(get!(node_segments, v, Tuple{Int,Int}[]), seg)
        end
    end

    return node_segments
end

function fragments_after_node_attack(
    hyperedges::Vector{Vector{Int}},
    removed_nodes::Set{Int};
    ell_min::Int=2
)
    surviving_fragments = Vector{Vector{Int}}()

    for line in hyperedges
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

function execute_intersection_node_attack(
    hyperedges::Vector{Vector{Int}},
    p_targets::Vector{Float64},
    L::Int,
    E_0::Float64,
    E_grid::Float64,
    total_edges::Int;
    ell_min::Int=2
)
    num_phys = (L + 1)^2

    sorted_nodes, node_to_hyperedges = sorted_lattice_nodes_by_hyperedge_occupancy(hyperedges)
    node_segments = node_to_internal_segments(hyperedges)

    removed_nodes = Set{Int}()
    removed_segments = Set{Tuple{Int,Int}}()

    p_real = Float64[]
    Q_vals = Float64[]
    S1_vals = Float64[]
    lcc0_mass = line_graph_lcc_mass(hyperedges)

    node_idx = 1

    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(Q_vals, 1.0)
            push!(S1_vals, 1.0)
            continue
        end

        while (length(removed_segments) / total_edges) < p_t &&
            node_idx <= length(sorted_nodes)

            node = sorted_nodes[node_idx]
            push!(removed_nodes, node)

            for seg in get(node_segments, node, Tuple{Int,Int}[])
                push!(removed_segments, seg)
            end

            node_idx += 1
        end

        #actual_p = length(removed_segments) / total_edges

        surviving_fragments = fragments_after_node_attack(
            hyperedges,
            removed_nodes;
            ell_min=ell_min
        )

        surviving_edges = sum(
            length(fragment) - 1
            for fragment in surviving_fragments;
            init=0
        )

        actual_p = (total_edges - surviving_edges) / total_edges

        S1_p = line_graph_lcc_ratio(surviving_fragments, lcc0_mass)

        G_damaged, _ = build_multiplex_routing_graph(surviving_fragments, L)
        E_p = multiplex_global_efficiency(G_damaged, num_phys)

        denominator = E_0 - E_grid
        Q_p = denominator > 0 ? max(0.0, (E_p - E_grid) / denominator) : 0.0

        push!(p_real, actual_p)
        push!(Q_vals, Q_p)
        push!(S1_vals, S1_p)
    end

    return p_real, Q_vals, S1_vals
end


# --- 3. Main Experiment Protocol (Updated with 5 Metrics) ---

function run_comparative_attack_protocols(L::Int, target_phi::Float64; save_dir=nothing)
    println("1. Generating Master Network (L=$L, phi=$target_phi)...")
    master_hyperedges = generate_network_to_fixed_phi(L, target_phi)

    total_edges = sum(length(he) - 1 for he in master_hyperedges)
    num_phys = (L + 1)^2

    println("2. Calculating Baselines & Topologies...")
    G_baseline, _ = build_multiplex_routing_graph(master_hyperedges, L)
    E_0 = multiplex_global_efficiency(G_baseline, num_phys)

    G_grid, _ = build_multiplex_routing_graph(Vector{Vector{Int}}(), L)
    E_grid = multiplex_global_efficiency(G_grid, num_phys)


    G_line = build_intersection_graph(master_hyperedges)

    println("3. Computing Target Sequences...")
    idx_length = sortperm([length(he) for he in master_hyperedges], rev=true)
    idx_degree = sortperm(degree(G_line), rev=true)

    println("   -> Computing betweenness centrality...")

    bc_scores = Graphs.betweenness_centrality(G_line)
    idx_betweenness = sortperm(bc_scores, rev=true)

    p_targets = generate_log_p_targets()

    println("4. Executing Attacks...")
    random_iterations = 10

    # --- Attack 1: Random Edge Fragmentation ---
    println("   -> Simulating Random Edge Attacks (Fragmentation)...")
    rand_edge_p_agg = zeros(length(p_targets))
    rand_edge_Q_agg = zeros(length(p_targets))
    rand_edge_S1_agg = zeros(length(p_targets))

    for _ in 1:random_iterations
        p_re, Q_re, S1_re = execute_random_edge_attack(master_hyperedges, p_targets, L, E_0, E_grid, total_edges)
        rand_edge_p_agg .+= p_re
        rand_edge_Q_agg .+= Q_re
        rand_edge_S1_agg .+= S1_re
    end
    p_rand_edge = rand_edge_p_agg ./ random_iterations
    Q_rand_edge = rand_edge_Q_agg ./ random_iterations
    S1_rand_edge = rand_edge_S1_agg ./ random_iterations

    # --- Attack 2: Random Line Removal ---
    println("   -> Simulating Random Line Attacks (Full Removal)...")
    rand_line_p_agg = zeros(length(p_targets))
    rand_line_Q_agg = zeros(length(p_targets))
    rand_line_S1_agg = zeros(length(p_targets))

    for _ in 1:random_iterations
        idx_rand_line = shuffle(1:length(master_hyperedges))
        p_rl, Q_rl, S1_rl = execute_line_attack(master_hyperedges, idx_rand_line, p_targets, L, E_0, E_grid, total_edges)
        rand_line_p_agg .+= p_rl
        rand_line_Q_agg .+= Q_rl
        rand_line_S1_agg .+= S1_rl
    end
    p_rand_line = rand_line_p_agg ./ random_iterations
    Q_rand_line = rand_line_Q_agg ./ random_iterations
    S1_rand_line = rand_line_S1_agg ./ random_iterations

    # --- Attacks 3-5: Targeted Line Removals ---
    println("   -> Simulating Targeted (Largest Lines)...")
    p_len, Q_len, S1_len = execute_line_attack(master_hyperedges, idx_length, p_targets, L, E_0, E_grid, total_edges)

    println("   -> Simulating Targeted (Highest Degree)...")
    p_deg, Q_deg, S1_deg = execute_line_attack(master_hyperedges, idx_degree, p_targets, L, E_0, E_grid, total_edges)

    println("   -> Simulating Targeted (Highest Betweenness)...")
    p_bet, Q_bet, S1_bet = execute_line_attack(master_hyperedges, idx_betweenness, p_targets, L, E_0, E_grid, total_edges)
    println("   -> Simulating Targeted (Shared Lattice Nodes)...")
    p_node, Q_node, S1_node = execute_intersection_node_attack(
        master_hyperedges,
        p_targets,
        L,
        E_0,
        E_grid,
        total_edges;
        ell_min=2
    )

    println("5. Plotting Results...")
    plt = plot(
        title="Comparative Resilience Metrics\n(L=$L, phi=$target_phi)",
        xlabel="Actual Fraction of Internal Edges Removed (p)",
        ylabel="Network Effectiveness Q(p)",
        legend=:bottomleft,
        grid=true,
        xaxis=:log10,
        framestyle=:box,
        foreground_color_border=:black,
        thickness_scaling=1,
        legendfontsize=10,
        titlefontsize=8,
        guidefontsize=8,
        tickfontsize=8,
        size=(350, 280),
        left_margin=2Plots.mm,
        bottom_margin=2Plots.mm,
        top_margin=1Plots.mm,
        right_margin=1Plots.mm
    )

    plot!(plt, [0.005; p_rand_edge[2:end]], Q_rand_edge, label="Random Edge (Avg)", marker=:circle, markersize=4, color=:gray, linewidth=2)
    plot!(plt, [0.005; p_rand_line[2:end]], Q_rand_line, label="Random Line (Avg)", marker=:cross, markersize=4, color=:black, linewidth=2, linestyle=:dash)
    plot!(plt, [0.005; p_len[2:end]], Q_len, label="Targeted: Largest", marker=:square, markersize=4, color=:blue, linewidth=2)
    plot!(plt, [0.005; p_deg[2:end]], Q_deg, label="Targeted: Degree", marker=:diamond, markersize=4, color=:orange, linewidth=2)
    plot!(plt, [0.005; p_bet[2:end]], Q_bet, label="Targeted: Betweenness", marker=:utriangle, markersize=4, color=:red, linewidth=2)
    plot!(
        plt,
        [0.005; p_node[2:end]],
        Q_node,
        label="Targeted: Shared Nodes",
        marker=:star5,
        markersize=5,
        color=:green,
        linewidth=2
    )
    display(plt)
    if save_dir !== nothing
        L_dir = joinpath(save_dir, "L=$(L)")
        mkpath(L_dir)
        phi_tag = replace(@sprintf("%.2f", target_phi), "." => "p")
        fig_file = joinpath(L_dir, "Q_L$(L)_phi$(phi_tag)_minlen2.pdf")
        savefig(plt, fig_file)
        println("Q figure saved to: ", fig_file)
        data_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag)_minlen2.jld2")

        JLD2.jldsave(
            data_file;
            L=L,
            target_phi=target_phi,
            p_targets=p_targets,
            p_rand_edge=p_rand_edge,
            Q_rand_edge=Q_rand_edge,
            S1_rand_edge=S1_rand_edge,
            p_rand_line=p_rand_line,
            Q_rand_line=Q_rand_line,
            S1_rand_line=S1_rand_line,
            p_largest=p_len,
            Q_largest=Q_len,
            S1_largest=S1_len,
            p_degree=p_deg,
            Q_degree=Q_deg,
            p_shared_node=p_node,
            Q_shared_node=Q_node,
            S1_shared_node=S1_node,
            master_hyperedges=master_hyperedges,
            p_betweenness=p_bet,
            Q_betweenness=Q_bet, S1_degree=S1_deg,
            S1_betweenness=S1_bet, bc_scores=bc_scores,
            idx_betweenness=idx_betweenness,
            E_0=E_0,
            E_grid=E_grid,
            total_edges=total_edges,
            ell_min=2,
            p_definition="fraction of original edges absent from valid surviving fragments"
        )

        println("Q data saved to: ", data_file)
    end

    return (
        hyperedges=master_hyperedges,
        targets=p_targets,
        random_edge=(p=p_rand_edge, Q=Q_rand_edge, S1=S1_rand_edge),
        random_line=(p=p_rand_line, Q=Q_rand_line, S1=S1_rand_line),
        length=(p=p_len, Q=Q_len, S1=S1_len),
        degree=(p=p_deg, Q=Q_deg, S1=S1_deg),
        betweenness=(p=p_bet, Q=Q_bet, S1=S1_bet),
        shared_node=(p=p_node, Q=Q_node, S1=S1_node),
        plot_Q=plt,
        #plot_S1=plt_s1
    )
end

function run_ensemble_semilog_ccdf_exp_fits(
    L::Int,
    phi_targets::AbstractVector,
    realizations::Int;
    min_prob::Float64=0.1,
    seed=nothing,
    outfile::Union{Nothing,String}=nothing
)
    if seed !== nothing
        Random.seed!(seed)
    end

    aggregated_degrees = Dict{Float64,Vector{Int}}()

    for phi in phi_targets
        aggregated_degrees[phi] = Int[]
    end

    for r in 1:realizations
        if r % 10 == 0
            println("Completed $r / $realizations realizations...")
        end

        snapshots = generate_continuous_networks(L, phi_targets)

        for phi in phi_targets
            if haskey(snapshots, phi)
                G = build_intersection_graph(snapshots[phi])
                append!(aggregated_degrees[phi], degree(G))
            end
        end
    end

    plt = Plots.plot(
        title="Semilog CCDF with exponential fits\nL=$L, runs=$realizations",
        xlabel="Degree k",
        ylabel="P(K >= k)",
        yaxis=:log10,
        legend=:bottomleft,
        grid=true
    )

    colors = Plots.palette(:tab10, length(phi_targets))
    fit_info = NamedTuple[]

    for (i, phi) in enumerate(sort(collect(phi_targets)))
        degs = aggregated_degrees[phi]

        if isempty(degs)
            println("No data for phi = $phi")
            continue
        end

        k_vals, ccdf_vals = compute_ccdf(degs)

        valid = (k_vals .> 0) .& (ccdf_vals .> 0)
        k_vals = k_vals[valid]
        ccdf_vals = ccdf_vals[valid]

        if length(k_vals) < 2
            continue
        end

        fit_vals, lambda = fit_exponential_ccdf(
            k_vals,
            ccdf_vals;
            min_prob=min_prob
        )

        Plots.scatter!(
            plt,
            k_vals,
            ccdf_vals;
            label="phi=$(round(phi, digits=2))",
            color=colors[i],
            markersize=3,
            markerstrokewidth=0,
            alpha=0.65
        )

        fit_valid = fit_vals .> 0

        Plots.plot!(
            plt,
            k_vals[fit_valid],
            fit_vals[fit_valid];
            label="exp fit λ=$(round(lambda, digits=3))",
            color=colors[i],
            linewidth=2,
            linestyle=:dash
        )

        push!(fit_info, (
            phi=phi,
            lambda=lambda,
            num_degrees=length(degs)
        ))
    end

    if outfile !== nothing
        mkpath(dirname(outfile))
        Plots.savefig(plt, outfile)
        println("Figure saved to: ", outfile)
    end

    display(plt)

    return (
        plot=plt,
        degrees=aggregated_degrees,
        fits=fit_info
    )
end

function run_Q_batch(L::Int)
    phi_list = [0.25, 0.50, 0.75, 0.95]
    save_dir = joinpath(@__DIR__, "Resiliencefig")

    results = Dict{Float64,Any}()

    for phi in phi_list
        println("\nRunning Q plot for L=$L, phi=$phi")
        results[phi] = run_comparative_attack_protocols(L, phi; save_dir=save_dir)
    end

    return results
end

function replot_Q_from_jld2(L::Int; save_dir=joinpath(@__DIR__, "Resiliencefig"))
    L_dir = joinpath(save_dir, "L=$(L)")
    files = sort(filter(f -> endswith(f, ".jld2") && occursin("Q_data", f), readdir(L_dir)))

    for file in files
        data = JLD2.load(joinpath(L_dir, file))

        target_phi = data["target_phi"]
        phi_tag = replace(@sprintf("%.2f", target_phi), "." => "p")

        #show_legend = target_phi == 0.25

        plt = plot(
            #title="Comparative Resilience Metrics\n(L=$L, phi=$target_phi)",
            #xlabel="Actual Fraction of Internal Edges Removed (p)",
            #ylabel="Network Effectiveness Q(p)",
            legend=false,
            grid=true,
            xaxis=:log10,
            framestyle=:box,
            foreground_color_border=:black,
            thickness_scaling=1,
            #legendfontsize=5,
            titlefontsize=8,
            guidefontsize=8,
            tickfontsize=8,
            size=(350, 280),
            left_margin=2Plots.mm,
            bottom_margin=2Plots.mm,
            top_margin=1Plots.mm,
            right_margin=1Plots.mm
        )

        plot!(plt, [0.005; data["p_rand_edge"][2:end]], data["Q_rand_edge"],
            label="Rand edge", marker=:circle, markersize=2, color=:gray, linewidth=1)

        plot!(plt, [0.005; data["p_rand_line"][2:end]], data["Q_rand_line"],
            label="Rand line", marker=:cross, markersize=2, color=:black, linewidth=1, linestyle=:dash)

        plot!(plt, [0.005; data["p_largest"][2:end]], data["Q_largest"],
            label="Largest", marker=:square, markersize=2, color=:blue, linewidth=1)

        plot!(plt, [0.005; data["p_degree"][2:end]], data["Q_degree"],
            label="Degree", marker=:diamond, markersize=2, color=:orange, linewidth=1)

        plot!(plt, [0.005; data["p_betweenness"][2:end]], data["Q_betweenness"],
            label="Betw.", marker=:utriangle, markersize=2, color=:red, linewidth=1)

        plot!(plt, [0.005; data["p_shared_node"][2:end]], data["Q_shared_node"],
            label="Shared nodes", marker=:star5, markersize=2, color=:green, linewidth=1)

        outfile = joinpath(L_dir, "Q_L$(L)_phi$(phi_tag)_notitle.pdf")
        savefig(plt, outfile)
        println("Saved: ", outfile)
    end
end

function execute_random_edge_attack_S1_only(hyperedges, p_targets, total_edges)
    p_real = Float64[]
    S1_vals = Float64[]
    lcc0_mass = line_graph_lcc_mass(hyperedges)

    all_segments = Tuple{Int,Int}[]
    for (i, he) in enumerate(hyperedges)
        for j in 1:(length(he)-1)
            push!(all_segments, (i, j))
        end
    end

    shuffled_segments = shuffle(all_segments)

    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(S1_vals, 1.0)
            continue
        end

        num_to_remove = min(round(Int, p_t * total_edges), length(shuffled_segments))
        removed_set = Set(shuffled_segments[1:num_to_remove])

        surviving_fragments = Vector{Vector{Int}}()

        for (i, he) in enumerate(hyperedges)
            current_frag = [he[1]]

            for j in 1:(length(he)-1)
                v = he[j+1]

                if in((i, j), removed_set)
                    if length(current_frag) > 1
                        push!(surviving_fragments, current_frag)
                    end
                    current_frag = [v]
                else
                    push!(current_frag, v)
                end
            end

            if length(current_frag) > 1
                push!(surviving_fragments, current_frag)
            end
        end

        push!(p_real, num_to_remove / total_edges)
        push!(S1_vals, line_graph_lcc_ratio(surviving_fragments, lcc0_mass))
    end

    return p_real, S1_vals
end

function execute_line_attack_S1_only(hyperedges, sorted_indices, p_targets, total_edges)
    p_real = Float64[]
    S1_vals = Float64[]
    lcc0_mass = line_graph_lcc_mass(hyperedges)

    removed_flags = falses(length(hyperedges))
    current_removed_edges = 0
    sort_idx = 1

    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(S1_vals, 1.0)
            continue
        end

        while (current_removed_edges / total_edges) < p_t && sort_idx <= length(sorted_indices)
            he_idx = sorted_indices[sort_idx]
            removed_flags[he_idx] = true
            current_removed_edges += length(hyperedges[he_idx]) - 1
            sort_idx += 1
        end

        surviving_lines = hyperedges[.!removed_flags]

        push!(p_real, current_removed_edges / total_edges)
        push!(S1_vals, line_graph_lcc_ratio(surviving_lines, lcc0_mass))
    end

    return p_real, S1_vals
end

function execute_intersection_node_attack_S1_only(hyperedges, p_targets, total_edges)
    sorted_nodes, _ = sorted_lattice_nodes_by_hyperedge_occupancy(hyperedges)
    node_segments = node_to_internal_segments(hyperedges)

    removed_nodes = Set{Int}()
    removed_segments = Set{Tuple{Int,Int}}()

    p_real = Float64[]
    S1_vals = Float64[]
    lcc0_mass = line_graph_lcc_mass(hyperedges)

    node_idx = 1

    for p_t in p_targets
        if p_t == 0.0
            push!(p_real, 0.0)
            push!(S1_vals, 1.0)
            continue
        end

        while (length(removed_segments) / total_edges) < p_t &&
            node_idx <= length(sorted_nodes)

            node = sorted_nodes[node_idx]
            push!(removed_nodes, node)

            for seg in get(node_segments, node, Tuple{Int,Int}[])
                push!(removed_segments, seg)
            end

            node_idx += 1
        end

        surviving_fragments = fragments_after_node_attack(
            hyperedges,
            removed_nodes;
            ell_min=2
        )

        surviving_edges = sum(
            length(fragment) - 1
            for fragment in surviving_fragments;
            init=0
        )

        actual_p = (total_edges - surviving_edges) / total_edges

        push!(p_real, actual_p)
        push!(S1_vals, line_graph_lcc_ratio(surviving_fragments, lcc0_mass))
    end

    return p_real, S1_vals
end

function run_S1_only_from_Q_data(
    L::Int;
    phi_list=[0.25, 0.50, 0.75, 0.95],
    save_dir=joinpath(@__DIR__, "Resiliencefig"),
    random_iterations=10
)
    L_dir = joinpath(save_dir, "L=$(L)")

    for target_phi in phi_list
        phi_tag = replace(@sprintf("%.2f", target_phi), "." => "p")
        q_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag).jld2")

        println("\nLoading Q data file: ", q_file)

        data = JLD2.load(q_file)
        master_hyperedges = data["master_hyperedges"]
        p_targets = data["p_targets"]
        total_edges = data["total_edges"]

        G_line = build_intersection_graph(master_hyperedges)

        idx_length = sortperm([length(he) for he in master_hyperedges], rev=true)
        idx_degree = sortperm(degree(G_line), rev=true)

        if haskey(data, "idx_betweenness")
            idx_betweenness = data["idx_betweenness"]
        else
            bc_scores = Graphs.betweenness_centrality(G_line)
            idx_betweenness = sortperm(bc_scores, rev=true)
        end

        rand_edge_p_agg = zeros(length(p_targets))
        rand_edge_S1_agg = zeros(length(p_targets))

        for _ in 1:random_iterations
            p_re, S1_re = execute_random_edge_attack_S1_only(master_hyperedges, p_targets, total_edges)
            rand_edge_p_agg .+= p_re
            rand_edge_S1_agg .+= S1_re
        end

        p_rand_edge = rand_edge_p_agg ./ random_iterations
        S1_rand_edge = rand_edge_S1_agg ./ random_iterations

        rand_line_p_agg = zeros(length(p_targets))
        rand_line_S1_agg = zeros(length(p_targets))

        for _ in 1:random_iterations
            idx_rand_line = shuffle(1:length(master_hyperedges))
            p_rl, S1_rl = execute_line_attack_S1_only(master_hyperedges, idx_rand_line, p_targets, total_edges)
            rand_line_p_agg .+= p_rl
            rand_line_S1_agg .+= S1_rl
        end

        p_rand_line = rand_line_p_agg ./ random_iterations
        S1_rand_line = rand_line_S1_agg ./ random_iterations

        p_len, S1_len = execute_line_attack_S1_only(master_hyperedges, idx_length, p_targets, total_edges)
        p_deg, S1_deg = execute_line_attack_S1_only(master_hyperedges, idx_degree, p_targets, total_edges)
        p_bet, S1_bet = execute_line_attack_S1_only(master_hyperedges, idx_betweenness, p_targets, total_edges)
        p_node, S1_node = execute_intersection_node_attack_S1_only(master_hyperedges, p_targets, total_edges)

        s1_file = joinpath(L_dir, "S1_data_L$(L)_phi$(phi_tag).jld2")

        JLD2.jldsave(
            s1_file;
            L=L,
            target_phi=target_phi,
            p_targets=p_targets, p_rand_edge=p_rand_edge,
            S1_rand_edge=S1_rand_edge, p_rand_line=p_rand_line,
            S1_rand_line=S1_rand_line, p_largest=p_len,
            S1_largest=S1_len, p_degree=p_deg,
            S1_degree=S1_deg, p_betweenness=p_bet,
            S1_betweenness=S1_bet, p_shared_node=p_node,
            S1_shared_node=S1_node, master_hyperedges=master_hyperedges,
            total_edges=total_edges
        )

        println("S1 data saved to: ", s1_file)
    end
end


function replot_all_S1_mixed_data(
    L_values=[25, 50, 75];
    phi_list=[0.25, 0.50, 0.75, 0.95],
    save_dir=joinpath(@__DIR__, "Resiliencefig")
)
    for L in L_values
        L_dir = joinpath(save_dir, "L=$(L)")

        for phi in phi_list
            phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

            s1_file = joinpath(L_dir, "S1_data_L$(L)_phi$(phi_tag).jld2")
            q_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag).jld2")

            data_file = if isfile(s1_file)
                s1_file
            elseif isfile(q_file)
                q_file
            else
                println("Missing data for L=$L, phi=$phi")
                continue
            end

            println("Loading: ", data_file)
            data = JLD2.load(data_file)

            plt = plot(
                #title="Line-Graph LCC / Hyperedge Path Percolation\n(L=$L, phi=$phi)",
                #xlabel="Actual Fraction of Internal Edges Removed (p)",
                #ylabel="S1(p)",
                legend=false,
                grid=true,
                xaxis=:log10,
                ylims=(0, 1.05),
                framestyle=:box,
                foreground_color_border=:black,
                thickness_scaling=1,
                #legendfontsize=5,
                titlefontsize=8,
                guidefontsize=8,
                tickfontsize=8,
                size=(350, 280),
                left_margin=2Plots.mm,
                bottom_margin=2Plots.mm,
                top_margin=1Plots.mm,
                right_margin=1Plots.mm
            )

            plot!(plt, [0.005; data["p_rand_edge"][2:end]], data["S1_rand_edge"],
                label="Random Edge (Avg)", marker=:circle, markersize=4, color=:gray, linewidth=2)

            plot!(plt, [0.005; data["p_rand_line"][2:end]], data["S1_rand_line"],
                label="Random Line (Avg)", marker=:cross, markersize=4, color=:black, linewidth=2, linestyle=:dash)

            plot!(plt, [0.005; data["p_largest"][2:end]], data["S1_largest"],
                label="Targeted: Largest", marker=:square, markersize=4, color=:blue, linewidth=2)

            plot!(plt, [0.005; data["p_degree"][2:end]], data["S1_degree"],
                label="Targeted: Degree", marker=:diamond, markersize=4, color=:orange, linewidth=2)

            plot!(plt, [0.005; data["p_betweenness"][2:end]], data["S1_betweenness"],
                label="Targeted: Betweenness", marker=:utriangle, markersize=4, color=:red, linewidth=2)

            plot!(plt, [0.005; data["p_shared_node"][2:end]], data["S1_shared_node"],
                label="Targeted: Shared Nodes", marker=:star5, markersize=5, color=:green, linewidth=2)

            outfile = joinpath(L_dir, "C_L$(L)_phi$(phi_tag).pdf")
            savefig(plt, outfile)
            println("Saved: ", outfile)
        end
    end
end

function random_edge_fragments_at_p(
    hyperedges,
    removed_set
)
    surviving_fragments = Vector{Vector{Int}}()

    for (i, he) in enumerate(hyperedges)
        current_frag = [he[1]]

        for j in 1:(length(he)-1)
            v = he[j+1]

            if (i, j) in removed_set
                if length(current_frag) > 1
                    push!(surviving_fragments, current_frag)
                end
                current_frag = [v]
            else
                push!(current_frag, v)
            end
        end

        if length(current_frag) > 1
            push!(surviving_fragments, current_frag)
        end
    end

    return surviving_fragments
end

function plot_all_attack_degree_evolution(
    L::Int,
    phi::Float64;
    p_values=[0.05, 0.10, 0.25, 0.50, 0.75, 0.95],
    save_dir=joinpath(@__DIR__, "Resiliencefig"),
    seed=1234
)
    Random.seed!(seed)

    L_dir = joinpath(save_dir, "L=$(L)")
    phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

    q_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag).jld2")
    data = JLD2.load(q_file)

    H = data["master_hyperedges"]
    total_edges = sum(length(he) - 1 for he in H)

    G_original = build_intersection_graph(H)

    idx_random_line = shuffle(1:length(H))
    idx_largest = sortperm([length(he) for he in H], rev=true)
    idx_degree = sortperm(degree(G_original), rev=true)

    bc_scores = Graphs.betweenness_centrality(G_original)
    idx_betweenness = sortperm(bc_scores, rev=true)

    all_segments = Tuple{Int,Int}[]
    for (i, he) in enumerate(H)
        for j in 1:(length(he)-1)
            push!(all_segments, (i, j))
        end
    end
    shuffled_segments = shuffle(all_segments)

    sorted_nodes, _ = sorted_lattice_nodes_by_hyperedge_occupancy(H)
    node_segments = node_to_internal_segments(H)

    strategies = [
        "random_edge",
        "random_line",
        "largest",
        "degree",
        "betweenness",
        "shared_node"
    ]

    line_orders = Dict(
        "random_line" => idx_random_line,
        "largest" => idx_largest,
        "degree" => idx_degree,
        "betweenness" => idx_betweenness
    )

    output_dir = joinpath(L_dir, "DegreeAfterAttack")
    mkpath(output_dir)

    all_degree_data = Dict{String,Any}()

    for strategy in strategies
        plt = plot(
            #label="Degree k",
            #label="P(K >= k)",
            yaxis=:log10,
            legend=false,
            grid=true,
            framestyle=:box,
            size=(350, 280)
        )

        colors = palette(:tab10, length(p_values))
        strategy_data = Dict{Float64,Any}()

        for (color_idx, p) in enumerate(p_values)
            damaged_H = Vector{Vector{Int}}()
            actual_p = 0.0

            if strategy == "random_edge"
                n_remove = min(round(Int, p * total_edges), total_edges)
                removed = Set(shuffled_segments[1:n_remove])

                damaged_H = random_edge_fragments_at_p(H, removed)
                actual_p = n_remove / total_edges

            elseif haskey(line_orders, strategy)
                order = line_orders[strategy]

                removed_flags = falses(length(H))
                removed_mass = 0
                order_idx = 1

                while removed_mass / total_edges < p &&
                    order_idx <= length(order)

                    he_idx = order[order_idx]
                    removed_flags[he_idx] = true
                    removed_mass += length(H[he_idx]) - 1
                    order_idx += 1
                end

                damaged_H = H[.!removed_flags]
                actual_p = removed_mass / total_edges

            elseif strategy == "shared_node"
                removed_nodes = Set{Int}()
                removed_segments = Set{Tuple{Int,Int}}()
                node_idx = 1

                while length(removed_segments) / total_edges < p &&
                    node_idx <= length(sorted_nodes)

                    node = sorted_nodes[node_idx]
                    push!(removed_nodes, node)

                    for segment in get(
                        node_segments,
                        node,
                        Tuple{Int,Int}[]
                    )
                        push!(removed_segments, segment)
                    end

                    node_idx += 1
                end

                damaged_H = fragments_after_node_attack(H, removed_nodes)
                actual_p = length(removed_segments) / total_edges
            end

            G_damaged = build_intersection_graph(damaged_H)
            degrees = degree(G_damaged)

            strategy_data[p] = (
                actual_p=actual_p,
                degrees=degrees,
                lengths=[length(he) - 1 for he in damaged_H]
            )

            if !isempty(degrees)
                k_vals, ccdf_vals = compute_ccdf(degrees)

                plot!(
                    plt,
                    k_vals,
                    ccdf_vals;
                    label="p=$(round(actual_p, digits=2))",
                    color=colors[color_idx],
                    marker=:circle,
                    markersize=2,
                    linewidth=1
                )
            end
        end

        all_degree_data[strategy] = strategy_data

        fig_file = joinpath(
            output_dir,
            "degree_CCDF_$(strategy)_L$(L)_phi$(phi_tag).pdf"
        )

        savefig(plt, fig_file)
        println("Saved: ", fig_file)
    end

    data_file = joinpath(
        output_dir,
        "degree_after_attacks_L$(L)_phi$(phi_tag).jld2"
    )

    JLD2.jldsave(
        data_file;
        L=L,
        phi=phi,
        p_values=p_values,
        seed=seed,
        degree_data=all_degree_data
    )

    println("Degree data saved to: ", data_file)

    return all_degree_data
end


function plot_random_edge_across_L(
    phi::Float64;
    L_values=[25, 50, 75, 100],
    metric=:Q,
    save_dir=joinpath(@__DIR__, "Resiliencefig")
)
    phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

    plt = plot(
        xaxis=:linear,
        grid=true,
        framestyle=:box,
        legend=:bottomleft,
        size=(350, 280)
    )

    colors = palette(:tab10, length(L_values))

    for (i, L) in enumerate(L_values)
        L_dir = joinpath(save_dir, "L=$(L)")

        q_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag).jld2")
        s1_file = joinpath(L_dir, "S1_data_L$(L)_phi$(phi_tag).jld2")


        file = if metric == :Q
            q_file
        elseif isfile(s1_file)
            s1_file
        else
            q_file
        end

        if !isfile(file)
            println("Missing: ", file)
            continue
        end

        data = JLD2.load(file)

        p = data["p_rand_edge"]
        y = metric == :Q ? data["Q_rand_edge"] : data["S1_rand_edge"]


        p_plot = copy(p)
        p_plot[1] = 0.005

        plot!(
            plt,
            p_plot,
            y;
            label="L=$L",
            color=colors[i],
            marker=:circle,
            markersize=3,
            linewidth=1.5
        )
    end

    metric_tag = metric == :Q ? "Q" : "C"

    outfile = joinpath(
        save_dir,
        "$(metric_tag)_random_edge_phi$(phi_tag)_across_L.pdf"
    )

    savefig(plt, outfile)
    display(plt)
    println("Saved: ", outfile)

    return plt
end



function scaled_x_mean_length(p_vals, line_lengths)
    mean_l = mean(line_lengths)
    return 1 .- (1 .- p_vals) .^ mean_l
end

function scaled_x_length_distribution(p_vals, line_lengths)
    x_vals = Float64[]

    for p in p_vals
        mean_survival = mean((1 - p) .^ line_lengths)
        push!(x_vals, 1 - mean_survival)
    end

    return x_vals
end

function plot_two_survival_scaled_Q(
    phi::Float64;
    L_values=[25, 50, 75, 100],
    save_dir=joinpath(@__DIR__, "Resiliencefig"),
    use_log_y::Bool=true
)
    phi_tag = replace(@sprintf("%.2f", phi), "." => "p")
    colors = palette(:tab10, length(L_values))

    yaxis_setting = use_log_y ? :log10 : :linear

    plt_b = plot(
        xlabel="1 - (1 - p)^<ℓ>",
        ylabel="Q(p)",
        title="Mean-length scaling",
        yaxis=yaxis_setting,
        grid=true,
        framestyle=:box,
        legend=:bottomleft
    )

    plt_c = plot(
        xlabel="1 - <(1 - p)^ℓ>",
        ylabel="Q(p)",
        title="Length-distribution scaling",
        yaxis=yaxis_setting,
        grid=true,
        framestyle=:box,
        legend=false
    )

    for (i, L) in enumerate(L_values)
        L_dir = joinpath(save_dir, "L=$(L)")
        q_file = joinpath(L_dir, "Q_data_L$(L)_phi$(phi_tag).jld2")

        if !isfile(q_file)
            println("Missing file: ", q_file)
            continue
        end

        data = JLD2.load(q_file)

        H = data["master_hyperedges"]
        line_lengths = [length(he) - 1 for he in H]

        p_vals = data["p_rand_edge"]
        Q_vals = data["Q_rand_edge"]

        x_mean = scaled_x_mean_length(p_vals, line_lengths)
        x_dist = scaled_x_length_distribution(p_vals, line_lengths)

        valid = Q_vals .> 0

        plot!(
            plt_b,
            x_mean[valid],
            Q_vals[valid];
            label="L=$L",
            color=colors[i],
            marker=:circle,
            markersize=3,
            linewidth=1.5
        )

        plot!(
            plt_c,
            x_dist[valid],
            Q_vals[valid];
            label="L=$L",
            color=colors[i],
            marker=:circle,
            markersize=3,
            linewidth=1.5
        )

        println(
            "L=$L, phi=$phi, <l>=",
            round(mean(line_lengths), digits=3),
            ", <l>/L=",
            round(mean(line_lengths) / L, digits=3),
            ", number of lines=",
            length(line_lengths)
        )
    end

    final_plot = plot(
        plt_b,
        plt_c;
        layout=(1, 2),
        size=(750, 300),
        margin=4Plots.mm
    )

    outfile = joinpath(
        save_dir,
        "Q_two_survival_scaled_phi$(phi_tag).pdf"
    )

    savefig(final_plot, outfile)
    display(final_plot)

    println("Saved: ", outfile)

    return final_plot
end

function plot_dynamic_mean_length_scaling(
    phi::Float64;
    L_values=[25, 50, 75, 100],
    save_dir=joinpath(@__DIR__, "Resiliencefig")
)
    phi_tag = replace(@sprintf("%.2f", phi), "." => "p")
    colors = palette(:tab10, length(L_values))

    plt_length = plot(
        xlabel="p",
        ylabel="<ℓ>(p)",
        title="Dynamic mean fragment length",
        grid=true,
        framestyle=:box,
        legend=:topright
    )

    plt_scaled = plot(
        xlabel="1 - (1 - p)^<ℓ(p)>",
        ylabel="Q(p)",
        title="Dynamic mean-length scaling",
        yaxis=:log10,
        grid=true,
        framestyle=:box,
        legend=false
    )

    for (i, L) in enumerate(L_values)
        L_dir = joinpath(save_dir, "L=$(L)")

        q_file = joinpath(
            L_dir,
            "Q_data_L$(L)_phi$(phi_tag).jld2"
        )

        lengths_file = joinpath(
            L_dir,
            "DegreeAfterAttack",
            "degree_after_attacks_L$(L)_phi$(phi_tag).jld2"
        )

        if !isfile(q_file) || !isfile(lengths_file)
            println("Missing data for L=$L")
            continue
        end

        q_data = JLD2.load(q_file)
        lengths_data = JLD2.load(lengths_file)

        degree_data = lengths_data["degree_data"]
        p_values = lengths_data["p_values"]

        actual_p = Float64[]
        mean_lengths = Float64[]

        for p in p_values
            result = degree_data["random_edge"][p]
            lengths = result.lengths

            if !isempty(lengths)
                push!(actual_p, result.actual_p)
                push!(mean_lengths, mean(lengths))
            end
        end

        # Find the nearest saved Q value for each actual p
        q_p = q_data["p_rand_edge"]
        q_values = q_data["Q_rand_edge"]

        matched_Q = [
            q_values[argmin(abs.(q_p .- p))]
            for p in actual_p
        ]

        x_dynamic = 1 .- (1 .- actual_p) .^ mean_lengths
        valid = matched_Q .> 0

        plot!(
            plt_length,
            actual_p,
            mean_lengths;
            label="L=$L",
            color=colors[i],
            marker=:circle,
            linewidth=1.5
        )

        plot!(
            plt_scaled,
            x_dynamic[valid],
            matched_Q[valid];
            label="L=$L",
            color=colors[i],
            marker=:circle,
            linewidth=1.5
        )
    end

    final_plot = plot(
        plt_length,
        plt_scaled;
        layout=(1, 2),
        size=(750, 300),
        margin=4Plots.mm
    )

    outfile = joinpath(
        save_dir,
        "Q_dynamic_mean_length_scaled_phi$(phi_tag).pdf"
    )

    savefig(final_plot, outfile)
    display(final_plot)

    return final_plot
end


function run_and_compare_shared_node_L25_minlen2(
    ;
    phi_values=[0.25, 0.50, 0.75, 0.95],
    old_dir=joinpath(@__DIR__, "Resiliencefig", "L=25"),
    new_dir=joinpath(@__DIR__, "Resiliencefig_shared_node_minlen2", "L=25"),
    comparison_dir=joinpath(@__DIR__, "SharedNode_minlen_comparison", "L=25"),
    ell_min::Int=2,
    p_floor=0.005
)
    L = 25
    mkpath(new_dir)
    mkpath(comparison_dir)

    colors = palette(:tab10, length(phi_values))

    new_Q_plot = plot(
        xlabel="p", ylabel="Q(p)",
        title="Shared-nodes attack, new minimum fragment length 2",
        xaxis=:log10, ylims=(0, 1.05),
        legend=:bottomleft, grid=true, framestyle=:box,
        size=(500, 400)
    )

    comparison_Q_plot = plot(
        xlabel="p", ylabel="Q(p)",
        title="Shared-nodes attack: old vs new minlen2",
        #xaxis=:log10, 
        ylims=(0, 1.05),
        xlims=(0, 1.05),
        legend=:bottomleft, grid=true, framestyle=:box,
        size=(500, 400)
    )

    for (i, phi) in enumerate(phi_values)
        phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

        old_file = joinpath(old_dir, "Q_data_L25_phi$(phi_tag).jld2")
        isfile(old_file) || error("Missing old file: $old_file")

        old_data = JLD2.load(old_file)

        master_hyperedges = old_data["master_hyperedges"]
        p_targets = Float64.(old_data["p_targets"])
        E_0 = old_data["E_0"]
        E_grid = old_data["E_grid"]
        total_edges = old_data["total_edges"]


        p_new, Q_new, S1_new = execute_intersection_node_attack(
            master_hyperedges,
            p_targets,
            L,
            E_0,
            E_grid,
            total_edges;
            ell_min
        )

        new_file = joinpath(
            new_dir,
            "shared_node_data_L25_phi$(phi_tag)_minlen2.jld2"
        )

        JLD2.jldsave(
            new_file;
            L=L,
            target_phi=phi,
            ell_min=ell_min,
            p_targets=p_targets,
            p_shared_node=p_new,
            Q_shared_node=Q_new,
            S1_shared_node=S1_new,
            master_hyperedges=master_hyperedges,
            E_0=E_0,
            E_grid=E_grid,
            total_edges=total_edges,
            p_definition="fraction of original edges absent from valid surviving fragments"
        )

        old_p = Float64.(old_data["p_shared_node"])
        old_Q = old_data["Q_shared_node"]
        new_p_plot = copy(p_new)
        old_p_plot = copy(old_p)

        new_p_plot[new_p_plot .== 0.0] .= p_floor
        old_p_plot[old_p_plot .== 0.0] .= p_floor

        phi_label = @sprintf("%.2f", phi)

        plot!(new_Q_plot, new_p_plot, Q_new;
            label="phi=$phi_label", color=colors[i],
            marker=:diamond, markersize=3, linewidth=1.8)

        plot!(comparison_Q_plot, old_p_plot, old_Q;
            label="phi=$phi_label, old", color=colors[i],
            linestyle=:dash, marker=:circle, markersize=3)

        plot!(comparison_Q_plot, new_p_plot, Q_new;
            label="phi=$phi_label, minlen2", color=colors[i],
            linestyle=:solid, marker=:diamond, markersize=3)
    end

    new_Q_file = joinpath(new_dir, "Q_shared_node_L25_all_phi_minlen2.pdf")
    comparison_file = joinpath(
        comparison_dir,
        "Q_shared_node_L25_old_vs_minlen2.pdf"
    )

    savefig(new_Q_plot, new_Q_file)
    savefig(comparison_Q_plot, comparison_file)

    display(new_Q_plot)
    display(comparison_Q_plot)

    println("New Shared-Node data and Q figure saved in: ", new_dir)
    println("Comparison Q figure saved to: ", comparison_file)

    return new_Q_plot, comparison_Q_plot
end

function compute_shared_node_minlen2(
    L::Int;
    phi_values=[0.25, 0.50, 0.75, 0.95],
    old_dir=joinpath(@__DIR__, "Resiliencefig", "L=$(L)"),
    new_dir=joinpath(@__DIR__, "Resiliencefig_shared_node_minlen2", "L=$(L)"),
    ell_min::Int=2
)
    mkpath(new_dir)

    for phi in phi_values
        phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

        old_file = joinpath(
            old_dir,
            "Q_data_L$(L)_phi$(phi_tag).jld2"
        )

        isfile(old_file) || error("Missing old file: $old_file")
        old_data = JLD2.load(old_file)

        master_hyperedges = old_data["master_hyperedges"]
        p_targets = Float64.(old_data["p_targets"])
        E_0 = old_data["E_0"]
        E_grid = old_data["E_grid"]
        total_edges = old_data["total_edges"]

        println("Computing Shared-Node: L=$L, phi=$phi")

        p_shared_node, Q_shared_node, S1_shared_node =
            execute_intersection_node_attack(
                master_hyperedges,
                p_targets,
                L,
                E_0,
                E_grid,
                total_edges;
                ell_min=ell_min
            )

        output_file = joinpath(
            new_dir,
            "shared_node_data_L$(L)_phi$(phi_tag)_minlen2.jld2"
        )

        JLD2.jldsave(
            output_file;
            L=L,
            target_phi=phi,
            ell_min=ell_min,
            p_shared_node=p_shared_node,
            Q_shared_node=Q_shared_node,
            S1_shared_node=S1_shared_node,
            master_hyperedges=master_hyperedges,
            E_0=E_0,
            E_grid=E_grid,
            total_edges=total_edges,
            p_definition="fraction of original edges absent from valid surviving fragments"
        )

        println("Saved: ", output_file)
    end
end

function compute_random_edge_minlen2(
    L::Int;
    phi_values=[0.25, 0.50, 0.75, 0.95],
    old_dir=joinpath(@__DIR__, "Resiliencefig", "L=$(L)"),
    new_dir=joinpath(@__DIR__, "Resiliencefig_random_edge_minlen2", "L=$(L)"),
    ell_min::Int=2,
    random_iterations::Int=10,
    seed::Int=1234
)
    mkpath(new_dir)

    for (phi_index, phi) in enumerate(phi_values)
        phi_tag = replace(@sprintf("%.2f", phi), "." => "p")

        old_file = joinpath(
            old_dir,
            "Q_data_L$(L)_phi$(phi_tag).jld2"
        )

        isfile(old_file) || error("Missing old file: $old_file")
        old_data = JLD2.load(old_file)

        master_hyperedges = old_data["master_hyperedges"]
        p_targets = Float64.(old_data["p_targets"])
        E_0 = old_data["E_0"]
        E_grid = old_data["E_grid"]
        total_edges = old_data["total_edges"]

        number_targets = length(p_targets)

        p_runs = zeros(random_iterations, number_targets)
        Q_runs = zeros(random_iterations, number_targets)
        C_runs = zeros(random_iterations, number_targets)

        seed_used = seed + 1000 * L + phi_index
        Random.seed!(seed_used)

        println("\nComputing Random Edge: L=$L, phi=$phi")

        for run in 1:random_iterations
            println("  realization $run / $random_iterations")

            p_run, Q_run, C_run = execute_random_edge_attack(
                master_hyperedges,
                p_targets,
                L,
                E_0,
                E_grid,
                total_edges;
                ell_min=ell_min
            )

            p_runs[run, :] .= p_run
            Q_runs[run, :] .= Q_run
            C_runs[run, :] .= C_run
        end

        p_rand_edge = vec(mean(p_runs, dims=1))
        Q_rand_edge = vec(mean(Q_runs, dims=1))
        C_rand_edge = vec(mean(C_runs, dims=1))

        output_file = joinpath(
            new_dir,
            "random_edge_data_L$(L)_phi$(phi_tag)_minlen2.jld2"
        )

        JLD2.jldsave(
            output_file;
            L=L,
            target_phi=phi,
            ell_min=ell_min,
            random_iterations=random_iterations,
            seed_used=seed_used,
            p_targets=p_targets,
            p_rand_edge=p_rand_edge,
            Q_rand_edge=Q_rand_edge,
            C_rand_edge=C_rand_edge,
            S1_rand_edge=C_rand_edge,
            p_runs=p_runs,
            Q_runs=Q_runs,
            C_runs=C_runs,
            master_hyperedges=master_hyperedges,
            E_0=E_0,
            E_grid=E_grid,
            total_edges=total_edges,
            p_definition="fraction of original edges absent from valid surviving fragments"
        )

        println("Saved: ", output_file)
    end
end


function random_edge_fragments_at_p_minlen2(
    hyperedges::Vector{Vector{Int}},
    removed_set::Set{Tuple{Int,Int}};
    ell_min::Int=2
)
    surviving_fragments = Vector{Vector{Int}}()

    for (i, he) in enumerate(hyperedges)
        current_frag = [he[1]]

        for j in 1:(length(he)-1)
            v = he[j+1]

            if (i, j) in removed_set
                if length(current_frag) - 1 >= ell_min
                    push!(surviving_fragments, current_frag)
                end
                current_frag = [v]
            else
                push!(current_frag, v)
            end
        end

        if length(current_frag) - 1 >= ell_min
            push!(surviving_fragments, current_frag)
        end
    end

    return surviving_fragments
end

