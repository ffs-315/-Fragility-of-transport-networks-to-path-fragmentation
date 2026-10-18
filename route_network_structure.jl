using Graphs, StatsBase, Plots, Random, NetworkLayout, JLD2

# ---------- Pathfinding ----------
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

# ---------- Network generation ----------
function generate_network_to_fixed_phi(L::Int, target_phi::Float64)
    L_nodes = L + 1
    L_R = Graphs.grid([L_nodes, L_nodes])
    routes = Vector{Vector{Int}}()
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
        push!(routes, path_nodes)
        total_length += length(path_edges)
    end
    return routes
end

function build_intersection_graph(routes::Vector{Vector{Int}})
    N = length(routes)
    G_line = SimpleGraph(N)
    station_sets = [Set(he) for he in routes]
    for i in 1:N
        for j in (i+1):N
            if !isdisjoint(station_sets[i], station_sets[j])
                add_edge!(G_line, i, j)
            end
        end
    end
    return G_line
end

# ---------- Lattice and route-intersection visualization ----------
function lattice_node_xy(id::Int, L::Int)
    L_nodes = L + 1
    x = (id - 1) % L_nodes
    y = (id - 1) ÷ L_nodes
    return x, y
end

function plot_figure1_lattice_and_projection(
    L::Int, phi::Float64; seed=nothing,
    outfile::String="results/figure1_lattice_projection.png"
)
    if seed !== nothing
        Random.seed!(seed)
    end
    mkpath(dirname(outfile))
    H = generate_network_to_fixed_phi(L, phi)
    G = build_intersection_graph(H)
    colors = palette(:tab20, max(length(H), 1))
    node_counts = Dict{Int,Int}()
    for he in H
        for v in unique(he)
            node_counts[v] = get(node_counts, v, 0) + 1
        end
    end
    shared_nodes = [v for (v, c) in node_counts if c >= 2]
    p_lattice = plot(title="Lattice Network", aspect_ratio=:equal,
        legend=false, grid=false, xlabel="", ylabel="",
        xlims=(-0.5, L + 0.5), ylims=(-0.5, L + 0.5), size=(800, 800))
    for x in 0:L
        plot!(p_lattice, [x, x], [0, L], color=:gray90, linewidth=0.5)
    end
    for y in 0:L
        plot!(p_lattice, [0, L], [y, y], color=:gray90, linewidth=0.5)
    end
    all_x = Float64[]
    all_y = Float64[]
    for v in 1:(L+1)^2
        x, y = lattice_node_xy(v, L)
        push!(all_x, x)
        push!(all_y, y)
    end
    scatter!(p_lattice, all_x, all_y, markercolor=:white,
        markerstrokecolor=:black, markerstrokewidth=0.8, markersize=2.2)
    for (i, he) in enumerate(H)
        xs = [lattice_node_xy(v, L)[1] for v in he]
        ys = [lattice_node_xy(v, L)[2] for v in he]
        plot!(p_lattice, xs, ys, color=colors[mod1(i, length(colors))],
            linewidth=1.2, alpha=0.6)
        scatter!(p_lattice, xs, ys, markercolor=colors[mod1(i, length(colors))],
            markerstrokecolor=:black, markerstrokewidth=0.2, markersize=1.2)
    end
    if !isempty(shared_nodes)
        sx = [lattice_node_xy(v, L)[1] for v in shared_nodes]
        sy = [lattice_node_xy(v, L)[2] for v in shared_nodes]
        scatter!(p_lattice, sx, sy, markercolor=:yellow,
            markerstrokecolor=:black, markerstrokewidth=0.2, markersize=1.2)
    end
    positions = NetworkLayout.spring(G)
    cx = [position[1] for position in positions]
    cy = [position[2] for position in positions]
    p_projection = plot(title="Abstract network", legend=false, grid=true,
        xlabel="", ylabel="", size=(800, 800))
    for e in edges(G)
        u = src(e)
        v = dst(e)
        plot!(p_projection, [cx[u], cx[v]], [cy[u], cy[v]],
            color=:gray65, linewidth=1.2, alpha=0.75)
    end
    node_sizes = [2.5 + 1.5 * log1p(length(he) - 1) for he in H]
    for i in eachindex(H)
        scatter!(p_projection, [cx[i]], [cy[i]], markercolor=colors[i],
            markerstrokecolor=:white, markerstrokewidth=0.25, markersize=node_sizes[i])
    end
    final_plot = plot(p_lattice, p_projection, layout=(1, 2),
        size=(1800, 800), margin=5Plots.mm)
    savefig(final_plot, outfile)
    display(final_plot)
    actual_phi = sum(length.(H) .- 1) / (2 * L * (L + 1))
    println("Figure saved to: ", outfile)
    println("L = ", L)
    println("target phi = ", phi)
    println("actual phi = ", round(actual_phi, digits=4))
    println("number of routes = ", length(H))
    return final_plot, H
end

# ---------- Deposition dynamics ----------
function track_deposition_dynamics(L::Int, phi_targets::AbstractVector; max_len_mode::Symbol=:L)
    L_nodes = L + 1
    max_len = max_len_mode == :L ? L :
              max_len_mode == :L2 ? L^2 :
              error("max_len_mode must be :L or :L2")
    L_R = Graphs.grid([L_nodes, L_nodes])
    routes = Vector{Vector{Int}}()
    total_length = 0
    denominator = 2 * L * (L + 1)
    fails = 0
    sorted_targets = sort(collect(phi_targets))
    target_idx = 1
    birth_phis = Float64[]
    lengths = Int[]
    degrees_at_phi = Dict{Float64,Vector{Int}}()
    tortuosity_ratios = Float64[]
    while target_idx <= length(sorted_targets)
        target_phi = sorted_targets[target_idx]
        current_phi = total_length / denominator
        if current_phi >= target_phi
            G_line = build_intersection_graph(routes)
            degrees_at_phi[target_phi] = degree(G_line)
            target_idx += 1
            continue
        end
        A, B = rand(1:nv(L_R)), rand(1:nv(L_R))
        if A == B
            continue
        end
        path_nodes, path_edges = constrained_closest_path(L_R, A, B, 2, max_len, L_nodes)
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
        actual_len = length(path_edges)
        actual_target = path_nodes[end]
        ideal_len = manhattan_dist(A, actual_target, L_nodes)
        if ideal_len > 0
            push!(tortuosity_ratios, actual_len / ideal_len)
        end
        push!(routes, path_nodes)
        push!(lengths, actual_len)
        push!(birth_phis, current_phi)
        total_length += actual_len
    end
    return birth_phis, lengths, degrees_at_phi, tortuosity_ratios
end

# ---------- Length and degree PDFs ----------
function compute_binned_pdf(data::AbstractVector{Int}; nbins::Int=30)
    if isempty(data)
        return Float64[], Float64[]
    end
    h = fit(Histogram, data, nbins=nbins)
    edges = h.edges[1]
    midpoints = (edges[1:end-1] .+ edges[2:end]) ./ 2.0
    bin_widths = step(edges)
    total_area = sum(h.weights) * bin_widths
    pdf_vals = h.weights ./ total_area
    valid_idx = pdf_vals .> 0
    return midpoints[valid_idx], pdf_vals[valid_idx]
end

function plot_ensemble_pdfs(L::Int, realizations::Int)
    phi_targets = [0.25, 0.50, 0.75, 0.95]
    agg_lengths = Dict{Float64,Vector{Int}}()
    agg_degrees = Dict{Float64,Vector{Int}}()
    for phi in phi_targets
        agg_lengths[phi] = Int[]
        agg_degrees[phi] = Int[]
    end
    println("Running $realizations realizations for L = $L...")
    for r in 1:realizations
        if r % 10 == 0
            println("Completed $r / $realizations realizations...")
        end
        _, lengths, degrees_at_phi, _ = track_deposition_dynamics(L, phi_targets)
        for phi in phi_targets
            if haskey(degrees_at_phi, phi)
                current_degrees = degrees_at_phi[phi]
                n_lines = length(current_degrees)
                if n_lines <= length(lengths)
                    append!(agg_lengths[phi], lengths[1:n_lines])
                    append!(agg_degrees[phi], current_degrees)
                end
            end
        end
    end
    p_len = plot(title="Ensemble Length PDF (N=$realizations)",
        xlabel="Length (l)", ylabel="P(l)", yaxis=:log10, grid=true, legend=:topright)
    p_deg = plot(title="Ensemble Degree PDF (N=$realizations)",
        xlabel="Degree (k)", ylabel="P(k)", yaxis=:log10, grid=true, legend=:topright)
    colors = palette(:tab10)
    global_min_y = Inf
    global_max_y = -Inf
    for (i, phi) in enumerate(phi_targets)
        if isempty(agg_degrees[phi])
            continue
        end
        len_vals, len_pdf = compute_binned_pdf(agg_lengths[phi], nbins=40)
        deg_vals, deg_pdf = compute_binned_pdf(agg_degrees[phi], nbins=40)
        if !isempty(len_pdf)
            global_min_y = min(global_min_y, minimum(len_pdf))
            global_max_y = max(global_max_y, maximum(len_pdf))
        end
        if !isempty(deg_pdf)
            global_min_y = min(global_min_y, minimum(deg_pdf))
            global_max_y = max(global_max_y, maximum(deg_pdf))
        end
        plot!(p_len, len_vals, len_pdf, label="phi = $phi",
            color=colors[i], marker=:circle, markersize=3, linewidth=1.5)
        plot!(p_deg, deg_vals, deg_pdf, label="phi = $phi",
            color=colors[i], marker=:square, markersize=3, linewidth=1.5)
    end
    if global_min_y < Inf && global_max_y > -Inf
        ylims!(p_len, (global_min_y * 0.5, global_max_y * 2.0))
        ylims!(p_deg, (global_min_y * 0.5, global_max_y * 2.0))
    end
    final_plot = plot(p_len, p_deg, layout=(1, 2), size=(900, 400), margin=5Plots.mm)
    display(final_plot)
    savefig(final_plot, "ensemble_pdfs_L$(L).pdf")
    return final_plot, agg_lengths, agg_degrees
end

# ---------- Length-degree correlations ----------
function plot_age_degree_scatter(L::Int)
    phi_targets = [0.25, 0.50, 0.75, 0.95]
    birth_phis, lengths, degrees_at_phi = track_deposition_dynamics(L, phi_targets)
    plots = []
    clims = (0.0, maximum(phi_targets))
    for phi in phi_targets
        if !haskey(degrees_at_phi, phi)
            continue
        end
        current_degrees = degrees_at_phi[phi]
        n_lines = length(current_degrees)
        current_lengths = lengths[1:n_lines]
        current_births = birth_phis[1:n_lines]
        p = scatter(current_lengths, current_degrees, zcolor=current_births,
            clims=clims, cmap=:viridis, markerstrokewidth=0, markersize=6,
            alpha=0.7, title="Phi = $phi", xlabel="Line Length",
            ylabel="Degree (Intersections)", legend=false,
            ylims=(0, 100), xlims=(0, 100), grid=true, aspect_ratio=1)
        push!(plots, p)
    end
    layout_dim = max(1, ceil(Int, length(plots) / 2))
    h1 = plot(plots..., layout=(layout_dim, 2),
        size=(900, 450 * layout_dim), margin=5Plots.mm)
    display(h1)
    savefig(h1, "age_degree_scatter_L$(L).pdf")
    return h1
end

# ---------- Interpolation and percentile bands ----------
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
    Y = reduce(hcat, Y_vecs)'
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
    plot!(p, x, p975, fillrange=p025, color=color, alpha=0.2, linewidth=0, label="")
    plot!(p, x, p75, fillrange=p25, color=color, alpha=0.4, linewidth=0, label="")
    plot!(p, x, mean_Y, color=color, linewidth=2, label="")
end

# ---------- Moving-average length and tortuosity ----------
function plot_moving_average_scaling_bands(
    L_values::AbstractVector{Int}=[25, 100, 250], realizations::Int=10,
    phi_target::Float64=0.95; max_len_mode::Symbol=:L,
    save_file::String="moving_average_data_bands.jld2"
)
    plots_len = []
    plots_tort = []
    all_data = Dict{Int,Vector{Dict{String,Any}}}()
    phi_grid = collect(range(0.01, stop=phi_target, length=200))
    for L in L_values
        println("Running simulations for L = $L...")
        all_data[L] = Vector{Dict{String,Any}}()
        p_len = plot(title="L = $L", xlabel="Lattice coverage",
            ylabel="relative path", legend=false, grid=true, ylim=(0.0, 1.0))
        p_tort = plot(title="L = $L", xlabel="Lattice coverage",
            ylabel="Tortuosity Ratio (Actual / Manhattan)",
            legend=false, grid=true, ylim=(0.0, 2.5))
        lengths_interp = Vector{Vector{Float64}}()
        tort_interp = Vector{Vector{Float64}}()
        for r in 1:realizations
            birth_phis, lengths, degrees_dict, tortuosity_ratios =
                track_deposition_dynamics(L, [phi_target]; max_len_mode=max_len_mode)
            ma_tortuosity = Float64[]
            ma_lengths = Float64[]
            ma_phis = Float64[]
            if !isempty(tortuosity_ratios)
                w = min(max(1, length(tortuosity_ratios) ÷ 20), 100)
                if length(tortuosity_ratios) >= w
                    ma_lengths = Float64[
                        sum(lengths[i:(i+w-1)]) / (w * L)
                        for i in 1:(length(lengths)-w+1)]
                    ma_tortuosity = Float64[
                        sum(tortuosity_ratios[i:(i+w-1)]) / w
                        for i in 1:(length(tortuosity_ratios)-w+1)]
                    ma_phis = Float64[
                        sum(birth_phis[i:(i+w-1)]) / w
                        for i in 1:(length(birth_phis)-w+1)]
                    if length(ma_phis) > 1
                        push!(lengths_interp, interp1d(ma_phis, ma_lengths, phi_grid))
                        push!(tort_interp, interp1d(ma_phis, ma_tortuosity, phi_grid))
                    end
                end
            end
            push!(all_data[L], Dict(
                "birth_phis" => birth_phis, "lengths" => lengths,
                "degrees" => get(degrees_dict, phi_target, Int[]),
                "ma_phis" => ma_phis, "ma_lengths" => ma_lengths,
                "ma_tortuosity" => ma_tortuosity))
        end
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
    final_plot = plot(all_plots..., layout=(2, length(L_values)),
        size=(400 * length(L_values), 800), margin=5Plots.mm)
    display(final_plot)
    pdf_filename = splitext(save_file)[1] * ".pdf"
    savefig(final_plot, pdf_filename)
    return final_plot, all_data
end