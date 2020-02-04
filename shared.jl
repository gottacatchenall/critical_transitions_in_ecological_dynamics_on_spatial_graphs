using StatsBase
using Distributions
using DataFrames
using CSV
using ProgressMeter


# =================================================
#    shared
#    types
#
# =================================================

mutable struct metapop
    n_pops::Int64
    k::Array{Float64}
    x::Array{Float64}
    y::Array{Float64}
end

function exp_kernel(alpha::Float64, d_ij::Float64)
    return(exp(-1*alpha*d_ij))
end

function gauss_kernel(alpha::Float64, d_ij::Float64)
    denom::Float64 = (alpha)
    return(exp(-1*denom*d_ij^2))
end

function uniform_kernel(alpha::Float64, d_ij::Float64)
    return(1.0)
end

function get_distance_between_pops(mp::metapop, p1::Int64, p2::Int64)
    d_ij::Float64 = sqrt((mp.x[p1] - mp.x[p2])^2 + (mp.y[p1] - mp.y[p2])^2)
    return(d_ij)
end

# =================================================
#    generators
#
#
# =================================================

function get_dispersal_matrix(mp::metapop, migration_rate::Float64, alpha::Float64, kernel::Function)
    n_pops::Int64 = mp.n_pops

    # get a probability distribution
    dispersal_distribution = zeros(n_pops, n_pops)

    for p1 = 1:n_pops
        row_sum::Float64 = 0.0
        for p2 = 1:n_pops
            if (p1 != p2)
                d_ij::Float64 = get_distance_between_pops(mp, p1,p2)
                dispersal_distribution[p1,p2] = kernel(alpha, d_ij)
                row_sum += dispersal_distribution[p1,p2]
            end
        end

        for p2 = 1:n_pops
            if (p1 != p2)
                dispersal_distribution[p1,p2] /= row_sum
            end
        end
    end

    dispersal_matrix = zeros(n_pops, n_pops)
    for p1 = 1:n_pops
        for p2 = 1:n_pops
            if (p1 != p2)
                dispersal_matrix[p1,p2] = dispersal_distribution[p1,p2] * migration_rate
            else
                dispersal_matrix[p1,p2] = 1.0 - migration_rate
            end
        end
    end

    return(dispersal_matrix)
end

function get_random_metapop(n_pops::Int64, k_total::Float64)
    x::Array{Float64} = zeros(n_pops)
    y::Array{Float64} = zeros(n_pops)
    k::Array{Float64} = zeros(n_pops)

    k_subdivided::Float64 = k_total / n_pops

    for p = 1:n_pops
        x[p] = rand(Uniform())
        y[p] = rand(Uniform())
        k[p] = k_subdivided
    end

    return(metapop(n_pops, k, x, y))
end
