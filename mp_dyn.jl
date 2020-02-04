include("./shared.jl")

function random_state_pop_dyn(mp::metapop)
    n_pops::Int64 = mp.n_pops

    x::Array{Float64} = zeros(n_pops)

    for p = 1:n_pops
        x[p] = rand(Uniform(0.0, mp.k[p]))
    end
    return(x)
end

function log_state(x, g::Int64)
    n_pops = length(x)

    open("./pop_dyn.csv", "a") do f
       for p in 1:n_pops
          x_p = x[p]
          write(f, "$p,$x_p,$g\n")
       end
   end
end

function compute_crosscorrelation_wrt_total_population(state::Array{Float64,2})
    n_pops = size(state)[1]
    n_timesteps = size(state)[2]
    cc_vector::Array{Float64} = zeros(n_pops)

    total_pop_size::Array{Float64} = zeros(n_timesteps)

    for t in 1:n_timesteps
        total_pop_size[t] = sum(state[:,t])
    end


    for p = 1:n_pops
        v = state[p,:]
        cc = crosscor(v,total_pop_size,[0])
        cc_vector[p] = cc[1]
    end

    return cc_vector
end

function compute_cc_matrix(state::Array{Float64,2})
    n_pops = size(state)[1]

    cc_matrix = zeros(n_pops, n_pops)

    for p1 = 1:n_pops
        for p2 = (p1+1):n_pops
            v1 = state[p1,:]
            v2 = state[p2,:]
            cc = crosscor(v1,v2,[0])
            cc_matrix[p1,p2] = cc[1]
        end
    end
    return(cc_matrix)
end

function run_dynamics(n_gen::Int64; n_pops::Int64 =5, k_total::Float64=100., migration_rate::Float64=0.5, alpha::Float64 = 3.0, lambda::Float64=1.5, sigma::Float64 = 1.0, kernel::Function = exp_kernel, dx::Function=stochastic_logistic, log_frequency::Int64 = 20)
    mp = get_random_metapop(n_pops, k_total)
    phi = get_dispersal_matrix(mp, migration_rate, alpha, kernel)

    x::Array{Float64} = random_state_pop_dyn(mp)
    open("./pop_dyn.csv", "w") do f
          write(f, "pop,count,gen\n")
    end

    state::Array{Float64, 2} = zeros(n_pops, n_gen)
    state[:,1] = random_state_pop_dyn(mp)
    for g = 2:n_gen
        x = dx(state[:,g-1], lambda, sigma, mp)
        state[:,g] = transpose(phi)*x
        if (g % log_frequency == 0)
            #log_state(x, g)
        end
    end

    #=
    cc_matrix = compute_cc_matrix(state)
    mean_cc::Float64 = 0.0
    ct = 0
    for p1 = 1:n_pops
        for p2 = (p1+1):n_pops
             mean_cc += cc_matrix[p1,p2]
             ct += 1
        end
    end
    mean_cc = mean_cc / ct =#

    cc_vector = compute_crosscorrelation_wrt_total_population(state)
    mean_cc::Float64 = sum(cc_vector) / n_pops
    return(mean_cc)
end

function stochastic_logistic(x, lambda, sigma::Float64, mp::metapop)
    n_pops::Int64 = mp.n_pops

    for p = 1:n_pops
        dx_p = x[p]*lambda*(mp.k[p]-x[p])
        x[p] = x[p] + dx_p*0.001 + rand(Normal(0, sigma))
        if (x[p] < 0)
            x[p] = 0
        end
    end
    return(x)
end

function migration_grad()
    migr_grad = vcat(collect(0.0001:0.0001:0.001), collect(0.001:0.001:0.01), collect(0.01:0.01:1))
    alpha_grad = [1.0, 3.0, 8.0, 15.0]
    sigma_grad = [0.1, 1.0, 5.0, 10.0]
    n_reps = 50

    df = DataFrame(rep=[], mig=[], mean_cc=[], alpha=[], sigma=[])

    for sigma in sigma_grad
        for alpha in alpha_grad
            for m in migr_grad
                for r = 1:n_reps
                    mean_cc = run_dynamics(1000, migration_rate=m, alpha=alpha, sigma=sigma)
                    push!(df.rep, r)
                    push!(df.mig, m)
                    push!(df.mean_cc, mean_cc)
                    push!(df.alpha, alpha)
                    push!(df.sigma, sigma)
                end
            end
        end
    end

    CSV.write("migration_grad.csv", df)
end

function alpha_grad()
    migr_grad = [0.0001, 0.001, 0.01]
    alpha_grad = collect(1.0:0.1:20.0)
    sigma_grad = [0.0, 0.001, 0.1]
    n_reps = 50

    df = DataFrame(rep=[], mig=[], mean_cc=[], alpha=[], sigma=[])

    for sigma in sigma_grad
        for alpha in alpha_grad
            for m in migr_grad
                for r = 1:n_reps
                    mean_cc = run_dynamics(1000, migration_rate=m, alpha=alpha, sigma=sigma)
                    push!(df.rep, r)
                    push!(df.mig, m)
                    push!(df.mean_cc, mean_cc)
                    push!(df.alpha, alpha)
                    push!(df.sigma, sigma)
                end
            end
        end
    end
    CSV.write("alpha_grad.csv", df)
end

function sigma_grad()
    migr_grad = [0.0001, 0.001, 0.01]
    alpha_grad = [1.0, 3.0, 8.0, 15.0]
    sigma_grad = collect(0.1:0.1:10.0)
    n_reps = 50

    df = DataFrame(rep=[], mig=[], mean_cc=[], alpha=[], sigma=[])

    for sigma in sigma_grad
        for alpha in alpha_grad
            for m in migr_grad
                for r = 1:n_reps
                    mean_cc = run_dynamics(1000, migration_rate=m, alpha=alpha, sigma=sigma)
                    push!(df.rep, r)
                    push!(df.mig, m)
                    push!(df.mean_cc, mean_cc)
                    push!(df.alpha, alpha)
                    push!(df.sigma, sigma)
                end
            end
        end
    end
    CSV.write("sigma_grad.csv", df)
end

function run_k_grad()
    migr_grad = vcat(collect(0.001:0.001:0.01), collect(0.01:0.01:1))
    alpha_grad = [1.0, 3.0, 8.0, 15.0]
    sigma_grad = [0.01, 0.05, 0.1, 0.25]
    k_grad = [250, 500, 1000, 2000]
    n_reps = 50
    df = DataFrame(rep=[], mig=[], k=[], mean_cc=[], alpha=[], sigma=[])

    @showprogress   for m in migr_grad
        for k in k_grad
            for sigma_p in sigma_grad
                for alpha in alpha_grad
                    for r = 1:n_reps
                        sigma = sigma_p*k
                        mean_cc = run_dynamics(k, migration_rate=m, alpha=alpha, sigma=sigma)
                        push!(df.rep, r)
                        push!(df.mig, m)
                        push!(df.mean_cc, mean_cc)
                        push!(df.alpha, alpha)
                        push!(df.sigma, sigma_p)
                        push!(df.k, k)
                    end
                end
            end
        end
    end
    CSV.write("k_grad.csv", df)
end
#run_k_grad()

## possibly rescale alpha for each kernel such that mean dist is same? idk
function different_dispersal_kernels()
    migr_grad = vcat(collect(0.001:0.001:0.01), collect(0.01:0.01:1))
    alpha_grad = [1.0, 3.0, 8.0, 15.0]
    sigma_grad = [1.0, 5.0, 10.0]
    n_reps = 50

    kernels = [exp_kernel, gauss_kernel, uniform_kernel ]

    df = DataFrame(rep=[], kernel=[], mig=[], mean_cc=[], alpha=[], sigma=[])

    @showprogress for sigma in sigma_grad
        for k in kernels
            for alpha in alpha_grad
                for m in migr_grad
                    for r = 1:n_reps
                        mean_cc = run_dynamics(1000, kernel=k,migration_rate=m, alpha=alpha, sigma=sigma)
                        push!(df.rep, r)
                        push!(df.mig, m)
                        push!(df.mean_cc, mean_cc)
                        push!(df.alpha, alpha)
                        push!(df.sigma, sigma)
                        push!(df.kernel, k)
                    end
                end
            end
        end
    end
    CSV.write("diff_diskerns.csv", df)
end
different_dispersal_kernels()
