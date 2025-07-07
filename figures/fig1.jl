using Distributions
using Plots, StatsPlots
using .Threads

include("../library/models/ode_logistic.jl")
include("../library/gp/gp_crowding.jl")
include("../library/stats.jl")
include("defaults.jl")

## GENERATE DATA

    # True parameters
    r  = 1.0
    K  = 5e-3
    u₀ = K .* (0.1:0.1:0.5)
    σ  = 1e-4

    pars = [[r,K,u₀ⁱ,σ] for u₀ⁱ in u₀]

    # Observation times
    t = range(0.0,5.0,21)

    # Replicates
    n = 5

    # Noise model (properly specified)
    ε = σ -> MvNormal(σ * ones(length(t)))

    # True crowding function
    f = x -> 1 - x^2

    # Model solution
    u = solve_logistic.(pars,f);

    # Generate data
    data = [uⁱ.(t) .+ rand(ε(σ),n) for uⁱ in u]


## INFERENCE

    # True parameters
    lp = [log.(p) for p in pars]

    # Likelihood function (misspecified, assume logistic)
    function loglike_logistic(lp,data)
        pars  = exp.(lp[1:4])
        u = solve_logistic(pars,[1.0, 0.0])
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Prior
    prior = Product(Uniform.(fill(-10.0,4),fill(2.0,4)))

    # Posterior (logistic)
    logpost_logistic(lp,data) = insupport(prior,lp) && (lp[3] < lp[2]) ? 
        loglike_logistic(lp,data) + logpdf(prior,lp) : -Inf

    # Inference
    res = Array{Any}(undef,length(u₀))

    @time @threads for i = 1:5
        res[i] = mcmc(lp[1], lp -> logpost_logistic(lp,data[i]), 500000;
            names=[:lr,:lK,:lu₀,:lσ])
    end


## Plot
 
    fig1a = plot(palette=palette(:Blues_6,rev=true))
    fig1b = plot(palette=palette(:Blues_6,rev=true))

    # Plot posteriors and data
    for i = eachindex(u₀)
        # (a) - posteriors
        density!(fig1a,exp.(res[i][:lr]),frange=0.0,lw=0.0,label="u₀ = $(u₀[i])")

        # (b) - data and best fit
        scatter!(fig1b,t,data[i],c=:black,α=0.2,msw=0.0,label="",
            shape=[:circle,:dtriangle,:diamond,:star5,:rect][i])
    end

    # Plot prediction at the MAP
    for i = eachindex(u₀)
        sol = solve_logistic(exp.(get_map(res[i])),[1.0,0.0])
        plot!(fig1b,sol,c=palette(:Blues_6,rev=true)[i],lw=2.0,label="u₀ = $(u₀[i])")
    end

    plot!(fig1a,ywiden=false,xlabel="r",ylabel="posterior density")
    plot!(fig1b,xlabel="t [d]",ylabel="cell density")

    fig1 = plot(fig1a,fig1b,size=(600,220))
    savefig(fig1,"$(@__DIR__)/fig1.svg")
    fig1