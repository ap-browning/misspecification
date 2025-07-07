using Distributions
using Plots, StatsPlots
using AdaptiveMCMC
using .Threads
using DataFrames, CSV

include("../library/library.jl")
include("defaults.jl")

## GENERATE DATA

    # True parameters (in the PDE)
    D = 2e3
    r = 1.0
    K = 5e-3
    u₀ = [0.1K,0.075K]
    σ = 1e-4

    p = [[r,K,u₀ⁱ,D,σ] for u₀ⁱ = u₀]

    # Observation times
    t = range(0.0,10.0,21)

    # Replicates
    n = [5,50,100,200]

    # Noise model (properly specified)
    ε = σ -> MvNormal(σ * ones(length(t)))

    # Initial conditions
    xmax = 2000.0
    ic = [  x -> (0.3 < x / xmax < 0.7) ? 0.0 : 1.0, 
            x -> (0.4 < x / xmax < 0.6) ? 0.0 : 1.0    ]

    u1 = solve_fkpp(p[1];tmax=maximum(t),ic=ic[1],xmax)
    u2 = solve_fkpp(p[2];tmax=maximum(t),ic=ic[2],xmax)

    data = [u.(t) .+ rand(ε(σ),n[end]) for u in [u1,u2]]

    # Gridded crowding function (initial guess)
    xxgrid = range(0.0,1.0,12)
    ffgrid = 1 .- xxgrid

## INFERENCE USING THE GP ODE

    # Likelihood
    function loglike(lp,data)
        pars = exp.(lp[1:4])
        fgrid = lp[5:end]
        u = solve_logistic(pars,[1.0;fgrid;0.0])
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Priors
    prior1 = Product(Uniform.(fill(-10.0,4),fill(2.0,4)))
    prior2 = f_gp_prior(length(ffgrid)-2)

    # Posterior
    logpost(lp,data) = insupport(prior1,lp[1:4]) && all(0.0 .≤ lp[5:end]) && lp[3] < lp[2] ? 
        loglike(lp,data) + 
        logpdf(prior1,lp[1:4]) + 
        logpdf(prior2,lp[5:end]) : -Inf

    # Initial condition 1
    res1 = Array{Any}(undef,length(n))
    lp = [log.([1.0,K,0.6 * 0.1 * K,σ]); ffgrid[2:end-1]]
    for i = eachindex(n)
        @time res1[i] = mcmc(lp, lp -> logpost(lp,data[1][:,1:n[i]]), 500000);
        lp = get_map(res1[i])
    end

    # Initial condition 2
    res2 = Array{Any}(undef,length(n))
    lp = [log.([1.0,xmax * K̄,0.6 * 0.1 * K̄ * xmax,σ]); ffgrid[2:end-1]]
    for i = eachindex(n)
        @time res2[i] = mcmc(lp, lp -> logpost(lp,data[2][:,1:n[i]]), 500000);
        lp = get_map(res2[i])
    end
    
## INFERENCE USING THE LOGISTIC MODEL

    # Likelihood
    function loglike(lp,data)
        pars = exp.(lp)
        u = solve_logistic(pars,[1.0;0.0])
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Priors
    prior1 = Product(Uniform.(fill(-10.0,4),fill(2.0,4)))

    # Posterior
    logpost(lp,data) = insupport(prior1,lp) ? 
        loglike(lp,data) + logpdf(prior1,lp)  : -Inf

    # Initial condition 1
    res3 = Array{Any}(undef,length(n))
    lp = log.([1.0,K,0.6 * 0.1 * K,σ])
    for i = eachindex(n)
        @time res3[i] = mcmc(lp, lp -> logpost(lp,data[1][:,1:n[i]]), 500000);
        lp = get_map(res3[i])
    end

    # Initial condition 2
    res4 = Array{Any}(undef,length(n))
    lp = log.([1.0,K,0.6 * 0.1 * K,σ])
    for i = eachindex(n)
        @time res4[i] = mcmc(lp, lp -> logpost(lp,data[2][:,1:n[i]]), 500000);
        lp = get_map(res4[i])
    end

## Table 1a

ic1_r_map = [exp(get_map(C)[1]) for C in res1]
ic1_r_025 = [quantile(exp.(C[:p1])[:],0.025) for C in res1]
ic1_r_975 = [quantile(exp.(C[:p1])[:],0.975) for C in res1]

ic2_r_map = [exp(get_map(C)[1]) for C in res2]
ic2_r_025 = [quantile(exp.(C[:p1])[:],0.025) for C in res2]
ic2_r_975 = [quantile(exp.(C[:p1])[:],0.975) for C in res2]

tab1a = DataFrame(
    ic = [fill(1,4); fill(2,4)],
    n = [n;n],
    MAP = [ic1_r_map; ic2_r_map],
    CI95_l = [ic1_r_025; ic2_r_025],
    CI95_u = [ic1_r_975; ic2_r_975]
)
CSV.write("$(@__DIR__)/tab1a.csv",tab1a)


## Table 1b

ic1_r_map = [exp(get_map(C)[1]) for C in res3]
ic1_r_025 = [quantile(exp.(C[:p1])[:],0.025) for C in res3]
ic1_r_975 = [quantile(exp.(C[:p1])[:],0.975) for C in res3]

ic2_r_map = [exp(get_map(C)[1]) for C in res4]
ic2_r_025 = [quantile(exp.(C[:p1])[:],0.025) for C in res4]
ic2_r_975 = [quantile(exp.(C[:p1])[:],0.975) for C in res4]

tab1b = DataFrame(
    ic = [fill(1,4); fill(2,4)],
    n = [n;n],
    MAP = [ic1_r_map; ic2_r_map],
    CI95_l = [ic1_r_025; ic2_r_025],
    CI95_u = [ic1_r_975; ic2_r_975]
)
CSV.write("$(@__DIR__)/tab1b.csv",tab1b)

## Figure S1

figS1 = plot(layout=grid(1,2),size=(600,220),palette=palette(:Blues_6,rev=true))

for i = eachindex(n)
    density!(figS1,subplot=1,exp.(res1[i][:p1]),label="n = $(n[i])",lw=2.0,frange=0.0,fα=0.3)
    density!(figS1,subplot=2,exp.(res2[i][:p1]),label="n = $(n[i])",lw=2.0,frange=0.0,fα=0.3)
end
vline!(figS1,subplot=1,[r],lw=2,ls=:dot,c=:black,label="")
vline!(figS1,subplot=2,[r],lw=2,ls=:dot,c=:black,label="")
plot!(figS1,ywiden=false,link=:all,ylabel="Posterior density",xlabel="r")
add_plot_labels!(figS1)

savefig(figS1,"$(@__DIR__)/figS1.svg")
figS1