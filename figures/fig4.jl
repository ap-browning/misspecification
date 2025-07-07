using Distributions
using Plots, StatsPlots
using AdaptiveMCMC
using JLD2

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
    n = 5

    # Noise model (properly specified)
    ε = σ -> MvNormal(σ * ones(length(t)))

    # Initial conditions
    xmax = 2000.0
    ic = [  x -> (0.3 < x / xmax < 0.7) ? 0.0 : 1.0, 
            x -> (0.4 < x / xmax < 0.6) ? 0.0 : 1.0    ]

    u1 = solve_fkpp(p[1];tmax=maximum(t),ic=ic[1],xmax)
    u2 = solve_fkpp(p[2];tmax=maximum(t),ic=ic[2],xmax)

    u1full = solve_fkpp(p[1];tmax=maximum(t),ic=ic[1],xmax,ret=:all)
    u2full = solve_fkpp(p[2];tmax=maximum(t),ic=ic[2],xmax,ret=:all)

    data = [u.(t) .+ rand(ε(σ),n) for u in [u1,u2]]
    #@save "$(@__DIR__)/fig4_data.jld2" data
    @load "$(@__DIR__)/fig4_data.jld2" data

    # Gridded crowding function (initial guess)
    xxgrid = range(0.0,1.0,12)
    ffgrid = 1 .- xxgrid

## INFERENCE USING THE PDE...

    # Likelihood
    function loglike_pde(lp,data,ic)
        pars = exp.(lp)
        u = solve_fkpp(pars;tmax=maximum(t),ic,xmax)
        loglikelihood(ε(pars[end]),data .- u.(t))
    end

    # Prior
    prior = Product(Uniform.(fill(-10.0,5),fill(10.0,5)))

    logpost_pde(lp,data,ic) = insupport(prior,lp) ? loglike_pde(lp,data,ic) + logpdf(prior,lp) : -Inf

    #@time res01 = mcmc(log.(p[1]), lp -> logpost_pde(lp,data[1],ic[1]), 50000);
    #@time res02 = mcmc(log.(p[2]), lp -> logpost_pde(lp,data[2],ic[2]), 50000);
    #@save "$(@__DIR__)/fig4_res.jld2" res01 res02
    @load "$(@__DIR__)/fig4_res.jld2" res01 res02

## INFERENCE USING THE GP ODE

    # Parameter guess
    lp = [log.([1.0,K,0.6 * 0.1 * K,σ]); ffgrid[2:end-1]]

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

    # Inference (n = 5)
    @time res1 = mcmc(lp, lp -> logpost(lp,data[1][:,1:5]), 500000);
    @time res2 = mcmc(lp, lp -> logpost(lp,data[2][:,1:5]), 500000);

## INFERENCE USING THE (MISSPECIFIED) LOGISTIC ODE

    # Parameter guess
    lp = log.([1.0,K,0.6 * 0.1 * K,σ])

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
        loglike(lp,data) + 
        logpdf(prior1,lp) : -Inf

    # Inference (n = 5)
    @time res3 = mcmc(lp, lp -> logpost(lp,data[1][:,1:5]), 500000);
    @time res4 = mcmc(lp, lp -> logpost(lp,data[2][:,1:5]), 500000);

## Plot fits (validation)

 fig4 = plot(layout=grid(2,4))

    # (a,e) Show PDE solution (at true value)
    tv = range(0.0,10,6)
    plot!(fig4,subplot=1,[x -> u1full(x,t) for t in tv],xlim=(0.0,xmax),palette=palette(:Purples,rev=true),
        label="",xlabel="x",ylabel="u(x,t)")
    plot!(fig4,subplot=5,[x -> u2full(x,t) for t in tv],xlim=(0.0,xmax),palette=palette(:Purples,rev=true),
        label="",xlabel="x",ylabel="u(x,t)")

    # (b,f) Show data (integrated PDE solution)
    scatter!(fig4,subplot=2,t,data[1][:,1:5],c=:black,msw=0.0,α=0.2,label="",ylim=(0.0,5.5e-3),xlabel="t [d]",ylabel="Overall cell density")
    scatter!(fig4,subplot=6,t,data[2][:,1:5],c=:black,msw=0.0,α=0.2,label="",ylim=(0.0,5.5e-3),xlabel="t [d]",ylabel="Overall cell density")
    lp1 = get_map(res1); pars1 = exp.(lp1[1:4]); fgrid1 = lp1[5:end]
    lp2 = get_map(res2); pars2 = exp.(lp2[1:4]); fgrid2 = lp2[5:end]
    lp3 = get_map(res3); pars3 = exp.(lp3);
    lp4 = get_map(res4); pars4 = exp.(lp4);
    ufit1 = solve_logistic(pars1,[1.0;fgrid1;0.0])
    ufit2 = solve_logistic(pars2,[1.0;fgrid2;0.0])
    ufit3 = solve_logistic(pars3,[1.0;0.0])
    ufit4 = solve_logistic(pars4,[1.0;0.0])
    plot!(fig4,subplot=2,t -> ufit3(t),xlim=(0.0,10.0),c=:blue,lw=2.0,label="logistic (n = 5)")
    plot!(fig4,subplot=6,t -> ufit4(t),xlim=(0.0,10.0),c=:blue,lw=2.0,label="")
    plot!(fig4,subplot=2,t -> ufit1(t),xlim=(0.0,10.0),c=:red,lw=2.0,label="gp (n = 5)")
    plot!(fig4,subplot=6,t -> ufit2(t),xlim=(0.0,10.0),c=:red,lw=2.0,label="")

    # (c,g) Posteriors for r
    density!(fig4,subplot=3,exp.(res1[:p1]),c=:red,frange=0.0,fα=0.3,label="ode (n = 5)",xlabel="r",ylabel="Posterior density")
    density!(fig4,subplot=7,exp.(res2[:p1]),c=:red,frange=0.0,fα=0.3,label="",xlabel="r",ylabel="Posterior density")
    density!(fig4,subplot=3,exp.(res3[:p1]),c=:blue,frange=0.0,fα=0.3,label="logistic (n = 5)",xlabel="r",ylabel="Posterior density")
    density!(fig4,subplot=7,exp.(res4[:p1]),c=:blue,frange=0.0,fα=0.3,label="",xlabel="r",ylabel="Posterior density")
    density!(fig4,subplot=3,exp.(res01[:p1]),c=:purple,frange=0.0,fα=0.3,label="pde (n = 5)",xlabel="r",ylabel="Posterior density")
    density!(fig4,subplot=7,exp.(res02[:p1]),c=:purple,frange=0.0,fα=0.3,label="",xlabel="r",ylabel="Posterior density")
    vline!(fig4,subplot=3,[r],c=:black,lw=2.0,ls=:dot,label="",ywiden=false,xlim=(0.61,1.31))
    vline!(fig4,subplot=7,[r],c=:black,lw=2.0,ls=:dot,label="",ywiden=false,xlim=(0.61,1.31))

    # (e,h) Inferred crowding function
    x1,l1,m1,u1 = get_crowding_function_quantiles(res1[:,5:end,:])
    x2,l2,m2,u2 = get_crowding_function_quantiles(res2[:,5:end,:])
    f_gp_plot!(fig4,prior2,subplot=4,bounded=true,ylim=(0.0,1.2))
    f_gp_plot!(fig4,prior2,subplot=8,legend=:none,bounded=true,ylim=(0.0,1.2))
    plot!(fig4,subplot=4,x1,m1,frange=(l1,u1),c=:red,lw=2.0,fα=0.3,label="gp")
    plot!(fig4,subplot=8,x2,m2,frange=(l2,u2),c=:red,lw=2.0,fα=0.3,label="")

    plot!(link=:x,xwiden=true,size=(800,380))
    add_plot_labels!(fig4)
    savefig(fig4,"$(@__DIR__)/fig4.svg")
    fig4