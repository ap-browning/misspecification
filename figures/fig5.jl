using Distributions
using Plots, StatsPlots
using AdaptiveMCMC
using JLD2

include("../library/library.jl")
include("defaults.jl")

## GENERATE DATA

    # True parameters (in the PDE)
    Dmax = 2e3
    r    = 1.0
    K    = 5e-3
    u₀   = 0.1K
    σ₁   = 5e-5
    σ₂   = 20.0

    p = [r,K,u₀,Dmax,σ₁,σ₂]

    # True diffusivity function
    tmax = 10.0
    D̂ = t -> t^3 / (t^3 + 3) + 0.1
    D = t -> D̂(t) / D̂(tmax)

    # Observation times
    t = range(0.0,tmax,21)

    # Replicates
    n = 5

    # Noise model (properly specified)
    ε = σ -> MvNormal(σ * ones(length(t)))

    # Initial condition
    xmax = 2000.0
    ic = x -> x / xmax < 0.1 ? 1.0 : 0.0

    # Solve the PDE
    U,F = solve_fkpp(p,D;xmax,tmax=maximum(t),ic,ret=:summary)

    Udata = U.(t) .+ rand(ε(σ₁),n)
    Fdata = F.(t) .+ rand(ε(σ₂),n)
    #@save "$(@__DIR__)/fig5_data.jld2" Udata Fdata
    @load "$(@__DIR__)/fig5_data.jld2" Udata Fdata

    scatter(t,Udata)
    scatter(t,Fdata)

## INFERENCE ON CONSTANT DIFFUSIVITY PDE MODEL

    lp = log.(p)

    # Solve model...
    function solve_model1(lp)
        pars = exp.(lp)
        solve_fkpp(pars;xmax,tmax=maximum(t),ic,ret=:summary)
    end

    # Likelihood
    function loglike1(lp)
        pars = exp.(lp)
        U,F = solve_fkpp(pars;xmax,tmax=maximum(t),ic,ret=:summary)
        loglikelihood(ε(pars[end-1]),Udata .- U.(t)) + 
        loglikelihood(ε(pars[end]),Fdata .- F.(t))
    end

    # Prior
    prior1 = Product(Uniform.(fill(-10.0,6),fill(10.0,6)))

    # Posterior
    logpost1(lp) = insupport(prior1,lp) ? loglike1(lp) + logpdf(prior1,lp) : -Inf

    # Inference (1 minute per 10000 samples)
    #@time res1 = mcmc(lp,logpost1,500000)
    #@save "$(@__DIR__)/fig5_res1.jld2" res1
    @load "$(@__DIR__)/fig5_res1.jld2" res1

## VARIABLE DIFFUSIVITY MODEL

    ttgrid = D_gp_knots(tmax)
    DDgrid = max.(1e-5,D.(ttgrid) / 2)

    lp = [log.(p); DDgrid]

    # Solve model...
    function solve_model2(lp)
        pars = exp.(lp[1:6])
        D = D_create(tmax,lp[7:end])
        solve_fkpp(pars,D;xmax,tmax=maximum(t),ic,ret=:summary)
    end

    # Likelihood
    function loglike2(lp)
        pars = exp.(lp[1:6])
        U,F = solve_model2(lp)
        loglikelihood(ε(pars[end-1]),Udata .- U.(t)) + 
        loglikelihood(ε(pars[end]),Fdata .- F.(t))
    end

    # Prior
    prior1 = Product(Uniform.(fill(-10.0,6),fill(10.0,6)))
    prior2 = D_gp_prior(tmax,ρ=2.0)

    # Posterior
    logpost2(lp) = insupport(prior1,lp[1:6]) && insupport(prior2,lp[7:end]) ? 
        loglike2(lp) + 
        logpdf(prior1,lp[1:6]) + 
        logpdf(prior2,lp[7:end]) : -Inf

    # Inference (2 minute per 10000 samples)
    #@time res2 = mcmc(lp,logpost2,500000);
    #@save "$(@__DIR__)/fig5_res2.jld2" res2
    @load "$(@__DIR__)/fig5_res2.jld2" res2

## Figure 5

    # (a) PDE
    tv = range(0.0,tmax,6)
    sol = solve_fkpp(p,D;xmax,tmax=maximum(t),ic,ret=:full)
    fig5a = plot([x -> sol(x,t) for t in tv],xlim=(0.0,xmax),palette=palette(:Purples,rev=true),
        label="",xlabel="x",ylabel="u(x,t)")
    for (i,t) in enumerate(tv)
        scatter!(fig5a,[F(t)],[1e-4],c=palette(:Purples,rev=true)[i],label="")
    end
    plot!(fig5a,xlim=(0.0,1500))

    # (b,c) Data and fits
    fig5b = scatter(t,Udata,c=:black,α=0.3,label="",xlabel="t",ylabel="Overall cell density")
    fig5c = scatter(t,Fdata,c=:black,α=0.3,label="",xlabel="t",ylabel="Front location [μm]")

    U1,F1 = solve_model1(get_map(res1))
    U2,F2 = solve_model2(get_map(res2))

    plot!(fig5b, t -> U1(t), xlim = (0.0,tmax),c=:blue,lw=1.5,label="D(t) = const")
    plot!(fig5b, t -> U2(t), xlim = (0.0,tmax),c=:red,lw=1.5,label="D(t) ~ GP")

    plot!(fig5c, t -> F1(t), xlim = (0.0,tmax),c=:blue,lw=1.5,label="D(t) = const")
    plot!(fig5c, t -> F2(t), xlim = (0.0,tmax),c=:red,lw=1.5,label="D(t) ~ GP")

    # (d,e) Posteriors for r and D(tmax)
    fig5d = plot(ywiden=false,xlabel="r",ylabel="Posterior density")
    density!(fig5d, exp.(res1[:p1]), c=:blue, frange=0.0, lw=1.0, fα=0.3,label="D(t) = const")
    density!(fig5d, exp.(res2[:p1]), c=:red, frange=0.0, lw=1.0, fα=0.3,label="D(t) ~ GP")
    vline!(fig5d,[r],ls=:dot,lw=1.5,c=:black,label="",ylim=(0.0,8.5))

    fig5e = plot(ywiden=false,xlabel="Dmax",ylabel="Posterior density")
    density!(fig5e, exp.(res1[:p4]), c=:blue, frange=0.0, lw=1.0, fα=0.3,label="D(t) = const")
    density!(fig5e, exp.(res2[:p4]), c=:red, frange=0.0, lw=1.0, fα=0.3,label="D(t) ~ GP")
    vline!(fig5e,[Dmax],ls=:dot,lw=1.5,c=:black,label="",ylim=(0.0,0.011))

    # (f) Inferred D(t)
    tv,l,μ,u = get_diffusivity_function_quantiles(tmax,Matrix(res2)[:,7:end];factor=exp.(res2[:p4])[:])
    lc,uc = quantile(exp.(res1[:p4])[:],[0.025,0.975]) 
    μc = mean(exp.(res1[:p4])[:])

    fig5f = plot(xlabel="t",ylabel="D(t)")
    hline!(fig5f,[μc],frange=(lc,uc),c=:blue,fα=0.3,lw=1.5,label="D(t) = const",xlim=(0.0,10.0))
    plot!(fig5f,tv,μ,frange=(l,u),c=:red,fα=0.3,lw=1.5,label="D(t) ~ GP")
    plot!(t -> Dmax * D(t),c=:black,lw=1.5,ls=:dot,label="True D(t)")
    fig5f

    # Figure 5
    fig5 = plot(fig5a,fig5b,fig5c,fig5d,fig5e,fig5f,layout=grid(2,3),size=(800,400),xwiden=true)
    add_plot_labels!(fig5)
    savefig(fig5,"$(@__DIR__)/fig5.svg")
    fig5