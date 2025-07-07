using Distributions
using Plots, StatsPlots
using AdaptiveMCMC

include("../library/library.jl")
include("defaults.jl")

## GENERATE DATA

    # True parameters
    r  = 1.0
    K  = 5e-3
    u₀ = [K/10, K/2]
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

    # Gridded crowding function
    xxgrid = range(0.0,1.0,12)
    ffgrid = f.(xxgrid)

## INFERENCE (INCL. ON CROWDING FUNCTION)

    # True parameters
    lp = [[log.(p); ffgrid[2:end-1]] for p in pars]

    # Likelihood function (infer crowding function)
    function loglike_gp(lp,data)
        pars  = exp.(lp[1:4])
        fgrid = lp[5:end]
        u = solve_logistic(pars,[1.0; fgrid; 0.0])
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Likelihood function (misspecified, assume logistic)
    function loglike_logistic(lp,data)
        pars  = exp.(lp[1:4])
        u = solve_logistic(pars,[1.0, 0.0])
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Likelihood function (true model)
    function loglike_true(lp,data)
        pars  = exp.(lp[1:4])
        u = solve_logistic(pars,f)
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Likelihood function (Richard's model)
    function loglike_richards(lp,data)
        pars  = exp.(lp[1:5])
        f = x -> 1 - max(0,x)^pars[5]
        u = solve_logistic(pars[1:4],f)
        loglikelihood(ε(pars[4]),data .- u.(t))
    end

    # Priors
    prior1 = Product(Uniform.(fill(-10.0,4),fill(2.0,4)))
    prior2 = f_gp_prior(length(ffgrid)-2)
    priorβ = Uniform(-2.0,2.0)

    # Posteriors
    logpost(lp,data) = insupport(prior1,lp[1:4]) && all(0.0 .≤ lp[5:end]) && lp[3] < lp[2] ? 
        loglike_gp(lp,data) + 
            logpdf(prior1,lp[1:4]) + 
            logpdf(prior2,lp[5:end]) : -Inf

    logpost_logistic(lp,data) = insupport(prior1,lp[1:4]) && lp[3] < lp[2] ? 
        loglike_logistic(lp,data) + 
            logpdf(prior1,lp[1:4]) : -Inf

    logpost_true(lp,data) = insupport(prior1,lp[1:4]) && lp[3] < lp[2] ? 
        loglike_true(lp,data) + 
            logpdf(prior1,lp[1:4]) : -Inf

    logpost_richards(lp,data) = insupport(prior1,lp[1:4]) && lp[3] < lp[2] && insupport(priorβ,lp[end]) ? 
        loglike_richards(lp,data) + 
            logpdf(prior1,lp[1:4]) + logpdf(priorβ,lp[5]) : -Inf

    ## Inference
    @time res1 = mcmc(lp[1], lp -> logpost(lp,data[1]), 500000);
    @time res2 = mcmc(lp[2], lp -> logpost(lp,data[2]), 500000);

    @time res1mis = mcmc(lp[1][1:4], lp -> logpost_logistic(lp,data[1]), 500000);
    @time res2mis = mcmc(lp[2][1:4], lp -> logpost_logistic(lp,data[2]), 500000);

    @time res1tru = mcmc(lp[1][1:4], lp -> logpost_true(lp,data[1]), 500000);
    @time res2tru = mcmc(lp[2][1:4], lp -> logpost_true(lp,data[2]), 500000);

    @time res1ric = mcmc([lp[1][1:4]; log(2.0)], lp -> logpost_richards(lp,data[1]), 500000);
    @time res2ric = mcmc([lp[2][1:4]; log(2.0)], lp -> logpost_richards(lp,data[2]), 500000);

## Figure 3

fig3 = plot(layout=grid(2,3),size=(800,400),link=:x)

    # Posteriors for data set 1
    density!(fig3,subplot=1,exp.(res1[:p1]),c=:red,frange=0.0,fα=0.3,label="gp",xlabel="r",ylabel="Posterior density",ywiden=false)
    density!(fig3,subplot=1,exp.(res1mis[:p1]),c=:blue,frange=0.0,fα=0.3,label="logistic")
    density!(fig3,subplot=1,exp.(res1tru[:p1]),c=:purple,frange=0.0,fα=0.3,label="true")
    density!(fig3,subplot=1,exp.(res1ric[:p1]),c=:orange,frange=0.0,fα=0.3,label="richards")
    vline!(fig3,subplot=1,[r],c=:black,ls=:dash,lw=2.0,label="")

    # Posteriors for data set 2
    density!(fig3,subplot=4,exp.(res2[:p1]),c=:red,frange=0.0,fα=0.3,label="",xlabel="r",ylabel="Posterior density",ywiden=false)
    density!(fig3,subplot=4,exp.(res2mis[:p1]),c=:blue,frange=0.0,fα=0.3,label="")
    density!(fig3,subplot=4,exp.(res2tru[:p1]),c=:purple,frange=0.0,fα=0.3,label="")
    density!(fig3,subplot=4,exp.(res2ric[:p1]),c=:orange,frange=0.0,fα=0.3,label="")

    vline!(fig3,subplot=4,[r],c=:black,ls=:dash,lw=2.0,label="")
    plot!(fig3,subplot=1,xlim=(0.5,2.0),xwiden=false)
    plot!(fig3,subplot=4,xlim=(0.5,2.0),xwiden=false)

    # Inferred crowding functions
    x1,l1,m1,u1 = get_crowding_function_quantiles(res1[:,5:end,:])
    x2,l2,m2,u2 = get_crowding_function_quantiles(res2[:,5:end,:])
    x3,l3,m3,u3 = get_crowding_function_quantiles(collect(res1ric[:p5])[:],(p,x) -> 1 .- x.^exp.(p))
    x4,l4,m4,u4 = get_crowding_function_quantiles(collect(res2ric[:p5])[:],(p,x) -> 1 .- x.^exp.(p))

    f_gp_plot!(fig3,prior2,subplot=2,bounded=true,ylim=(0.0,1.2))
    f_gp_plot!(fig3,prior2,subplot=5,legend=:none,bounded=true,ylim=(0.0,1.2))

    plot!(fig3,subplot=2,x1,m1,frange=(l1,u1),c=:red,lw=2.0,fα=0.3,label="gp")
    plot!(fig3,subplot=5,x2,m2,frange=(l2,u2),c=:red,lw=2.0,fα=0.3,label="")
    plot!(fig3,subplot=2,x3,m3,frange=(l3,u3),c=:orange,lw=2.0,fα=0.3,label="richards")
    plot!(fig3,subplot=5,x4,m4,frange=(l4,u4),c=:orange,lw=2.0,fα=0.3,label="")

    plot!(fig3,subplot=2,f,c=:black,ls=:dash,lw=2.0,label="true",xlim=(0.0,1.0))
    plot!(fig3,subplot=5,f,c=:black,ls=:dash,lw=2.0,label="",xlim=(0.0,1.0))

    # Plot data
    scatter!(fig3,subplot=3,t,data[1],c=:black,msw=0.0,α=0.2,label="",xlabel="t",ylabel="Cell density")
    scatter!(fig3,subplot=6,t,data[2],c=:black,msw=0.0,α=0.2,label="",xlabel="t",ylabel="Cell density")

    # Plot predictions
    tv = range(extrema(t)...,101)
    model = lp -> solve_logistic(exp.(lp[1:4]),[1.0; lp[5:end]; 0.0])
    modelmis = lp -> solve_logistic(exp.(lp[1:4]),[1.0, 0.0])
    l1,m1,u1 = get_model_prediction_quantiles(model,res1,tv)
    l1mis,m1mis,u1mis = get_model_prediction_quantiles(modelmis,res1mis,tv)
    l2,m2,u2 = get_model_prediction_quantiles(model,res2,tv)
    l2mis,m2mis,u2mis = get_model_prediction_quantiles(modelmis,res2mis,tv)

    plot!(fig3,subplot=3,tv,m1,frange=(l1,u1),c=:red,label="",fα=0.2,lw=2.0)
    plot!(fig3,subplot=3,tv,m1mis,frange=(l1mis,u1mis),c=:blue,label="",fα=0.3,lw=2.0)
    plot!(fig3,subplot=6,tv,m2,frange=(l2,u2),c=:red,label="",fα=0.3,lw=2.0)
    plot!(fig3,subplot=6,tv,m2mis,frange=(l2mis,u2mis),c=:blue,label="",fα=0.3,lw=2.0)

    plot!(link=:x)
    add_plot_labels!(fig3)
    savefig(fig3,"$(@__DIR__)/fig3.svg")
    fig3