using Distributions
using LinearAlgebra
using Random

"""
    MvUniform(P)

Construct a Gaussian copula with correlation matrix `P`.
"""
abstract type Copula end
struct MvUniform{N} <: Copula
    X::MvNormal         # Underlying multivariate normal distribution
end
MvUniform(P::Array) = MvUniform{size(P,1)}(MvNormal(P))

Base.rand(rng::AbstractRNG,U::MvUniform{N}) where {N} = cdf.(Normal(),rand(rng,U.X))
Base.rand(rng::AbstractRNG,U::MvUniform{N},n::Int) where {N} = cdf.(Normal(),rand(rng,U.X,n))
Base.rand(U::MvUniform{N},n::Int) where {N} = cdf.(Normal(),rand(U.X,n))
Base.rand(U::MvUniform{N}) where {N} = cdf.(Normal(),rand(U.X))

Distributions.logpdf(U::MvUniform{N},u::Vector) where {N} = (x = quantile(Normal(),u); logpdf(U.X,x) + length(x) / 2 * log(2π) + 0.5 * dot(x,x))
Distributions.insupport(U::MvUniform,u::Vector) = all(0.0 .< u .< 1.0)

function D_gp_prior(tmax,nknots=20;ρ=2.0,α=1e-8)

    # Kernel
    k(t₁,t₂) = exp(-(t₁ - t₂)^2 / 2ρ^2)

    # Create the distribution
    t = [D_gp_knots(tmax,nknots); tmax]
    Σ = [k(t₁,t₂) for t₁ in t, t₂ in t]
    Σ = (Σ + Σ') / 2 + α * I

    # Conditional on u(tmax) = 0.5 ⇒ x(tmax) = 0.0
    Σ̂ = Σ[1:end-1,1:end-1] - Σ[1:end-1,end] * Σ[end,1:end-1]' / Σ[end,end]
    Σ̂ = 0.5 * (Σ̂ + Σ̂)

    # Create MvUniform distribution
    MvUniform(Σ̂)

end

D_gp_knots(tmax,nknots=20) = range(0.0,tmax,nknots+1)[1:end-1]


"""
    Plot the GP prior
"""
function D_gp_plot!(plt,gp,tmax;kwargs...)
    x = Normal.(0.0,std(gp.X))
    t = range(0.0,tmax,length(x)+1)
    μ = ones(size(t))
    fl = [2cdf(Normal(0.0,1.0),quantile.(x,0.025)); 1.0]
    fu = [2cdf(Normal(0.0,1.0),quantile.(x,0.975)); 1.0]
    plot!(plt,t,μ,frange=(fl, fu);fα=0.1,c=:black,α=0.5,lw=2.0,label="prior",
    xlabel="t",ylabel="D(t)",kwargs...)
end
D_gp_plot(args...;kwargs...) = D_gp_plot!(plot(),args...;kwargs...)

"""
    Interior points to an interpolated piecewise-linear function
"""
function D_create(tmax::Number,ȳ::Vector)
    y = 2 * [ȳ; 0.5]
    t = range(0.0,tmax,length(y))
    itp = linear_interpolation(t,y,extrapolation_bc=Flat())
    t -> itp(t)
end

"""
    Crowding function quantiles from the posterior distribution
"""
function get_diffusivity_function_quantiles(tmax,samples;factor=ones(size(samples,1)))
    t = [D_gp_knots(tmax,size(samples,2)); tmax]
    Dt = 2factor .* samples
    l = [[quantile(fᵢ,0.025) for fᵢ in eachcol(Dt)]; quantile(factor,0.025)]
    μ = [[mean(fᵢ) for fᵢ in eachcol(Dt)]; mean(factor)]
    u = [[quantile(fᵢ,0.975) for fᵢ in eachcol(Dt)]; quantile(factor,0.975)]
    return t,l,μ,u
end

