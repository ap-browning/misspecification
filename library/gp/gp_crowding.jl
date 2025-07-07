using Distributions
using LinearAlgebra
using Interpolations
using MCMCChains

"""
    Create a GP prior for the discretised crowding function.
"""
function f_gp_prior(nknots=10;σ=0.2,ρ=0.5)

    # Kernel
    k(x₁,x₂) = σ^2 * exp(-(x₁ - x₂)^2 / 2ρ^2) * x₁ * (1 - x₁) * x₂ * (1 - x₂) / 0.5^4
    μ(x) = 1 - x

    # Create the distribution
    x = f_gp_knots(nknots)
    M = μ.(x)
    Σ = [k(x₁,x₂) for x₁ in x, x₂ in x]
    Σ = (Σ + Σ') / 2
    
    # Return    
    MvNormal(M,Σ)

end
f_gp_knots(nknots=10) = range(0.0,1.0,nknots + 2)[2:end-1]

"""
    Plot the GP prior
"""
function f_gp_plot!(plt,gp;bounded=false,kwargs...)
    nknots = length(gp.μ)    
    x = [0.0; f_gp_knots(nknots); 1.0]
    μ = [1.0; mean(gp); 0.0]
    σ = [0.0; sqrt.(var(gp)); 0.0]
    d = Normal.(μ,σ)
    if bounded 
        d = Truncated.(d,0.0,Inf)
    end
    fl = quantile.(d,0.025) .- eps(); fl[1] = 1.0; fl[end] = 0.0;  # bug fix for zero variance truncated
    fu = quantile.(d,0.975) .+ eps(); fu[1] = 1.0; fu[end] = 0.0; 
    plot!(plt,x,μ,frange=(fl, fu);fα=0.1,c=:black,α=0.5,lw=2.0,label="prior",
        xlabel="x",ylabel="f(x)",kwargs...)
end
f_gp_plot(args...;kwargs...) = f_gp_plot!(plot(),args...;kwargs...)

"""
    Interior points of crowding function to an interpolated piecewise-linear function.
"""
function f_create(ȳ::Vector)
    y = [1.0; ȳ; 0.0]
    x = range(0.0,1.0,length(y))
    itp = linear_interpolation(x,y,extrapolation_bc=Flat())
    x -> itp(x)
end
f_create(gp::Distribution=f_gp_prior()) = f_create(rand(gp))

"""
    Get crowding function quantiles from the posterior distribution
"""
function get_crowding_function_quantiles(samples)
    x = range(0.0,1.0,size(samples,1)+2)
    l = [1.0; [quantile(fᵢ,0.025) for fᵢ in eachrow(samples)]; 0.0]
    μ = [1.0; [mean(fᵢ) for fᵢ in eachrow(samples)]; 0.0]
    u = [1.0; [quantile(fᵢ,0.975) for fᵢ in eachrow(samples)]; 0.0]
    return x,l,μ,u
end
get_crowding_function_quantiles(C::Chains) = get_crowding_function_quantiles(Matrix(C)')


function get_crowding_function_quantiles(pars,func)
    x = range(0.0,1.0,51)
    l = [quantile(func(pars,xᵢ),0.025) for xᵢ in x]
    μ = [mean(func(pars,xᵢ)) for xᵢ in x]
    u = [quantile(func(pars,xᵢ),0.975) for xᵢ in x]
    return x,l,μ,u
end

"""
    Sampling with rejection to ensure that 0 ≤ fᵢ ≤ 1
"""
function f_gp_rand(gp;bounded=true)
    fgrid = rand(gp)
    if !bounded
        return fgrid
    else
        while !all(0 .≤ fgrid)
            fgrid = rand(gp)
        end
        return fgrid
    end
end