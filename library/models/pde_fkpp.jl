#=
    Scratch assay PDE
=#

"""
    Solve one-dimensional Fisher-KPP model

        ∂u∂t = D ∇²u + r u (1 - u / K)

    subject to the initial condition
        
        u(x,0) = u₀ * ic(x / xmax)

    were x ∈ [0,xmax].

    Default kwargs:
        xmax    = 2000
        ic      = x -> (0.3 < x < 0.7) ? 0.0 : 1.0
        tmax    = 10
        N       = 201  (spatial discretisation)
        ret     = :average (to return U(t) = ∫ₓ u(x,t) dx).

"""
function solve_fkpp(pars;xmax = 2000,tmax = 10,N = 201,ic=x -> default_ic(x / xmax),ret=:average,β=1e-4)
    # Parameters
    r,K,u₀,D = pars
    # Discretisation
    x = range(0.0,xmax,N)
    Δ = x[2] - x[1]
    # Numerical solution
    function pde!(du,u,p,t)
        uₓₓ = [2(u[2] - u[1]); (diff ∘ diff)(u); 2(u[end-1] - u[end])]  / Δ^2
        du .= D * uₓₓ + r * u .* (1 .- u / K)
    end
    sol = solve(ODEProblem(pde!,u₀ * ic.(x),(0.0,tmax)))
    # Solution handling
    if ret == :average
        # Averaged solution (trap rule)
        U = [Δ / 2 * sum(u[1:end-1] + u[2:end]) for u in sol.u] / xmax
        return linear_interpolation(sol.t,U,extrapolation_bc=Flat())
    elseif ret == :summary
        # Averaged solution (trap rule)
        U = linear_interpolation(sol.t,[Δ / 2 * sum(u[1:end-1] + u[2:end]) for u in sol.u]  / xmax,extrapolation_bc=Flat())
        # Location of boundary
        F = linear_interpolation(sol.t,[get_boundary(x,u,β) for u in sol.u],extrapolation_bc=Flat())
        return U,F
    else
        # Interpolated solution
        return interpolate((x,sol.t),hcat(sol.u...),Gridded(Linear()))
    end
end

default_ic = x -> (0.3 < x < 0.7) ? 0.0 : 1.0

"""
    Solve Fisher-KPP with time-varying diffusivity, D(t) = Dmax * D̂(t), where D̂(t) is given as the second input.
"""
function solve_fkpp(pars,D::Function;xmax = 2000,tmax = 10,N = 201,ic=x -> default_ic(x / xmax),ret=:average,β=1e-4)
    # Parameters
    r,K,u₀,Dmax = pars
    # Discretisation
    x = range(0.0,xmax,N)
    Δ = x[2] - x[1]
    # Numerical solution
    function pde!(du,u,p,t)
        uₓₓ = [2(u[2] - u[1]); (diff ∘ diff)(u); 2(u[end-1] - u[end])]  / Δ^2
        du .= Dmax * D(t) * uₓₓ + r * u .* (1 .- u / K)
    end
    sol = solve(ODEProblem(pde!,u₀ * ic.(x),(0.0,tmax)))
    # Solution handling
    if ret == :average
        # Averaged solution (trap rule)
        U = [Δ / 2 * sum(u[1:end-1] + u[2:end]) for u in sol.u] / xmax
        return linear_interpolation(sol.t,U,extrapolation_bc=Flat())
    elseif ret == :summary
        # Averaged solution (trap rule)
        U = linear_interpolation(sol.t,[Δ / 2 * sum(u[1:end-1] + u[2:end]) for u in sol.u] / xmax,extrapolation_bc=Flat())
        # Location of boundary
        F = linear_interpolation(sol.t,[get_boundary(x,u,β) for u in sol.u],extrapolation_bc=Flat())
        return U,F
    else
        # Interpolated solution
        return interpolate((x,sol.t),hcat(sol.u...),Gridded(Linear()))
    end
end

"""
    Get the location of the boundary, assuming that u is monotonically decreasing.
"""
function get_boundary(x,u,β)
    if u[end] > β
        z = x[end]
    elseif u[1] < β
        z = 0
    else
        idx = findfirst(u .< β)
        x₁,u₁ = x[idx-1],u[idx-1]
        x₂,u₂ = x[idx],u[idx]
        z = x₁ + (β - u₁) / (u₂ - u₁) * (x₂ - x₁)
    end
    return z
end


"""
    Solve Fisher-KPP with density-dependent diffusivity and proliferation rate
"""
function solve_fkpp(pars,D::Function,f::Function;xmax = 2000,tmax = 10,N = 201,ic=x -> default_ic(x / xmax),saveat=0:12:36)
    # Parameters
    r,K,u₀,Dmax = pars
    # Discretisation
    x = range(0.0,xmax,N)
    Δ = x[2] - x[1]
    # Numerical solution
    function pde!(du,u,p,t)
        uₓₓ = [2(u[2] - u[1]); (diff ∘ diff)(u); 2(u[end-1] - u[end])]  / Δ^2
        du .= Dmax * D.(u / K) .* uₓₓ + r * u .* f.(u / K)
    end
    sol = solve(ODEProblem(pde!,u₀ * ic.(x),(0.0,tmax));saveat)
    # Interpolated solution
    return interpolate((x,sol.t),hcat(sol.u...),Gridded(Linear()))
end