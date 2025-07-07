using DifferentialEquations

"""
    Numerical solution to the ODE with given crowding function:

        u'(t) = r * u(t) * f(u(t) / K)

"""
function solve_logistic(pars,f::Function)
    # Solve ODE numerically, crowding function given
    function ode(u,pars,t)
        r,K = pars
        r * u * f(u / K)
    end
    u = solve(ODEProblem(ode,pars[3],extrema(t),pars))
end

"""
    Analytical solution to the ODE with piecewise-linear crowding function

        u'(t) = r * u(t) * f(u(t) / K)

    where f(x) : x ∈ (0,1) → (0,1) is monotonically decreasing, and given at evenly space
    discrete points (xᵢ,fᵢ) where fᵢ ∈ fgrid is given.
"""
function solve_logistic(pars,fgrid::Vector)
    r,K,u₀ = pars
    xgrid = range(0.0,K,length(fgrid))

    # Parameterise r f(u) as aᵢ u + bᵢ
    A = diff(r * fgrid) ./ diff(xgrid)
    B = -A .* xgrid[1:end-1] + r * fgrid[1:end-1]

    # Work out switch times
    T = -1 ./ zeros(length(xgrid) - 1)
    U = zeros(length(xgrid) - 1)
    
    # Where do we start?
    idx = findlast(xgrid .< u₀)
    T[idx] = 0.0
    U[idx] = u₀

    # Remaining "bins"
    for i = idx:length(xgrid)-2
        T[i+1] = logistic_piece_inverse(xgrid[i+1],T[i],U[i],A[i],B[i])
        U[i+1] = xgrid[i+1]
    end

    # Construct the solution
    return t -> begin
        idx = findlast(T .≤ t)
        logistic_piece(t,T[idx],U[idx],A[idx],B[idx])
    end

end

function logistic_piece(t,t₀,u₀,aᵢ,bᵢ)
    bᵢ * exp(bᵢ * t) * u₀ / (exp(bᵢ * t₀) * (bᵢ + aᵢ * u₀) - aᵢ * exp(bᵢ * t) * u₀)
end
function logistic_piece_inverse(v,t₀,u₀,aᵢ,bᵢ)
    log(exp(bᵢ * t₀) * (bᵢ + aᵢ * u₀) * v / u₀ / (bᵢ + aᵢ * v)) / bᵢ
end

