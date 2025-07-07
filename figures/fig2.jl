using Distributions
using Plots, StatsPlots
using .Threads

include("../library/library.jl")
include("defaults.jl")

# Create GP
f_gp = f_gp_prior(10)
D_gp = D_gp_prior(10.0,ρ=2.0)

# Figure 2
fig2a = f_gp_plot(f_gp,bounded=true)
fig2b = D_gp_plot(D_gp,10.0,bounded=true)

# Plot five realisations (bounded such that f(x) ≥ 0)
for i = 1:5
    fgrid = f_gp_rand(f_gp,bounded=true)
    Dgrid = rand(D_gp)
    f = f_create(fgrid)
    D = D_create(10.0,Dgrid)
    plot!(fig2a,f,c=:orange,lw=2.0,label=i==1 ? "f(x) ~ gp" : "")
    plot!(fig2b,D,c=:orange,lw=2.0,label=i==1 ? "D(t) ~ gp" : "")
end
fig2a
fig2b

# Plot crowding function 1 - x^2 (Richards model with β = 2)
plot!(fig2a,x -> 1 - x.^2, c=:black,ls=:dash,label="f(x) = 1 - x^2")

# Figure 2
fig2 = plot(fig2a,fig2b,size=(600,220))
savefig(fig2,"$(@__DIR__)/fig2.svg")
fig2

