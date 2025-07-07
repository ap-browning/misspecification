using AdaptiveMCMC
using MCMCChains


"""
    Perform MCMC
"""
function mcmc(q,fdensity,iters;
    names=["p$i" for i in eachindex(q)],
    algorithm=:aswam,kwargs...)

    # Adaptive MCMC
    res = adaptive_rwm(q, fdensity, iters; algorithm, kwargs...)

    # Create chain
    C = Chains(res.X',names;evidence=reshape(res.D[1],1,length(res.D[1])))

    # Check that we actually moved!
    if length(unique(C.logevidence)) == 1
        # Try again...
        display("Trying again!")
        return mcmc(q,fdensity,iters;names,algorithm,kwargs...)
    else
        return C
    end

end

"""
    Obtain MAP
"""
function get_map(C::Chains)
    idx = findmax(C.logevidence[:])[2]
    C.value[idx,:,:].data[:]
end

"""
    Get range as a matrix
"""
function Matrix(C::Chains)
    C.value.data[:,:,1]
end