"""
    Get quantiles from a posterior distribution of parameters, where
    
        model(lp)(t)

    gives the model solution at parameters lp and time t.
"""
function get_model_prediction_quantiles(model,samples,t;n=1000)
    # Create storage
    Y = zeros(length(t),n)
    # Calculate quantiles
    lp = rand([eachcol(samples)...],n)
    for i = 1:n
        Y[:,i] = model(lp[i]).(t)
    end
    l = [quantile(y,0.025) for y in eachrow(Y)]
    μ = [mean(y) for y in eachrow(Y)]
    u = [quantile(y,0.975) for y in eachrow(Y)]
    return l,μ,u
end
get_model_prediction_quantiles(model,C::Chains,t;kwargs...) = get_model_prediction_quantiles(model,Matrix(C)',t;kwargs...)