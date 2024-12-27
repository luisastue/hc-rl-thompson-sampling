module ThompsonSampling
export Checkpoint, PPTSRun, run_ts_mcmc!

using ..PPModel
using ..SepsisTypes
using ..Sepsis
using ..Inference
using ..ValueIteration
using PyCall
sepsis_gym = pyimport("custom_sepsis")

using Dates
using Gen

struct TSCheckpoint
    scores::Vector
    acceptance::Float64
    params::Vector{Dict}
end

mutable struct PPTSRun
    name::String
    model::MCMCModel
    policies::Dict{Int,Policy}
    models::Dict{Int,Checkpoint}
    rewards::Vector{Float}
    mean_rewards::Dict{Int,Vector{Float64}}
    info::Dict

    function PPTSRun(name::String, model::MCMCModel, info::Dict=Dict())
        hist = new(name, model, Dict(), Dict(), [], Dict(), info)
        hist.info["name"] = hist.name
        hist.info["date"] = now()
        return hist
    end
end

function run_ts_mcmc!(ts::PPTSRun, index::Int, steps::Int)
    model = ts.model
    trace, _ = generate(sepsis_model, (model.policies, model.start_states, model.functions), model.choices)
    params = [trace[:parameters]]
    scores = [get_score(trace)]
    acceptance = 0.0

    for _ in 1:steps
        trace, a = get_update_function(ts.model.type)(trace, 0.01)
        push!(params, trace[:parameters])
        push!(scores, get_score(trace))
        acceptance += a
    end
    acceptance /= steps

    param = params[end]

    policy, V = optimize(param, model.functions)
    mean_rew = evaluate_policy(to_gym_pol(policy), model.start_states, model.functions, 50000)
    ts.policies[index] = policy

    checkpoint = Checkpoint(scores, acceptance, params)

    ts.models[index] = checkpoint
    ts.mean_rewards[index] = mean_rew

    return ts
end


end