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
    models::Dict{Int,TSCheckpoint}
    mean_rewards::Dict{Int,Float64}
    info::Dict

    function PPTSRun(name::String, model::MCMCModel, info::Dict=Dict())
        hist = new(name, model, Dict(), Dict(), Dict(), info)
        hist.info["name"] = hist.name
        hist.info["date"] = now()
        return hist
    end
end

function run_ts_mcmc!(ts::PPTSRun, index::Int, steps::Int)
    model = ts.model
    functions = get_functions(ts.model.type)
    trace, _ = generate(sepsis_model, (model.policies, model.start_states, functions), model.choices)
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

    policy, V = optimize(param, functions)
    mean_rew = sepsis_gym.evaluate_policy(to_gym_pol(policy), 100000)
    ts.policies[index] = policy

    checkpoint = TSCheckpoint(scores, acceptance, params)

    ts.models[index] = checkpoint
    ts.mean_rewards[index] = mean_rew

    return ts
end


end