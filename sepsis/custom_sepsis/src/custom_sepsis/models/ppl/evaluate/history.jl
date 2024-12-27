module History
export Checkpoint, PPHistoryRun, run_history_mcmc!

using ..PPModel
using ..SepsisTypes
using ..Sepsis
using ..ValueIteration

using Dates
using Gen

struct Checkpoint
    scores::Vector{Int}
    acceptance::Float64
    params::Dict
    sampled_params::Vector{Dict}
end

mutable struct PPHistoryRun
    name::String
    model::MCMCModel
    mh_update::Any
    policies::Dict{Int,Vector{Policy}}
    models::Dict{Int,Checkpoint}
    mean_rewards::Dict{Int,Vector{Float64}}
    info::Dict

    function PPHistoryRun(name::String, model::MCMCModel, mh_update::Any, info::Dict=Dict())
        hist = new(name, model, mh_update, Dict(), Dict(), Dict(), info)
        hist.info["name"] = hist.name
        hist.info["date"] = now()
        return hist
    end
end

function run_history_mcmc!(history::PPHistoryRun, index::Int, steps::Int)
    model = history.model
    trace, _ = generate(sepsis_model, (model.policies, model.start_states, model.functions), model.choices)
    params = [trace[:parameters]]
    scores = [get_score(trace)]
    acceptance = 0.0

    for _ in 1:steps
        trace, a = history.mh_update(trace, 0.01)
        push!(params, model.functions.extract_parameters(trace))
        push!(scores, get_score(trace))
        acceptance += a
    end
    acceptance /= steps

    posterior = params[end-100:end]

    mean_rew = []
    sampled_params = []
    policies = []
    for i in 1:10
        param = rand(posterior)
        push!(sampled_params, param)
        policy, V = optimize(param, model.functions)
        pol = to_gym_pol(policy)
        push!(policies, pol)
        r = sepsis_gym.evaluate_policy(pol, 100000)
        push!(mean_rew, r)
    end
    checkpoint = Checkpoint(scores, acceptance, params, sampled_params)

    history.models[index] = checkpoint
    history.policies[index] = policies
    history.mean_rewards[history.index] = mean_rew

    return history
end


end