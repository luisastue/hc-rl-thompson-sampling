module History
export Checkpoint, PPHistoryRun, run_history_mcmc!

using ..PPModel
using ..SepsisTypes
using ..Sepsis
using ..ValueIteration
using PyCall
sepsis_gym = pyimport("custom_sepsis")

using Dates
using Gen

struct Checkpoint
    scores::Vector
    acceptance::Float64
    params::Vector
    sampled_params::Vector{Dict}
end

mutable struct PPHistoryRun
    name::String
    model::MCMCModel
    policies::Dict{Int,Vector{Policy}}
    models::Dict{Int,Checkpoint}
    mean_rewards::Dict{Int,Vector{Float64}}
    info::Dict

    function PPHistoryRun(name::String, model::MCMCModel, info::Dict=Dict())
        hist = new(name, model, Dict(), Dict(), Dict(), info)
        hist.info["name"] = hist.name
        hist.info["date"] = now()
        return hist
    end
end

function run_history_mcmc!(history::PPHistoryRun, index::Int, steps::Int)
    model = history.model
    functions = get_functions(model.type)
    trace, _ = generate(sepsis_model, (model.policies, model.start_states, functions), model.choices)
    params = [trace[:parameters]]
    scores = [get_score(trace)]
    acceptance = 0.0

    for _ in 1:steps
        trace, a = functions.update(trace, 0.01)
        push!(params, trace[:parameters])
        push!(scores, get_score(trace))
        acceptance += a
    end
    acceptance /= steps

    sampled_params = params[end-9:end]

    mean_rew = []
    policies = []
    for i in 1:10
        param = sampled_params[i]
        policy, V = optimize(param, functions)
        pol = to_gym_pol(policy)
        push!(policies, policy)
        r = sepsis_gym.evaluate_policy(pol, 100000)
        push!(mean_rew, r)
    end
    param = params[end]
    history.model.choices = functions.set_parameters(history.model.choices, param)
    checkpoint = Checkpoint(scores, acceptance, params, sampled_params)

    history.models[index] = checkpoint
    history.policies[index] = policies
    history.mean_rewards[index] = mean_rew

    return history
end

end