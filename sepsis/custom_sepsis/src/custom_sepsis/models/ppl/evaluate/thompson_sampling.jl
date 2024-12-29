module ThompsonSampling
export Checkpoint, PPTSRun, run_ts_mcmc!, update_model!

using ..PPModel
using ..SepsisTypes
using ..Sepsis
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

function update_model!(model::MCMCModel, until::Int, policy::Policy)
    for i in length(model.policies):until
        pol = policy
        if policy !== nothing
            pol = to_gym_pol(policy)
        else
            pol = sepsis_gym.random_policy()
            policy = to_policy(pol)
        end
        episode = sepsis_gym.run_episode(pol)
        model.choices = update_choicemap!(model.choices, i, episode)
        start_state = to_state(episode.visited[1])
        trace, sc = generate(sepsis_model, ([policy], [start_state], get_functions(model.type)), model.choices)
        if sc == -Inf
            println(i, " Score was -Inf.")
        end
        push!(model.policies, policy)
        push!(model.start_states, start_state)
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
        trace, a = functions.update(trace, 0.01)
        push!(params, trace[:parameters])
        push!(scores, get_score(trace))
        acceptance += a
    end
    acceptance /= steps

    param = params[end]
    ts.model.choices = functions.set_parameters(ts.model.choices, param)

    policy, V = optimize(param, functions)
    mean_rew = sepsis_gym.evaluate_policy(to_gym_pol(policy), 100000)
    ts.policies[index] = policy

    checkpoint = TSCheckpoint(scores, acceptance, params)

    ts.models[index] = checkpoint
    ts.mean_rewards[index] = mean_rew

    return ts
end


end