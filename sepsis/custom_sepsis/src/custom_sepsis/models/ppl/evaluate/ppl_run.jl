module PPLRun
export run_mcmc, evaluate_mcmc, run_batch, thompson_sampling, history_run, evaluate_on_samples

using ..PPModel
using ..SepsisTypes
using ..Sepsis
using ..ValueIteration
using ..Save
using PyCall
sepsis_gym = pyimport("custom_sepsis")

using Gen

function run_mcmc(model::ModelData, functions, steps::Int)
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
    return MCMCRun(scores, acceptance, params)
end

function evaluate_mcmc(mcmc::MCMCRun, functions)
    # perform planning
    param = mcmc.params[end]
    policy, V = optimize(param, functions)

    # evaluate policy
    mean_reward = sepsis_gym.evaluate_policy(to_gym_pol(policy), 100000)
    return mean_reward, policy
end


function evaluate_on_samples(mcmc::MCMCRun, functions, n_samples::Int)
    # perform planning
    posterior_size = Int(round(length(mcmc.scores[end]) / 3))
    posterior = mcmc.params[end-posterior_size:end]
    mean_rewards = []
    sampled_params = []
    policies = []
    for i in 1:n_samples
        param = rand(posterior)
        push!(sampled_params, param)
        policy, V = optimize(param, functions)
        push!(policies, policy)
        # evaluate policy
        mean_reward = sepsis_gym.evaluate_policy(to_gym_pol(policy), 100000)
        push!(mean_rewards, mean_reward)
    end

    return mean_rewards, policies, sampled_params
end

function run_batch(model_data::ModelData, functions, get_policy, batch_size::Int, param)
    # perform environment interaction
    pols = [get_policy() for _ in 1:batch_size]
    episodes = [sepsis_gym.run_episode(pol) for pol in pols]
    policies = [to_policy(pol) for pol in pols]

    # add data to model (for next iteration)
    start_states = [to_state(episode.visited[1]) for episode in episodes]
    choices = functions.set_parameters(model_data.choices, param)
    for (i, episode) in enumerate(episodes)
        choices = update_choicemap!(choices, i, episode)
    end
    return ModelData(
        choices,
        vcat(model_data.policies, policies),
        vcat(model_data.start_states, start_states)
    )
end


function thompson_sampling(run_data::RunData, batch_size::Int, nr_batches::Int)
    for i in 1:nr_batches
        iterations = i == 1 && length(run_data.model.policies) == 0 ? 200 : 50
        functions = get_functions(run_data.type)
        mcmc_data = run_mcmc(run_data.model, functions, iterations)
        run_data.mcmcs[run_data.index] = mcmc_data
        mean_reward, policy = evaluate_mcmc(mcmc_data, functions)
        run_data.mean_rewards[run_data.index] = mean_reward
        model_data = run_batch(run_data.model, functions, () -> to_gym_pol(policy), batch_size, mcmc_data.params[end])
        run_data = RunData(run_data.name, run_data.type, model_data, run_data.mcmcs, run_data.mean_rewards, run_data.index + batch_size)
        save_run_jld(run_data)
        return run_data
    end
end


function history_run(run_data::RunData, batch_size::Int, nr_batches::Int)
    for i in 1:nr_batches
        iterations = i == 1 && length(run_data.model.policies) == 0 ? 200 : 50
        functions = get_functions(run_data.type)
        mcmc_data = run_mcmc(run_data.model, functions, iterations)
        run_data.mcmcs[run_data.index] = mcmc_data
        mean_reward, policy = evaluate_mcmc(mcmc_data, functions)
        run_data.mean_rewards[run_data.index] = mean_reward
        model_data = run_batch(run_data.model, functions, sepsis_gym.random_policy, batch_size, mcmc_data.params[end])
        run_data = RunData(run_data.name, run_data.type, model_data, run_data.mcmcs, run_data.mean_rewards, run_data.index + batch_size)
        save_run_jld(run_data)
        return run_data
    end
end

end