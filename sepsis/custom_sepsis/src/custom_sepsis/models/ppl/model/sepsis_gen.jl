

module Sepsis
export get_transition_matrix, sepsis_model, get_reward, get_next_state

using Gen
using ..SepsisTypes
using LinearAlgebra
using PyCall
sepsis_gym = pyimport("custom_sepsis")


@dist function labeled_categorical(labels, probs)
    index = categorical(probs)
    labels[index]
end

function get_reward(state::State)::Int
    reward = 0
    critical_counts = count(c -> c != NORMAL, [state.hr, state.bp, state.o2, state.glu])
    if critical_counts >= 3
        reward = -1
    elseif critical_counts == 0 && !state.abx && !state.vaso && !state.vent
        reward = 1
    end
    return reward
end

function random_initial_state()::State
    return State(
        Level(rand(-1:1)),
        Level(rand(-1:1)),
        Level(rand(-1:0)),
        Level(rand(-2:2)),
        rand(Bool),
        false,
        false,
        false
    )
end


@gen function get_next_state(params::Parameters, state::State, action::Action, functions::SepsisParams)::State
    hr ~ labeled_categorical(HR_LEVELS, functions.hr_probs(params, state, action))
    bp ~ labeled_categorical(BP_LEVELS, functions.bp_probs(params, state, action))
    o2 ~ labeled_categorical(O2_LEVELS, functions.o2_probs(params, state, action))
    glu ~ labeled_categorical(GLU_LEVELS, functions.glu_probs(params, state, action))
    next_state = State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)
    return next_state
end

@gen function simulate_episode(beliefs::Parameters, policy::Policy, start_state::State, functions::SepsisParams)
    states = [start_state]
    state = start_state
    rewards = []
    for t in 1:10
        new_state = {:trajectory => t} ~ get_next_state(beliefs, state, policy[state], functions)
        push!(states, new_state)
        reward = get_reward(new_state)
        push!(rewards, reward)
        state = new_state
        if reward != 0
            break
        end
    end
    return states, rewards
end

@gen function sepsis_model(policies::Vector{Policy}, start_states::Vector{State}, functions::SepsisParams)
    parameters = {:parameters} ~ functions.get_parameters()
    episodes = []
    for (i, policy) in enumerate(policies)
        start_state = start_states[i]
        episode = {:episodes => i} ~ simulate_episode(parameters, policy, start_state, functions)
        push!(episodes, episode)
    end
    return parameters
end


end

