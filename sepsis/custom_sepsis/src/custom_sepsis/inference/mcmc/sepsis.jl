

module Sepsis
export labeled_categorical, get_reward, random_initial_state, get_next_state, simulate_episode, get_parameters, sepsis_model, hr_probs, bp_probs, o2_probs, glu_probs

using Gen
using ..SepsisTypes
using LinearAlgebra


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

softmax(vector) = exp.(vector) / sum(exp.(vector))
add_0(vector) = [vector; 0]

function hr_probs(parameters::Parameters, state::State, action::Action)
    hr_probs = softmax(add_0(
        parameters[(:hr, state.hr, :intercept)]
        .+
        parameters[(:hr, state.hr, :abx)] .* Int(action.abx)
        .+
        parameters[(:hr, state.hr, :vaso)] .* Int(action.vaso)
    ))
    return hr_probs
end

function bp_probs(parameters::Parameters, state::State, action::Action)
    bp_probs = softmax(add_0(
        parameters[(:bp, state.bp, :intercept)]
        .+
        parameters[(:bp, state.bp, :abx)] .* Int(action.abx)
        .+
        parameters[(:bp, state.bp, :vaso)] .* Int(action.vaso)
    ))
    return bp_probs
end

function o2_probs(parameters::Parameters, state::State, action::Action)
    o2_probs = softmax(add_0(
        parameters[(:o2, state.o2, :intercept)]
        .+
        parameters[(:o2, state.o2, :vent)] .* Int(action.vent)
    ))
    return o2_probs
end

function glu_probs(parameters::Parameters, state::State, action::Action)
    glu_probs = softmax(add_0(
        parameters[(:glu, state.glu, :intercept)]
        .+
        parameters[(:glu, state.glu, :vaso)] .* Int(action.vaso)
    ))
    return glu_probs
end

@gen function get_next_state(params::Parameters, state::State, action::Action)::State
    hr ~ labeled_categorical(HR_LEVELS, hr_probs(params, state, action))
    bp ~ labeled_categorical(BP_LEVELS, bp_probs(params, state, action))
    o2 ~ labeled_categorical(O2_LEVELS, o2_probs(params, state, action))
    glu ~ labeled_categorical(GLU_LEVELS, glu_probs(params, state, action))
    next_state = State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)
    return next_state
end

@gen function simulate_episode(beliefs::Parameters, policy::Policy, start_state::State)
    states = [start_state]
    state = start_state
    rewards = []
    for t in 1:10
        new_state = {:trajectory => t} ~ get_next_state(beliefs, state, policy[state])
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

@gen function get_parameters()::Parameters

    parameters = Parameters()

    for level in HR_LEVELS
        parameters[(:hr, level, :intercept)] =
            {:hr => level => :intercept} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
        parameters[(:hr, level, :abx)] =
            {:hr => level => :abx} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
        parameters[(:hr, level, :vaso)] =
            {:hr => level => :vaso} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
    end

    for level in BP_LEVELS
        parameters[(:bp, level, :intercept)] =
            {:bp => level => :intercept} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
        parameters[(:bp, level, :abx)] =
            {:bp => level => :abx} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
        parameters[(:bp, level, :vaso)] =
            {:bp => level => :vaso} ~ mvnormal([0, 0], Diagonal([1.0, 1.0]))
    end

    for level in O2_LEVELS
        parameters[(:o2, level, :intercept)] =
            {:o2 => level => :intercept} ~ mvnormal([0], Diagonal([1.0]))
        parameters[(:o2, level, :vent)] =
            {:o2 => level => :vent} ~ mvnormal([0], Diagonal([1.0]))
    end

    for level in GLU_LEVELS
        parameters[(:glu, level, :intercept)] =
            {:glu => level => :intercept} ~ mvnormal([0, 0, 0, 0], Diagonal([1.0, 1.0, 1.0, 1.0]))
        parameters[(:glu, level, :vaso)] =
            {:glu => level => :vaso} ~ mvnormal([0, 0, 0, 0], Diagonal([1.0, 1.0, 1.0, 1.0]))
    end

    return parameters
end

@gen function sepsis_model(policies::Vector{Policy}, start_states::Vector{State})
    parameters = {:parameters} ~ get_parameters()
    episodes = []
    for (i, policy) in enumerate(policies)
        start_state = start_states[i]
        episode = {:episodes => i} ~ simulate_episode(parameteres, policy, start_state)
        push!(episodes, episode)
    end
    return parameters
end


end

