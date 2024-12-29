module ValueIteration
export optimize

using Gen
using ..SepsisTypes
using ..Sepsis
using LinearAlgebra
using PyCall
sepsis_gym = pyimport("custom_sepsis")

STATES = [to_state(state) for state in sepsis_gym.STATES]
n_states = length(STATES)
ACTIONS = [to_action(action) for action in sepsis_gym.ACTIONS]
n_actions = length(ACTIONS)
REWARDS = [get_reward(state) for state in STATES, action in ACTIONS]
"""
REWARDS (Matrix{Float64}): Reward matrix of shape (n_states, n_actions).
STATES (Vector): List of states.
ACTIONS (Vector): List of actions.
"""

function get_transition_matrix(parameters::Dict, functions::SepsisParams)
    n_states = length(STATES)
    n_actions = length(ACTIONS)
    transition_matrix = zeros(n_states, n_actions, n_states)
    for (i, state) in enumerate(STATES)
        for (j, action) in enumerate(ACTIONS)
            hr_p = functions.hr_probs(parameters, state, action)
            hr_p = Dict(
                LOW => hr_p[1],
                NORMAL => hr_p[2],
                HIGH => hr_p[3]
            )
            bp_p = functions.bp_probs(parameters, state, action)
            bp_p = Dict(
                LOW => bp_p[1],
                NORMAL => bp_p[2],
                HIGH => bp_p[3]
            )
            o2_p = functions.o2_probs(parameters, state, action)
            o2_p = Dict(
                LOW => o2_p[1],
                NORMAL => o2_p[2],
            )
            glu_p = functions.glu_probs(parameters, state, action)
            glu_p = Dict(
                SUPER_LOW => glu_p[1],
                LOW => glu_p[2],
                NORMAL => glu_p[3],
                HIGH => glu_p[4],
                SUPER_HIGH => glu_p[5]
            )
            diab_p = Dict(true => Int(state.diabetic), false => Int(!state.diabetic))
            abx_p = Dict(true => Int(action.abx), false => Int(!action.abx))
            vaso_p = Dict(true => Int(action.vaso), false => Int(!action.vaso))
            vent_p = Dict(true => Int(action.vent), false => Int(!action.vent))

            for (k, next_state) in enumerate(STATES)
                transition_matrix[i, j, k] = hr_p[next_state.hr] * bp_p[next_state.bp] * o2_p[next_state.o2] * glu_p[next_state.glu] * diab_p[next_state.diabetic] * abx_p[next_state.abx] * vaso_p[next_state.vaso] * vent_p[next_state.vent]
            end
        end
    end
    return transition_matrix
end

function value_iteration(transition_model, prev_V=zeros(n_states), gamma=0.99, theta=1e-5)
    """
    Optimized Value Iteration using matrix operations.

    Args:
        transition_model (Array{Float64, 3}): A 3D array of shape (n_states, n_actions, n_states) representing 
                                              transition probabilities.
        gamma (Float64): Discount factor.
        theta (Float64): Convergence threshold.

    Returns:
        policy (Array{Int64, 1}): Optimal policy as a 1D array of shape (n_states,).
        V (Array{Float64, 1}): Optimal value function as a 1D array of shape (n_states,).
    """
    # Initialize value function
    V = prev_V
    Q = nothing

    while true
        # Compute Q-values for all state-action pairs
        Q = REWARDS .+ gamma .* sum(transition_model .* reshape(V, 1, 1, n_states), dims=3)

        # Perform Bellman update
        new_V = maximum(Q, dims=2)  # Take the maximum value over actions
        delta = maximum(abs.(new_V .- V))  # Measure the largest change

        # Update value function
        V = new_V

        # Stop if converged
        if delta < theta
            break
        end
    end

    # Derive policy: the action that maximizes Q-value for each state
    pol = argmax(Q, dims=2)
    policy = Dict(STATES[i] => ACTIONS[pol[i][2]] for i in 1:n_states)
    return policy, V
end

function optimize(parameters::Dict, functions::SepsisParams, prev_V=zeros(n_states))
    transition_model = get_transition_matrix(parameters, functions)
    policy, V = value_iteration(transition_model, prev_V)
    return policy, V
end

end
