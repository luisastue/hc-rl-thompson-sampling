

module Sepsis
export labeled_categorical, get_reward, random_policy, random_initial_state, get_next_state, simulate_episode, get_beliefs, sepsis_model

using Gen
include("sepsis_types.jl")
using .SepsisTypes


@dist function labeled_categorical(labels, probs)
    index = categorical(probs)
    labels[index]
end

function get_reward(state::State)::Float64
    reward = 0.0
    critical_counts = count(c -> c != 0, [state.hr, state.bp, state.o2, state.glu])
    if critical_counts >= 3
        reward = -1.0
    elseif critical_counts == 0 && !state.abx && !state.vaso && !state.vent
        reward = 1.0
    end
    return reward
end

function random_policy()::Policy
    policy = Dict{State,Action}()
    for state in STATES
        policy[state] = rand(ACTIONS)
    end
    return policy
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

@gen function get_next_state(env_params::EnvParameters, state::State, action::Action)::State
    hr = state.hr
    bp = state.bp
    o2 = state.o2
    glu = state.glu

    # Antibiotics -----------------------------
    if action.abx
        if state.hr == Level.HIGH
            hr = @trace(bernoulli(env_params.abx_on_hr_H_N), :abx_hr) ? Level.NORMAL : hr
        end
        if state.bp == Level.HIGH
            bp = @trace(bernoulli(env_params.abx_on_bp_H_N), :abx_bp) ? Level.NORMAL : bp
        end
    elseif !action.abx && state.abx  # withdrawn
        if state.hr == Level.NORMAL
            hr = @trace(bernoulli(env_params.abx_withdrawn_hr_N_H), :abx_withdrawn_hr) ? Level.HIGH : hr
        end
        if state.bp == Level.NORMAL
            bp = @trace(bernoulli(env_params.abx_withdrawn_bp_N_H), :abx_withdrawn_bp) ? Level.HIGH : bp
        end
    end

    # Ventilation -----------------------------
    if action.vent
        if state.o2 == Level.LOW
            o2 = @trace(bernoulli(env_params.vent_on_o2_L_N), :vent_o2) ? Level.NORMAL : o2
        end
    elseif !action.vent && state.vent  # withdrawn
        if state.o2 == Level.NORMAL
            o2 = @trace(bernoulli(env_params.vent_withdrawn_o2_N_L), :vent_withdrawn_o2) ? Level.LOW : o2
        end
    end

    # Vasopressors ----------------------------
    if !state.diabetic
        if action.vaso
            # blood pressure ---------------------
            if state.bp == Level.LOW
                bp = @trace(bernoulli(env_params.nond_vaso_on_bp_L_N), :nond_vaso_bp_L_N) ? Level.NORMAL : bp
            end
            if state.bp == Level.NORMAL
                bp = @trace(bernoulli(env_params.nond_vaso_on_bp_N_H), :nond_vaso_bp_N_H) ? Level.HIGH : bp
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == Level.NORMAL
                bp = @trace(bernoulli(env_params.nond_vaso_withdrawn_bp_N_L), :nond_vaso_withdrawn_bp_N_L) ? Level.LOW : bp
            end
            if state.bp == Level.HIGH
                bp = @trace(bernoulli(env_params.nond_vaso_withdrawn_bp_H_N), :nond_vaso_withdrawn_bp_H_N) ? Level.NORMAL : bp
            end
        end
    else
        if action.vaso
            # blood pressure ---------------------
            if state.bp == Level.LOW
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_L_N), :diab_vaso_bp_L_N) ? Level.NORMAL : bp
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_L_H), :diab_vaso_bp_L_H) ? Level.HIGH : bp
            end
            if state.bp == Level.NORMAL
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_N_H), :diab_vaso_bp_N_H) ? Level.HIGH : bp
            end
            # glucose -----------------------------
            if state.glu == Level.SUPER_LOW
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_SUPER_LOW) ? Level.LOW : glu
            end
            if state.glu == Level.LOW
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_LOW) ? Level.NORMAL : glu
            end
            if state.glu == Level.NORMAL
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_NORMAL) ? Level.HIGH : glu
            end
            if state.glu == Level.HIGH
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_HIGH) ? Level.SUPER_HIGH : glu
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == Level.NORMAL
                bp = @trace(bernoulli(env_params.diab_vaso_withdrawn_bp_N_L), :diab_vaso_withdrawn_bp_N_L) ? Level.LOW : bp
            end
            if state.bp == Level.HIGH
                bp = @trace(bernoulli(env_params.diab_vaso_withdrawn_bp_H_N), :diab_vaso_withdrawn_bp_H_N) ? Level.NORMAL : bp
            end
        end
    end

    # Fluctuations ----------------------------
    # random fluctuations only if no change in treatment
    if state.abx == action.abx && @trace(bernoulli(env_params.fluct), :fluct_hr)
        # heart rate is only affected by antibiotics
        hr = Level(clamp(state.hr.value + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_hr_change), -1, 1))
    end
    if state.abx == action.abx && state.vaso == action.vaso && @trace(bernoulli(env_params.fluct), :fluct_bp)
        # blood pressure is affected by antibiotics and vasopressors --> both need to stay the same for fluctuation
        bp = Level(clamp(state.bp.value + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_bp_change), -1, 1))
    end
    if state.vent == action.vent && @trace(bernoulli(env_params.fluct), :fluct_o2)
        # oxygen is only affected by ventilation
        o2 = Level(clamp(state.o2.value + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_o2_change), -1, 0))
    end
    glu_prob = state.diabetic ? env_params.diab_fluct_glu : env_params.fluct
    if state.vaso == action.vaso && @trace(bernoulli(glu_prob), :fluct_glu)
        # glucose is only affected by vasopressors
        glu = Level(clamp(state.glu.value + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_glu_change), -2, 2))
    end

    return State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)
end



@gen function simulate_episode(beliefs::EnvParameters, actions::Vector{Action}, start_state::State)
    states = [start_state]
    state = start_state
    rewards = []
    for (t, action) in enumerate(actions)
        new_state = {t} ~ get_next_state(beliefs, state, action)
        new_state = {:new_state => t} ~ labeled_categorical([new_state], [1])
        push!(states, new_state)
        reward = {:reward => t} ~ labeled_categorical([get_reward(new_state)], [1])
        push!(rewards, reward)
        state = new_state
    end
    return states, rewards
end

@gen function get_beliefs()::EnvParameters
    abx_on_hr_H_N = @trace(beta(1, 1), :abx_on_hr_H_N)
    abx_on_bp_H_N = @trace(beta(1, 1), :abx_on_bp_H_N)
    abx_withdrawn_hr_N_H = @trace(beta(1, 1), :abx_withdrawn_hr_N_H)
    abx_withdrawn_bp_N_H = @trace(beta(1, 1), :abx_withdrawn_bp_N_H)
    vent_on_o2_L_N = @trace(beta(1, 1), :vent_on_o2_L_N)
    vent_withdrawn_o2_N_L = @trace(beta(1, 1), :vent_withdrawn_o2_N_L)
    nond_vaso_on_bp_L_N = @trace(beta(1, 1), :nond_vaso_on_bp_L_N)
    nond_vaso_on_bp_N_H = @trace(beta(1, 1), :nond_vaso_on_bp_N_H)
    nond_vaso_withdrawn_bp_N_L = @trace(beta(1, 1), :nond_vaso_withdrawn_bp_N_L)
    nond_vaso_withdrawn_bp_H_N = @trace(beta(1, 1), :nond_vaso_withdrawn_bp_H_N)
    diab_vaso_on_bp_L_N = @trace(beta(1, 1), :diab_vaso_on_bp_L_N)
    diab_vaso_on_bp_L_H = @trace(beta(1, 1), :diab_vaso_on_bp_L_H)
    diab_vaso_on_bp_N_H = @trace(beta(1, 1), :diab_vaso_on_bp_N_H)
    diab_vaso_on_glu_up = @trace(beta(1, 1), :diab_vaso_on_glu_up)
    diab_vaso_withdrawn_bp_N_L = @trace(beta(1, 1), :diab_vaso_withdrawn_bp_N_L)
    diab_vaso_withdrawn_bp_H_N = @trace(beta(1, 1), :diab_vaso_withdrawn_bp_H_N)
    fluct = @trace(beta(1, 1), :fluct)
    diab_fluct_glu = @trace(beta(1, 1), :diab_fluct_glu)

    return EnvParameters(
        abx_on_hr_H_N,
        abx_on_bp_H_N,
        abx_withdrawn_hr_N_H,
        abx_withdrawn_bp_N_H,
        vent_on_o2_L_N,
        vent_withdrawn_o2_N_L,
        nond_vaso_on_bp_L_N,
        nond_vaso_on_bp_N_H,
        nond_vaso_withdrawn_bp_N_L,
        nond_vaso_withdrawn_bp_H_N,
        diab_vaso_on_bp_L_N,
        diab_vaso_on_bp_L_H,
        diab_vaso_on_bp_N_H,
        diab_vaso_on_glu_up,
        diab_vaso_withdrawn_bp_N_L,
        diab_vaso_withdrawn_bp_H_N,
        fluct,
        diab_fluct_glu
    )

end

@gen function sepsis_model(policies::Vector{Policy}, start_states::Vector{State})
    beliefs = {:beliefs} ~ get_beliefs()
    episodes = []
    for (i, policy) in enumerate(policies)
        start_state = start_states[i]
        episode = {i => :episode} ~ simulate_episode(beliefs, policy, start_state)
        push!(episodes, episode)
    end
    return beliefs
end
end

