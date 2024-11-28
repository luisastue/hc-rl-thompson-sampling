

module Sepsis
export labeled_categorical, get_reward, random_initial_state, get_next_state, simulate_episode, get_beliefs, sepsis_model

using Gen
using ..SepsisTypes


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
        if state.hr == HIGH
            hr = @trace(bernoulli(env_params.abx_on_hr_H_N), :abx_hr) ? NORMAL : hr
        end
        if state.bp == HIGH
            bp = @trace(bernoulli(env_params.abx_on_bp_H_N), :abx_bp) ? NORMAL : bp
        end
    elseif !action.abx && state.abx  # withdrawn
        if state.hr == NORMAL
            hr = @trace(bernoulli(env_params.abx_withdrawn_hr_N_H), :abx_withdrawn_hr) ? HIGH : hr
        end
        if state.bp == NORMAL
            bp = @trace(bernoulli(env_params.abx_withdrawn_bp_N_H), :abx_withdrawn_bp) ? HIGH : bp
        end
    end

    # Ventilation -----------------------------
    if action.vent
        if state.o2 == LOW
            o2 = @trace(bernoulli(env_params.vent_on_o2_L_N), :vent_o2) ? NORMAL : o2
        end
    elseif !action.vent && state.vent  # withdrawn
        if state.o2 == NORMAL
            o2 = @trace(bernoulli(env_params.vent_withdrawn_o2_N_L), :vent_withdrawn_o2) ? LOW : o2
        end
    end

    # Vasopressors ----------------------------
    if !state.diabetic
        if action.vaso
            # blood pressure ---------------------
            if state.bp == LOW
                bp = @trace(bernoulli(env_params.nond_vaso_on_bp_L_N), :nond_vaso_bp_L_N) ? NORMAL : bp
            end
            if state.bp == NORMAL
                bp = @trace(bernoulli(env_params.nond_vaso_on_bp_N_H), :nond_vaso_bp_N_H) ? HIGH : bp
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == NORMAL
                bp = @trace(bernoulli(env_params.nond_vaso_withdrawn_bp_N_L), :nond_vaso_withdrawn_bp_N_L) ? LOW : bp
            end
            if state.bp == HIGH
                bp = @trace(bernoulli(env_params.nond_vaso_withdrawn_bp_H_N), :nond_vaso_withdrawn_bp_H_N) ? NORMAL : bp
            end
        end
    else
        if action.vaso
            # blood pressure ---------------------
            if state.bp == LOW
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_L_N), :diab_vaso_bp_L_N) ? NORMAL : bp
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_L_H), :diab_vaso_bp_L_H) ? HIGH : bp
            end
            if state.bp == NORMAL
                bp = @trace(bernoulli(env_params.diab_vaso_on_bp_N_H), :diab_vaso_bp_N_H) ? HIGH : bp
            end
            # glucose -----------------------------
            if state.glu == SUPER_LOW
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_SUPER_LOW) ? LOW : glu
            end
            if state.glu == LOW
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_LOW) ? NORMAL : glu
            end
            if state.glu == NORMAL
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_NORMAL) ? HIGH : glu
            end
            if state.glu == HIGH
                glu = @trace(bernoulli(env_params.diab_vaso_on_glu_up), :diab_vaso_glu_HIGH) ? SUPER_HIGH : glu
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == NORMAL
                bp = @trace(bernoulli(env_params.diab_vaso_withdrawn_bp_N_L), :diab_vaso_withdrawn_bp_N_L) ? LOW : bp
            end
            if state.bp == HIGH
                bp = @trace(bernoulli(env_params.diab_vaso_withdrawn_bp_H_N), :diab_vaso_withdrawn_bp_H_N) ? NORMAL : bp
            end
        end
    end

    # Fluctuations ----------------------------
    # random fluctuations only if no change in treatment
    if state.abx == action.abx && @trace(bernoulli(env_params.fluct), :fluct_hr)
        # heart rate is only affected by antibiotics
        hr = Level(clamp(Int(state.hr) + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_hr_change), -1, 1))
    end
    if state.abx == action.abx && state.vaso == action.vaso && @trace(bernoulli(env_params.fluct), :fluct_bp)
        # blood pressure is affected by antibiotics and vasopressors --> both need to stay the same for fluctuation
        bp = Level(clamp(Int(state.bp) + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_bp_change), -1, 1))
    end
    if state.vent == action.vent && @trace(bernoulli(env_params.fluct), :fluct_o2)
        # oxygen is only affected by ventilation
        o2 = Level(clamp(Int(state.o2) + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_o2_change), -1, 0))
    end
    glu_prob = state.diabetic ? env_params.diab_fluct_glu : env_params.fluct
    if state.vaso == action.vaso && @trace(bernoulli(glu_prob), :fluct_glu)
        # glucose is only affected by vasopressors
        glu = Level(clamp(Int(state.glu) + @trace(labeled_categorical([-1, 1], [0.5, 0.5]), :fluct_glu_change), -2, 2))
    end

    return State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)
end


@gen function simulate_episode(beliefs::EnvParameters, policy::Policy, start_state::State)
    states = [start_state]
    state = start_state
    rewards = []
    for t in 1:10
        new_state = {:action => t} ~ get_next_state(beliefs, state, policy[state])
        new_state = {:new_state => t} ~ labeled_categorical([new_state], [1])
        push!(states, new_state)
        reward = {:reward => t} ~ labeled_categorical([get_reward(new_state)], [1])
        if reward != 0.0
            break
        end
        push!(rewards, reward)
        state = new_state
    end
    return states, rewards
end

@gen function get_beliefs()::EnvParameters
    abx_on_hr_H_N = @trace(beta(50, 50), :abx_on_hr_H_N) # true is 0.5
    abx_on_bp_H_N = @trace(beta(50, 50), :abx_on_bp_H_N) # true is 0.5
    abx_withdrawn_hr_N_H = @trace(beta(10, 90), :abx_withdrawn_hr_N_H) # true is 0.1
    abx_withdrawn_bp_N_H = @trace(beta(50, 50), :abx_withdrawn_bp_N_H) # true is 0.5
    vent_on_o2_L_N = @trace(beta(70, 30), :vent_on_o2_L_N) # true is 0.7
    vent_withdrawn_o2_N_L = @trace(beta(10, 90), :vent_withdrawn_o2_N_L) # true is 0.1
    nond_vaso_on_bp_L_N = @trace(beta(70, 30), :nond_vaso_on_bp_L_N) # true is 0.7
    nond_vaso_on_bp_N_H = @trace(beta(70, 30), :nond_vaso_on_bp_N_H) # true is 0.7
    nond_vaso_withdrawn_bp_N_L = @trace(beta(50, 50), :nond_vaso_withdrawn_bp_N_L) # true is 0.5
    nond_vaso_withdrawn_bp_H_N = @trace(beta(40, 60), :nond_vaso_withdrawn_bp_H_N) # true is 0.4
    diab_vaso_on_bp_L_N = @trace(beta(90, 10), :diab_vaso_on_bp_L_N) # true is 0.9
    diab_vaso_on_bp_L_H = @trace(beta(50, 50), :diab_vaso_on_bp_L_H) # true is 0.5
    diab_vaso_on_bp_N_H = @trace(beta(10, 90), :diab_vaso_on_bp_N_H) # true is 0.1
    diab_vaso_on_glu_up = @trace(beta(10, 90), :diab_vaso_on_glu_up) # true is 0.1
    diab_vaso_withdrawn_bp_N_L = @trace(beta(5, 95), :diab_vaso_withdrawn_bp_N_L) # true is 0.05
    diab_vaso_withdrawn_bp_H_N = @trace(beta(5, 95), :diab_vaso_withdrawn_bp_H_N) # true is 0.05
    fluct = @trace(beta(10, 90), :fluct) # true is 0.1
    diab_fluct_glu = @trace(beta(30, 70), :diab_fluct_glu) # true is 0.3

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
        episode = {:episode => i} ~ simulate_episode(beliefs, policy, start_state)
        push!(episodes, episode)
    end
    return beliefs
end


end

