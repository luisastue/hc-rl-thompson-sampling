

module Sepsis
export labeled_categorical, get_reward, random_initial_state, get_next_state, simulate_episode, get_beliefs, sepsis_model

using Gen
using ..SepsisTypes


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

@gen function get_next_state(env_params::EnvParameters, state::State, action::Action)::State
    hr = state.hr
    bp = state.bp
    o2 = state.o2
    glu = state.glu
    abx_hr ~ bernoulli(env_params.abx_on_hr_H_N)
    abx_bp ~ bernoulli(env_params.abx_on_bp_H_N)
    abx_withdrawn_hr ~ bernoulli(env_params.abx_withdrawn_hr_N_H)
    abx_withdrawn_bp ~ bernoulli(env_params.abx_withdrawn_bp_N_H)
    vent_o2 ~ bernoulli(env_params.vent_on_o2_L_N)
    vent_withdrawn_o2 ~ bernoulli(env_params.vent_withdrawn_o2_N_L)
    nond_vaso_bp_L_N ~ bernoulli(env_params.nond_vaso_on_bp_L_N)
    nond_vaso_bp_N_H ~ bernoulli(env_params.nond_vaso_on_bp_N_H)
    nond_vaso_withdrawn_bp_N_L ~ bernoulli(env_params.nond_vaso_withdrawn_bp_N_L)
    nond_vaso_withdrawn_bp_H_N ~ bernoulli(env_params.nond_vaso_withdrawn_bp_H_N)
    diab_vaso_bp_L_N ~ bernoulli(env_params.diab_vaso_on_bp_L_N)
    diab_vaso_bp_L_H ~ bernoulli(env_params.diab_vaso_on_bp_L_H)
    diab_vaso_bp_N_H ~ bernoulli(env_params.diab_vaso_on_bp_N_H)
    diab_vaso_glu_up ~ bernoulli(env_params.diab_vaso_on_glu_up)
    diab_vaso_withdrawn_bp_N_L ~ bernoulli(env_params.diab_vaso_withdrawn_bp_N_L)
    diab_vaso_withdrawn_bp_H_N ~ bernoulli(env_params.diab_vaso_withdrawn_bp_H_N)
    fluct_hr ~ bernoulli(env_params.fluct)
    fluct_bp ~ bernoulli(env_params.fluct)
    fluct_o2 ~ bernoulli(env_params.fluct)
    fluct_glu ~ bernoulli(state.diabetic ? env_params.diab_fluct_glu : env_params.fluct)
    fluct_hr_change ~ labeled_categorical([-1, 1], [0.5, 0.5])
    fluct_bp_change ~ labeled_categorical([-1, 1], [0.5, 0.5])
    fluct_o2_change ~ labeled_categorical([-1, 1], [0.5, 0.5])
    fluct_glu_change ~ labeled_categorical([-1, 1], [0.5, 0.5])

    # Antibiotics -----------------------------
    if action.abx
        if state.hr == HIGH
            hr = abx_hr ? NORMAL : hr
        end
        if state.bp == HIGH
            bp = abx_bp ? NORMAL : bp
        end
    elseif !action.abx && state.abx  # withdrawn
        if state.hr == NORMAL
            hr = abx_withdrawn_hr ? HIGH : hr
        end
        if state.bp == NORMAL
            bp = abx_withdrawn_bp ? HIGH : bp
        end
    end

    # Ventilation -----------------------------
    if action.vent
        if state.o2 == LOW
            o2 = vent_o2 ? NORMAL : o2
        end
    elseif !action.vent && state.vent  # withdrawn
        if state.o2 == NORMAL
            o2 = vent_withdrawn_o2 ? LOW : o2
        end
    end

    # Vasopressors ----------------------------
    if !state.diabetic
        if action.vaso
            # blood pressure ---------------------
            if state.bp == LOW
                bp = nond_vaso_bp_L_N ? NORMAL : bp
            end
            if state.bp == NORMAL
                bp = nond_vaso_bp_N_H ? HIGH : bp
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == NORMAL
                bp = nond_vaso_withdrawn_bp_N_L ? LOW : bp
            end
            if state.bp == HIGH
                bp = nond_vaso_withdrawn_bp_H_N ? NORMAL : bp
            end
        end
    else
        if action.vaso
            # blood pressure ---------------------
            if state.bp == LOW
                bp = diab_vaso_bp_L_N ? NORMAL : bp
                bp = diab_vaso_bp_L_H ? HIGH : bp
            end
            if state.bp == NORMAL
                bp = diab_vaso_bp_N_H ? HIGH : bp
            end
            # glucose -----------------------------
            if state.glu == SUPER_LOW
                glu = diab_vaso_glu_up ? LOW : glu
            end
            if state.glu == LOW
                glu = diab_vaso_glu_up ? NORMAL : glu
            end
            if state.glu == NORMAL
                glu = diab_vaso_glu_up ? HIGH : glu
            end
            if state.glu == HIGH
                glu = diab_vaso_glu_up ? SUPER_HIGH : glu
            end
        elseif !action.vaso && state.vaso  # withdrawn
            if state.bp == NORMAL
                bp = diab_vaso_withdrawn_bp_N_L ? LOW : bp
            end
            if state.bp == HIGH
                bp = diab_vaso_withdrawn_bp_H_N ? NORMAL : bp
            end
        end
    end

    # Fluctuations ----------------------------
    # random fluctuations only if no change in treatment
    if state.abx == action.abx && fluct_hr
        # heart rate is only affected by antibiotics
        hr = Level(clamp(Int(state.hr) + fluct_hr_change, -1, 1))
    end
    if state.abx == action.abx && state.vaso == action.vaso && fluct_bp
        # blood pressure is affected by antibiotics and vasopressors --> both need to stay the same for fluctuation
        bp = Level(clamp(Int(state.bp) + fluct_bp_change, -1, 1))
    end
    if state.vent == action.vent && fluct_o2
        # oxygen is only affected by ventilation
        o2 = Level(clamp(Int(state.o2) + fluct_o2_change, -1, 0))
    end
    if state.vaso == action.vaso && fluct_glu
        # glucose is only affected by vasopressors
        glu = Level(clamp(Int(state.glu) + fluct_glu_change, -2, 2))
    end
    next_state = State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)
    return next_state
end


@gen function simulate_episode(beliefs::EnvParameters, policy::Policy, start_state::State)
    states = [start_state]
    state = start_state
    rewards = []
    # done = false
    for t in 1:10
        new_state = {:get_next => t} ~ get_next_state(beliefs, state, policy[state])

        hr = {:new_state => t => :hr} ~ labeled_categorical([Int(new_state.hr)], [1])
        bp = {:new_state => t => :bp} ~ labeled_categorical([Int(new_state.bp)], [1])
        o2 = {:new_state => t => :o2} ~ labeled_categorical([Int(new_state.o2)], [1])
        glu = {:new_state => t => :glu} ~ labeled_categorical([Int(new_state.glu)], [1])
        diabetic = {:new_state => t => :diabetic} ~ labeled_categorical([Int(new_state.diabetic)], [1])
        abx = {:new_state => t => :abx} ~ labeled_categorical([Int(new_state.abx)], [1])
        vaso = {:new_state => t => :vaso} ~ labeled_categorical([Int(new_state.vaso)], [1])
        vent = {:new_state => t => :vent} ~ labeled_categorical([Int(new_state.vent)], [1])

        push!(states, new_state)
        reward = get_reward(new_state)
        # reward = {:reward => t} ~ labeled_categorical([Int(get_reward(new_state))], [1])
        # if reward != 0
        #     done = true
        # end
        push!(rewards, reward)
        state = new_state
    end
    return states, rewards
end

@gen function get_beliefs()::EnvParameters
    abx_on_hr_H_N = @trace(beta(1, 1), :abx_on_hr_H_N) # true is 0.5 -> beta(50, 50)
    abx_on_bp_H_N = @trace(beta(1, 1), :abx_on_bp_H_N) # true is 0.5 -> beta(50, 50)
    abx_withdrawn_hr_N_H = @trace(beta(1, 1), :abx_withdrawn_hr_N_H) # true is 0.1 -> beta(10, 90)
    abx_withdrawn_bp_N_H = @trace(beta(1, 1), :abx_withdrawn_bp_N_H) # true is 0.5 -> beta(50, 50)
    vent_on_o2_L_N = @trace(beta(1, 1), :vent_on_o2_L_N) # true is 0.7 -> beta(70, 30)
    vent_withdrawn_o2_N_L = @trace(beta(1, 1), :vent_withdrawn_o2_N_L) # true is 0.1 -> beta(10, 90)
    nond_vaso_on_bp_L_N = @trace(beta(1, 1), :nond_vaso_on_bp_L_N) # true is 0.7 -> beta(70, 30)
    nond_vaso_on_bp_N_H = @trace(beta(1, 1), :nond_vaso_on_bp_N_H) # true is 0.7 -> beta(70, 30)
    nond_vaso_withdrawn_bp_N_L = @trace(beta(1, 1), :nond_vaso_withdrawn_bp_N_L) # true is 0.5 -> beta(50, 50)
    nond_vaso_withdrawn_bp_H_N = @trace(beta(1, 1), :nond_vaso_withdrawn_bp_H_N) # true is 0.4 -> beta(40, 60)
    diab_vaso_on_bp_L_N = @trace(beta(1, 1), :diab_vaso_on_bp_L_N) # true is 0.9 -> beta(90, 10)
    diab_vaso_on_bp_L_H = @trace(beta(1, 1), :diab_vaso_on_bp_L_H) # true is 0.5 -> beta(50, 50)
    diab_vaso_on_bp_N_H = @trace(beta(1, 1), :diab_vaso_on_bp_N_H) # true is 0.1 -> beta(10, 90)
    diab_vaso_on_glu_up = @trace(beta(1, 1), :diab_vaso_on_glu_up) # true is 0.1 -> beta(10, 90)
    diab_vaso_withdrawn_bp_N_L = @trace(beta(1, 1), :diab_vaso_withdrawn_bp_N_L) # true is 0.05 -> beta(5, 95)
    diab_vaso_withdrawn_bp_H_N = @trace(beta(1, 1), :diab_vaso_withdrawn_bp_H_N) # true is 0.05 -> beta(5, 95)
    fluct = @trace(beta(1, 1), :fluct) # true is 0.1 -> beta(10, 90)
    diab_fluct_glu = @trace(beta(1, 1), :diab_fluct_glu) # true is 0.3 -> beta(30, 70)

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

