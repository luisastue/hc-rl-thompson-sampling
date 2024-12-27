

module Simple
export simple_functions, block_update_simple

using Gen
using ..SepsisTypes
using ..Sepsis

const Params = Dict{Pair{Symbol,Symbol},Float64}

# assumes that antibiotics have a lowering effect on the heart rate. 
function hr_probs(parameters::Params, state::State, action::Action)
    if action.abx
        behavior = :abx_on
    elseif state.abx
        behavior = :abx_withdrawn
    else
        behavior = :abx_off
    end

    # Extract parameters
    hr_fluctuation = parameters[:hr=>:fluctuation] / 2
    abx_effect = parameters[:hr=>:abx_effect]
    abx_withdrawal_effect = parameters[:hr=>:abx_withdrawal_effect]

    if state.hr == LOW
        return [1 - hr_fluctuation, hr_fluctuation, 0.0]
    elseif state.hr == NORMAL
        if behavior == :abx_withdrawn
            return [0.0, 1 - abx_withdrawal_effect, abx_withdrawal_effect]
        else
            return [hr_fluctuation, 1 - 2 * hr_fluctuation, hr_fluctuation]
        end
    elseif state.hr == HIGH
        if behavior == :abx_on
            return [0.0, abx_effect, 1 - abx_effect]
        else
            return [0.0, hr_fluctuation, 1 - hr_fluctuation]
        end
    end
end

# assumes that antibiotics have a lowering effect on the blood pressure. 
# assumes that vasopressors have a raising effect on the blood pressure. 
# priority: antibiotics_on, vasopressors_on, antibiotics_withdrawn, vasopressors_withdrawn, fluctuation
function bp_probs(parameters::Params, state::State, action::Action)
    if action.abx
        abx_behavior = :abx_on
    elseif state.abx
        abx_behavior = :abx_withdrawn
    else
        abx_behavior = :abx_off
    end
    if action.vaso
        vaso_behavior = :vaso_on
    elseif state.vaso
        vaso_behavior = :vaso_withdrawn
    else
        vaso_behavior = :vaso_off
    end

    # Extract parameters
    bp_fluctuation = parameters[:bp=>:fluctuation] / 2
    abx_effect = parameters[:bp=>:abx_effect]
    abx_withdrawal_effect = parameters[:bp=>:abx_withdrawal_effect]
    vaso_effect = parameters[:bp=>:vaso_effect]
    vaso_withdrawal_effect = parameters[:bp=>:vaso_withdrawal_effect]

    if state.bp == LOW
        if vaso_behavior == :vaso_on
            return [1 - vaso_effect, vaso_effect, 0.0]
        else
            return [1 - bp_fluctuation, bp_fluctuation, 0.0]
        end
    elseif state.bp == NORMAL
        if vaso_behavior == :vaso_on
            return [0.0, 1 - vaso_effect, vaso_effect]
        elseif abx_behavior == :abx_withdrawn
            return [0.0, 1 - abx_withdrawal_effect, abx_withdrawal_effect]
        elseif vaso_behavior == :vaso_withdrawn
            return [vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0]
        else
            return [bp_fluctuation, 1 - 2 * bp_fluctuation, bp_fluctuation]
        end
    elseif state.bp == HIGH
        if abx_behavior == :abx_on
            return [0.0, abx_effect, 1 - abx_effect]
        elseif vaso_behavior == :vaso_withdrawn
            return [0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect]
        else
            return [0.0, bp_fluctuation, 1 - bp_fluctuation]
        end
    end
end

# assumes that ventilation has a raising effect on the oxygen level.
function o2_probs(parameters::Params, state::State, action::Action)
    if action.vent
        behavior = :vent_on
    elseif state.vent
        behavior = :vent_withdrawn
    else
        behavior = :vent_off
    end

    # Extract parameters
    o2_fluctuation = parameters[:o2=>:fluctuation] / 2
    vent_effect = parameters[:o2=>:vent_effect]
    vent_withdrawal_effect = parameters[:o2=>:vent_withdrawal_effect]

    if state.o2 == LOW
        if behavior == :vent_on
            return [1 - vent_effect, vent_effect]
        else
            return [1 - o2_fluctuation, o2_fluctuation]
        end
    elseif state.o2 == NORMAL
        if behavior == :vent_withdrawn
            return [vent_withdrawal_effect, 1 - vent_withdrawal_effect]
        else
            return [o2_fluctuation, 1 - o2_fluctuation]
        end
    end
end

# assumes that vasopressors have a raising effect on the glucose level.
function glu_probs(parameters::Params, state::State, action::Action)
    if action.vaso
        behavior = :vaso_on
    elseif state.vaso
        behavior = :vaso_withdrawn
    else
        behavior = :vaso_off
    end

    # Extract parameters
    glu_fluctuation = parameters[:glu=>:fluctuation] / 2
    vaso_effect = parameters[:glu=>:vaso_effect]
    vaso_withdrawal_effect = parameters[:glu=>:vaso_withdrawal_effect]

    if state.glu == SUPER_LOW
        if behavior == :vaso_on
            return [1 - vaso_effect, vaso_effect, 0.0, 0.0, 0.0]
        else
            return [1 - glu_fluctuation, glu_fluctuation, 0.0, 0.0, 0.0]
        end
    elseif state.glu == LOW
        if behavior == :vaso_on
            return [0.0, 1 - vaso_effect, vaso_effect, 0.0, 0.0]
        elseif behavior == :vaso_withdrawn
            return [vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0, 0.0, 0.0]
        else
            return [glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation, 0.0, 0.0]
        end
    elseif state.glu == NORMAL
        if behavior == :vaso_on
            return [0.0, 0.0, 1 - vaso_effect, vaso_effect, 0.0]
        elseif behavior == :vaso_withdrawn
            return [0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0, 0.0]
        else
            return [0.0, glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation, 0.0]
        end
    elseif state.glu == HIGH
        if behavior == :vaso_on
            return [0.0, 0.0, 0.0, vaso_effect, 1 - vaso_effect]
        elseif behavior == :vaso_withdrawn
            return [0.0, 0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0]
        else
            return [0.0, 0.0, glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation]
        end
    elseif state.glu == SUPER_HIGH
        if behavior == :vaso_withdrawn
            return [0.0, 0.0, 0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect]
        else
            return [0.0, 0.0, 0.0, glu_fluctuation, 1 - glu_fluctuation]
        end
    end

end

@gen function get_parameters()::Params
    parameters = Params()

    parameters[:hr=>:fluctuation] = {:hr => :fluctuation} ~ beta(1, 1)
    parameters[:hr=>:abx_effect] = {:hr => :abx_effect} ~ beta(1, 1)
    parameters[:hr=>:abx_withdrawal_effect] = {:hr => :abx_withdrawal_effect} ~ beta(1, 1)
    parameters[:bp=>:fluctuation] = {:bp => :fluctuation} ~ beta(1, 1)
    parameters[:bp=>:abx_effect] = {:bp => :abx_effect} ~ beta(1, 1)
    parameters[:bp=>:abx_withdrawal_effect] = {:bp => :abx_withdrawal_effect} ~ beta(1, 1)
    parameters[:bp=>:vaso_effect] = {:bp => :vaso_effect} ~ beta(1, 1)
    parameters[:bp=>:vaso_withdrawal_effect] = {:bp => :vaso_withdrawal_effect} ~ beta(1, 1)
    parameters[:o2=>:fluctuation] = {:o2 => :fluctuation} ~ beta(1, 1)
    parameters[:o2=>:vent_effect] = {:o2 => :vent_effect} ~ beta(1, 1)
    parameters[:o2=>:vent_withdrawal_effect] = {:o2 => :vent_withdrawal_effect} ~ beta(1, 1)
    parameters[:glu=>:fluctuation] = {:glu => :fluctuation} ~ beta(1, 1)
    parameters[:glu=>:vaso_effect] = {:glu => :vaso_effect} ~ beta(1, 1)
    parameters[:glu=>:vaso_withdrawal_effect] = {:glu => :vaso_withdrawal_effect} ~ beta(1, 1)

    return parameters
end

function extract_parameters(trace)::Params
    return trace[:parameters]
end

const simple_functions = SepsisParams(
    get_parameters,
    hr_probs,
    bp_probs,
    o2_probs,
    glu_probs,
)

end