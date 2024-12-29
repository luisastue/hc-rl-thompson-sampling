

module Simple
export simple_functions, block_update_simple

using Gen
using ..SepsisTypes
using ..Sepsis

const Params = Dict{Union{Pair{Symbol,Symbol},Symbol},Float64}

function add_leak(leak::Float64, probs::Vector{Float64})
    return (probs .+ leak) / sum(probs .+ leak)
end

# assumes that antibiotics have a lowering effect on the heart rate. 
function hr_probs(parameters::Params, state::State, action::Action)
    leak = parameters[:fluctuation_leakage]
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
        return add_leak(leak, [1 - hr_fluctuation, hr_fluctuation, 0.0])
    elseif state.hr == NORMAL
        if behavior == :abx_withdrawn
            return add_leak(leak, [0.0, 1 - abx_withdrawal_effect, abx_withdrawal_effect])
        else
            return add_leak(leak, [hr_fluctuation, 1 - 2 * hr_fluctuation, hr_fluctuation])
        end
    elseif state.hr == HIGH
        if behavior == :abx_on
            return add_leak(leak, [0.0, abx_effect, 1 - abx_effect])
        else
            return add_leak(leak, [0.0, hr_fluctuation, 1 - hr_fluctuation])
        end
    end
end

# assumes that antibiotics have a lowering effect on the blood pressure. 
# assumes that vasopressors have a raising effect on the blood pressure. 
# priority: antibiotics_on, vasopressors_on, antibiotics_withdrawn, vasopressors_withdrawn, fluctuation
function bp_probs(parameters::Params, state::State, action::Action)
    leak = parameters[:fluctuation_leakage]
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
            return add_leak(leak, [1 - vaso_effect, vaso_effect, 0.0])
        else
            return add_leak(leak, [1 - bp_fluctuation, bp_fluctuation, 0.0])
        end
    elseif state.bp == NORMAL
        if vaso_behavior == :vaso_on
            return add_leak(leak, [0.0, 1 - vaso_effect, vaso_effect])
        elseif abx_behavior == :abx_withdrawn
            return add_leak(leak, [0.0, 1 - abx_withdrawal_effect, abx_withdrawal_effect])
        elseif vaso_behavior == :vaso_withdrawn
            return add_leak(leak, [vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0])
        else
            return add_leak(leak, [bp_fluctuation, 1 - 2 * bp_fluctuation, bp_fluctuation])
        end
    elseif state.bp == HIGH
        if abx_behavior == :abx_on
            return add_leak(leak, [0.0, abx_effect, 1 - abx_effect])
        elseif vaso_behavior == :vaso_withdrawn
            return add_leak(leak, [0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect])
        else
            return add_leak(leak, [0.0, bp_fluctuation, 1 - bp_fluctuation])
        end
    end
end

# assumes that ventilation has a raising effect on the oxygen level.
function o2_probs(parameters::Params, state::State, action::Action)
    leak = parameters[:fluctuation_leakage]
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
            return add_leak(leak, [1 - vent_effect, vent_effect])
        else
            return add_leak(leak, [1 - o2_fluctuation, o2_fluctuation])
        end
    elseif state.o2 == NORMAL
        if behavior == :vent_withdrawn
            return add_leak(leak, [vent_withdrawal_effect, 1 - vent_withdrawal_effect])
        else
            return add_leak(leak, [o2_fluctuation, 1 - o2_fluctuation])
        end
    end
end

# assumes that vasopressors have a raising effect on the glucose level.
function glu_probs(parameters::Params, state::State, action::Action)
    leak = parameters[:fluctuation_leakage]
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
            return add_leak(leak, [1 - vaso_effect, vaso_effect, 0.0, 0.0, 0.0])
        else
            return add_leak(leak, [1 - glu_fluctuation, glu_fluctuation, 0.0, 0.0, 0.0])
        end
    elseif state.glu == LOW
        if behavior == :vaso_on
            return add_leak(leak, [0.0, 1 - vaso_effect, vaso_effect, 0.0, 0.0])
        elseif behavior == :vaso_withdrawn
            return add_leak(leak, [vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0, 0.0, 0.0])
        else
            return add_leak(leak, [glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation, 0.0, 0.0])
        end
    elseif state.glu == NORMAL
        if behavior == :vaso_on
            return add_leak(leak, [0.0, 0.0, 1 - vaso_effect, vaso_effect, 0.0])
        elseif behavior == :vaso_withdrawn
            return add_leak(leak, [0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0, 0.0])
        else
            return add_leak(leak, [0.0, glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation, 0.0])
        end
    elseif state.glu == HIGH
        if behavior == :vaso_on
            return add_leak(leak, [0.0, 0.0, 0.0, vaso_effect, 1 - vaso_effect])
        elseif behavior == :vaso_withdrawn
            return add_leak(leak, [0.0, 0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect, 0.0])
        else
            return add_leak(leak, [0.0, 0.0, glu_fluctuation, 1 - 2 * glu_fluctuation, glu_fluctuation])
        end
    elseif state.glu == SUPER_HIGH
        if behavior == :vaso_withdrawn
            return add_leak(leak, [0.0, 0.0, 0.0, vaso_withdrawal_effect, 1 - vaso_withdrawal_effect])
        else
            return add_leak(leak, [0.0, 0.0, 0.0, glu_fluctuation, 1 - glu_fluctuation])
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
    parameters[:fluctuation_leakage] = {:fluctuation_leakage} ~ beta(1, 100)

    return parameters
end

function set_parameters(choices::ChoiceMap, params::Params)
    choices[:parameters=>:hr=>:fluctuation] = params[:hr=>:fluctuation]
    choices[:parameters=>:hr=>:abx_effect] = params[:hr=>:abx_effect]
    choices[:parameters=>:hr=>:abx_withdrawal_effect] = params[:hr=>:abx_withdrawal_effect]
    choices[:parameters=>:bp=>:fluctuation] = params[:bp=>:fluctuation]
    choices[:parameters=>:bp=>:abx_effect] = params[:bp=>:abx_effect]
    choices[:parameters=>:bp=>:abx_withdrawal_effect] = params[:bp=>:abx_withdrawal_effect]
    choices[:parameters=>:bp=>:vaso_effect] = params[:bp=>:vaso_effect]
    choices[:parameters=>:bp=>:vaso_withdrawal_effect] = params[:bp=>:vaso_withdrawal_effect]
    choices[:parameters=>:o2=>:fluctuation] = params[:o2=>:fluctuation]
    choices[:parameters=>:o2=>:vent_effect] = params[:o2=>:vent_effect]
    choices[:parameters=>:o2=>:vent_withdrawal_effect] = params[:o2=>:vent_withdrawal_effect]
    choices[:parameters=>:glu=>:fluctuation] = params[:glu=>:fluctuation]
    choices[:parameters=>:glu=>:vaso_effect] = params[:glu=>:vaso_effect]
    choices[:parameters=>:glu=>:vaso_withdrawal_effect] = params[:glu=>:vaso_withdrawal_effect]
    choices[:parameters=>:fluctuation_leakage] = params[:fluctuation_leakage]
    return choices
end

function update(trace, step_size)
    acceptance = 0
    (trace, a1) = mh(trace, select(:parameters => :hr => :fluctuation))
    (trace, a2) = mh(trace, select(:parameters => :hr => :abx_effect))
    (trace, a3) = mh(trace, select(:parameters => :hr => :abx_withdrawal_effect))
    (trace, a4) = mh(trace, select(:parameters => :bp => :fluctuation))
    (trace, a5) = mh(trace, select(:parameters => :bp => :abx_effect))
    (trace, a6) = mh(trace, select(:parameters => :bp => :abx_withdrawal_effect))
    (trace, a7) = mh(trace, select(:parameters => :bp => :vaso_effect))
    (trace, a8) = mh(trace, select(:parameters => :bp => :vaso_withdrawal_effect))
    (trace, a9) = mh(trace, select(:parameters => :o2 => :fluctuation))
    (trace, a10) = mh(trace, select(:parameters => :o2 => :vent_effect))
    (trace, a11) = mh(trace, select(:parameters => :o2 => :vent_withdrawal_effect))
    (trace, a12) = mh(trace, select(:parameters => :glu => :fluctuation))
    (trace, a13) = mh(trace, select(:parameters => :glu => :vaso_effect))
    (trace, a14) = mh(trace, select(:parameters => :glu => :vaso_withdrawal_effect))
    (trace, a15) = mh(trace, select(:fluctuation_leakage))
    acceptance += Int(a1) + Int(a2) + Int(a3) + Int(a4) + Int(a5) + Int(a6) + Int(a7) + Int(a8) + Int(a9) + Int(a10) + Int(a11) + Int(a12) + Int(a13) + Int(a14) + Int(a15)
    acceptance /= 15

    return trace, acceptance
end


const simple_functions = SepsisParams(
    get_parameters,
    set_parameters,
    hr_probs,
    bp_probs,
    o2_probs,
    glu_probs,
    update,
)

end