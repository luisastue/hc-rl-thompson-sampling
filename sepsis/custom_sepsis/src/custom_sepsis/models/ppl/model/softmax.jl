

module Softmax
export softmax_functions

using Gen
using ..SepsisTypes
using ..Sepsis
using LinearAlgebra

const Params = Dict{Tuple{Symbol,Level,Symbol},Vector{Float64}}


softmax(vector) = exp.(vector) / sum(exp.(vector))
add_0(vector) = [vector; 0]

function hr_probs(parameters::Params, state::State, action::Action)
    hr_probs = softmax(add_0(
        parameters[(:hr, state.hr, :intercept)]
        .+
        parameters[(:hr, state.hr, :abx)] .* Int(action.abx)
        .+
        parameters[(:hr, state.hr, :vaso)] .* Int(action.vaso)
    ))
    return hr_probs
end

function bp_probs(parameters::Params, state::State, action::Action)
    bp_probs = softmax(add_0(
        parameters[(:bp, state.bp, :intercept)]
        .+
        parameters[(:bp, state.bp, :abx)] .* Int(action.abx)
        .+
        parameters[(:bp, state.bp, :vaso)] .* Int(action.vaso)
    ))
    return bp_probs
end

function o2_probs(parameters::Params, state::State, action::Action)
    o2_probs = softmax(add_0(
        parameters[(:o2, state.o2, :intercept)]
        .+
        parameters[(:o2, state.o2, :vent)] .* Int(action.vent)
    ))
    return o2_probs
end

function glu_probs(parameters::Params, state::State, action::Action)
    glu_probs = softmax(add_0(
        parameters[(:glu, state.glu, :intercept)]
        .+
        parameters[(:glu, state.glu, :vaso)] .* Int(action.vaso)
    ))
    return glu_probs
end

@gen function get_parameters()::Params

    parameters = Params()

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

function set_parameters(choices::ChoiceMap, parameters::Params)
    for level in HR_LEVELS
        choices[:parameters=>:hr=>level=>:intercept] = parameters[(:hr, level, :intercept)]
        choices[:parameters=>:hr=>level=>:abx] = parameters[(:hr, level, :abx)]
        choices[:parameters=>:hr=>level=>:vaso] = parameters[(:hr, level, :vaso)]
    end
    for level in BP_LEVELS
        choices[:parameters=>:bp=>level=>:intercept] = parameters[(:bp, level, :intercept)]
        choices[:parameters=>:bp=>level=>:abx] = parameters[(:bp, level, :abx)]
        choices[:parameters=>:bp=>level=>:vaso] = parameters[(:bp, level, :vaso)]
    end
    for level in O2_LEVELS
        choices[:parameters=>:o2=>level=>:intercept] = parameters[(:o2, level, :intercept)]
        choices[:parameters=>:o2=>level=>:vent] = parameters[(:o2, level, :vent)]
    end
    for level in GLU_LEVELS
        choices[:parameters=>:glu=>level=>:intercept] = parameters[(:glu, level, :intercept)]
        choices[:parameters=>:glu=>level=>:vaso] = parameters[(:glu, level, :vaso)]
    end
    return choices
end


@gen function drift_proposal(trace, step_size, parameter_name)
    current_value = trace[parameter_name]
    {parameter_name} ~ mvnormal(current_value, Diagonal([step_size for _ in current_value]))
    return trace
end

function drift_update(trace, step_size)
    acceptance = 0

    for hr in HR_LEVELS
        (trace, a1) = mh(trace, drift_proposal, (step_size, :parameters => :hr => hr => :intercept))
        (trace, a2) = mh(trace, drift_proposal, (step_size, :parameters => :hr => hr => :abx))
        (trace, a3) = mh(trace, drift_proposal, (step_size, :parameters => :hr => hr => :vaso))
        acceptance += Int(a1) + Int(a2) + Int(a3)
    end

    for bp in BP_LEVELS
        (trace, a1) = mh(trace, drift_proposal, (step_size, :parameters => :bp => bp => :intercept))
        (trace, a2) = mh(trace, drift_proposal, (step_size, :parameters => :bp => bp => :abx))
        (trace, a3) = mh(trace, drift_proposal, (step_size, :parameters => :bp => bp => :vaso))
        acceptance += Int(a1) + Int(a2) + Int(a3)
    end

    for o2 in O2_LEVELS
        (trace, a1) = mh(trace, drift_proposal, (step_size, :parameters => :o2 => o2 => :intercept))
        (trace, a2) = mh(trace, drift_proposal, (step_size, :parameters => :o2 => o2 => :vent))
        acceptance += Int(a1) + Int(a2)
    end

    for glu in GLU_LEVELS
        (trace, a1) = mh(trace, drift_proposal, (step_size, :parameters => :glu => glu => :intercept))
        (trace, a2) = mh(trace, drift_proposal, (step_size, :parameters => :glu => glu => :vaso))
        acceptance += Int(a1) + Int(a2)
    end


    acceptance /= (3 * length(HR_LEVELS) + 3 * length(BP_LEVELS) + 2 * length(O2_LEVELS) + 2 * length(GLU_LEVELS))
    return trace, acceptance
end


const softmax_functions = SepsisParams(
    get_parameters,
    set_parameters,
    hr_probs,
    bp_probs,
    o2_probs,
    glu_probs,
    drift_update
)


end