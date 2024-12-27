

module Softmax
export softmax_functions

using Gen
using ..SepsisTypes
using ..Sepsis
using LinearAlgebra

softmax(vector) = exp.(vector) / sum(exp.(vector))
add_0(vector) = [vector; 0]

const Params = Dict{Tuple{Symbol,Level,Symbol},Float64}


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



const softmax_functions = SepsisParams(
    get_parameters,
    hr_probs,
    bp_probs,
    o2_probs,
    glu_probs,
)


end