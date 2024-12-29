

module Smart
export smart_functions

using Gen
using ..SepsisTypes
using ..Sepsis
using LinearAlgebra



function hr_probs(parameters::Dict, state::State, action::Action)
    if state.hr == LOW
        return [0.1, 0.9, 0]
    elseif state.hr == NORMAL
        return [0.1, 0.8, 0.1]
    elseif state.hr == HIGH
        if action.abx
            return [0, 0.9, 0.1]
        else
            return [0, 0.1, 0.9]
        end
    end
end


function bp_probs(parameters::Dict, state::State, action::Action)
    if state.bp == LOW
        if action.vaso
            return [0.1, 0.9, 0]
        else
            return [0.9, 0.1, 0]
        end
    elseif state.bp == NORMAL
        if action.vaso
            return [0, 0.3, 0.7]
        else
            return [0.1, 0.8, 0.1]
        end
    elseif state.bp == HIGH
        if action.abx
            return [0, 0.9, 0.1]
        else
            return [0, 0.1, 0.9]
        end
    end
end

function o2_probs(parameters::Dict, state::State, action::Action)
    if state.o2 == LOW
        if action.vent
            return [0.1, 0.9]
        else
            return [0.9, 0.1]
        end
    elseif state.o2 == NORMAL
        if action.vent
            return [0, 1]
        else
            return [0.1, 0.9]
        end
    end
end

function glu_probs(parameters::Dict, state::State, action::Action)
    if state.glu == SUPER_LOW
        if action.vaso
            return [0.1, 0.9, 0, 0, 0]
        else
            return [0.9, 0.1, 0, 0, 0]
        end
    elseif state.glu == LOW
        if action.vaso
            return [0, 0.1, 0.9, 0, 0]
        else
            return [0, 0.9, 0.1, 0, 0]
        end
    elseif state.glu == NORMAL
        if action.vaso
            return [0, 0, 0.1, 0.9, 0]
        else
            return [0, 0.05, 0.9, 0.05, 0]
        end
    elseif state.glu == HIGH
        if action.vaso
            return [0, 0, 0, 0.1, 0.9]
        else
            return [0, 0, 0.1, 0.9, 0]
        end
    elseif state.glu == SUPER_HIGH
        if action.vaso
            return [0, 0, 0, 0, 1]
        else
            return [0, 0, 0, 0.1, 0.9]
        end
    end
end


function dummy()
    pass
end

const smart_functions = SepsisParams(
    dummy,
    dummy,
    hr_probs,
    bp_probs,
    o2_probs,
    glu_probs,
    dummy)

end