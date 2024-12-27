module Inference
export get_update_function

using Gen


function block_update_simple(trace, step_size)
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
    acceptance += Int(a1) + Int(a2) + Int(a3) + Int(a4) + Int(a5) + Int(a6) + Int(a7) + Int(a8) + Int(a9) + Int(a10) + Int(a11) + Int(a12) + Int(a13) + Int(a14)
    acceptance /= 14

    return trace, acceptance
end


@gen function drift_proposal(trace, step_size, parameter_name)
    current_value = trace[parameter_name]
    {parameter_name} ~ mvnormal(current_value, Diagonal([step_size for _ in current_value]))
    return trace
end

function drift_update_softmax(trace, step_size)
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

function get_update_function(type::Symbol)
    if type == :simple
        return block_update_simple
    elseif type == :softmax
        return drift_update_softmax
    else
        error("Unknown update type")
    end
end

end