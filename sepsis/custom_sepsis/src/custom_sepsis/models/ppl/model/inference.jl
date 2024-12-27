
module Inference
export block_update, drift_update, update_choicemap

using Gen
using ..Sepsis
using ..SepsisTypes
using LinearAlgebra

function block_update(trace)
    acceptance = 0

    for hr in HR_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :hr => hr => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :hr => hr => :abx))
        (trace, a3) = mh(trace, select(:parameters => :hr => hr => :vaso))
        acceptance += Int(a1) + Int(a2) + Int(a3)
    end

    for bp in BP_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :bp => bp => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :bp => bp => :abx))
        (trace, a3) = mh(trace, select(:parameters => :bp => bp => :vaso))
        acceptance += Int(a1) + Int(a2) + Int(a3)
    end

    for o2 in O2_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :o2 => o2 => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :o2 => o2 => :vent))
        acceptance += Int(a1) + Int(a2)
    end

    for glu in GLU_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :glu => glu => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :glu => glu => :vaso))
        acceptance += Int(a1) + Int(a2)
    end

    acceptance /= (3 * length(HR_LEVELS) + 3 * length(BP_LEVELS) + 2 * length(O2_LEVELS) + 2 * length(GLU_LEVELS))

    return trace, acceptance
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


function update_choicemap(episodes::ChoiceMap, i::Int, episode::Any) #episode comes from python
    t = length(episode.visited)
    for j in 1:t-1
        hr, bp, o2, glu, diab, abx, vaso, vent = episode.visited[j+1]
        episodes[:episodes=>i=>:trajectory=>j=>:hr] = Level(hr)
        episodes[:episodes=>i=>:trajectory=>j=>:bp] = Level(bp)
        episodes[:episodes=>i=>:trajectory=>j=>:o2] = Level(o2)
        episodes[:episodes=>i=>:trajectory=>j=>:glu] = Level(glu)
    end
    return episodes
end


end
