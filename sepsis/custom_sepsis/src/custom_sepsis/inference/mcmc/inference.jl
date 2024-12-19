
module Inference
export block_update, update_choicemap
using Gen
using ..Sepsis
using ..SepsisTypes

function block_update(trace)
    acceptance = 0

    for hr in HR_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :hr => hr => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :hr => hr => :abx))
        (trace, a3) = mh(trace, select(:parameters => :hr => hr => :vaso))
        acceptance += a1 + a2 + a3
    end

    for bp in BP_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :bp => bp => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :bp => bp => :abx))
        (trace, a3) = mh(trace, select(:parameters => :bp => bp => :vaso))
        acceptance += a1 + a2 + a3
    end

    for o2 in O2_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :o2 => o2 => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :o2 => o2 => :vent))
        acceptance += a1 + a2
    end

    for glu in GLU_LEVELS
        (trace, a1) = mh(trace, select(:parameters => :glu => glu => :intercept))
        (trace, a2) = mh(trace, select(:parameters => :glu => glu => :vaso))
        acceptance += a1 + a2
    end

    acceptance /= 10

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
