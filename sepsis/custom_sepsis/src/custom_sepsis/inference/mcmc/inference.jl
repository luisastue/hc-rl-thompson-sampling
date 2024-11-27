module Inference
include("sepsis.jl")
include("sepsis_types.jl")
using .Sepsis
using .SepsisTypes


function block_update(trace, nr_episodes)
    (trace, _) = mh(trace, select(:beliefs))
    for i in 1:nr_episodes
        states, rewards = trace[i=>:episode]
        for t in 1:length(states)
            (trace, _) = mh(trace, select(i => :episode => t))
        end
    end
    return trace
end

end
