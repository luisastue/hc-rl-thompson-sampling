
module Inference
export block_update, update_choicemap
using Gen
using ..Sepsis
using ..SepsisTypes

function block_update(trace, nr_episodes)
    (trace, acc) = mh(trace, select(:beliefs => :abx_on_hr_H_N))
    (trace, acc) = mh(trace, select(:beliefs => :abx_on_bp_H_N))
    (trace, acc) = mh(trace, select(:beliefs => :abx_withdrawn_hr_N_H))
    (trace, acc) = mh(trace, select(:beliefs => :abx_withdrawn_bp_N_H))
    (trace, acc) = mh(trace, select(:beliefs => :vent_on_o2_L_N))
    (trace, acc) = mh(trace, select(:beliefs => :vent_withdrawn_o2_N_L))
    (trace, acc) = mh(trace, select(:beliefs => :nond_vaso_on_bp_L_N))
    (trace, acc) = mh(trace, select(:beliefs => :nond_vaso_on_bp_N_H))
    (trace, acc) = mh(trace, select(:beliefs => :nond_vaso_withdrawn_bp_N_L))
    (trace, acc) = mh(trace, select(:beliefs => :nond_vaso_withdrawn_bp_H_N))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_on_bp_L_N))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_on_bp_L_H))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_on_bp_N_H))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_on_glu_up))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_withdrawn_bp_N_L))
    (trace, acc) = mh(trace, select(:beliefs => :diab_vaso_withdrawn_bp_H_N))
    (trace, acc) = mh(trace, select(:beliefs => :fluct))
    (trace, acc) = mh(trace, select(:beliefs => :diab_fluct_glu))
    for i in 1:nr_episodes
        states, rewards = trace[:episode=>i]
        for t in 1:length(states)
            (trace, acc) = mh(trace, select(:episode => i => t => :abx_hr))
            (trace, acc) = mh(trace, select(:episode => i => t => :abx_bp))
            (trace, acc) = mh(trace, select(:episode => i => t => :abx_withdrawn_hr))
            (trace, acc) = mh(trace, select(:episode => i => t => :abx_withdrawn_bp))
            (trace, acc) = mh(trace, select(:episode => i => t => :vent_o2))
            (trace, acc) = mh(trace, select(:episode => i => t => :vent_withdrawn_o2))
            (trace, acc) = mh(trace, select(:episode => i => t => :nond_vaso_bp_L_N))
            (trace, acc) = mh(trace, select(:episode => i => t => :nond_vaso_bp_N_H))
            (trace, acc) = mh(trace, select(:episode => i => t => :nond_vaso_withdrawn_bp_N_L))
            (trace, acc) = mh(trace, select(:episode => i => t => :nond_vaso_withdrawn_bp_H_N))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_bp_L_N))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_bp_L_H))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_bp_N_H))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_glu_SUPER_LOW))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_glu_LOW))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_glu_NORMAL))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_glu_HIGH))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_glu_HIGH))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_withdrawn_bp_N_L))
            (trace, acc) = mh(trace, select(:episode => i => t => :diab_vaso_withdrawn_bp_H_N))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_hr))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_hr_change))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_bp))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_bp_change))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_o2))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_o2_change))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_glu))
            (trace, acc) = mh(trace, select(:episode => i => t => :fluct_glu_change))
        end
    end
    return trace
end

function update_choicemap(episodes::ChoiceMap, i::Int, episode::Any)
    for (j, state) in enumerate(episode.visited[2:end])
        episodes[:episode=>i=>:new_state=>j] = to_state(episode.visited[j+1])
        episodes[:episode=>i=>:reward=>j] = episode.rewards[j]
    end
    return episodes
end

end
