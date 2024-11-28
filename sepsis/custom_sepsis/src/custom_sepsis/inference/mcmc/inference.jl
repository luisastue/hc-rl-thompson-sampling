
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
        for t in 1:10
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :abx_hr))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :abx_bp))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :abx_withdrawn_hr))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :abx_withdrawn_bp))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :vent_o2))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :vent_withdrawn_o2))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :nond_vaso_bp_L_N))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :nond_vaso_bp_N_H))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :nond_vaso_withdrawn_bp_N_L))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :nond_vaso_withdrawn_bp_H_N))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_bp_L_N))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_bp_L_H))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_bp_N_H))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_glu_up))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_withdrawn_bp_N_L))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :diab_vaso_withdrawn_bp_H_N))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_hr))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_hr_change))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_bp))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_bp_change))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_o2))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_o2_change))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_glu))
            (trace, acc) = mh(trace, select(:episode => i => :get_next => t => :fluct_glu_change))
        end
    end
    return trace
end

function update_choicemap(episodes::ChoiceMap, i::Int, episode::Any)
    for j in 1:10
        hr, bp, o2, glu, diab, abx, vaso, vent = episode.visited[j+1]
        episodes[:episode=>i=>:new_state=>j=>:hr] = Int(hr)
        episodes[:episode=>i=>:new_state=>j=>:bp] = Int(bp)
        episodes[:episode=>i=>:new_state=>j=>:o2] = Int(o2)
        episodes[:episode=>i=>:new_state=>j=>:glu] = Int(glu)
        episodes[:episode=>i=>:new_state=>j=>:diabetic] = Int(diab)
        episodes[:episode=>i=>:new_state=>j=>:abx] = Int(abx)
        episodes[:episode=>i=>:new_state=>j=>:vaso] = Int(vaso)
        episodes[:episode=>i=>:new_state=>j=>:vent] = Int(vent)
        # episodes[:episode=>i=>:reward=>j] = Int(episode.rewards[j])
    end
    return episodes
end

end
