module PPModel
export MCMCModel, update_model!, update_choicemap!

using ..SepsisTypes
using PyCall
sepsis_gym = pyimport("custom_sepsis")
using Gen

mutable struct MCMCModel
    type::Symbol
    functions::SepsisParams
    choices::ChoiceMap
    policies::Vector{Policy}
    start_states::Vector{State}
end


function update_choicemap!(episodes::ChoiceMap, i::Int, episode::Any) #episode comes from python
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

function update_model!(model::MCMCModel, until::Int, policy::Union{Policy,Nothing}=nothing)
    for i in length(model.policies):until
        pol = policy
        if policy !== nothing
            pol = to_gym_pol(policy)
        else
            pol = sepsis_gym.random_policy()
            policy = to_policy(pol)
        end
        episode = sepsis_gym.run_episode(pol)
        model.choices = update_choicemap!(model.choices, i, episode)
        push!(model.policies, policy)
        push!(model.start_states, to_state(episode.visited[1]))
    end
end


end