module Save
export MCMCRun, ModelData, RunData, save_run_jld, load_jld

using ..PPModel
using ..SepsisTypes
using Serialization
using Gen


struct MCMCRun
    scores::Vector
    acceptance::Float64
    params::Vector
end

struct ModelData
    choices::ChoiceMap
    policies::Vector{Policy}
    start_states::Vector{State}
end

struct RunData
    name::String
    type::Symbol
    model::ModelData
    mcmcs::Dict{Int,MCMCRun}
    mean_rewards::Dict{Int,Float64}
    index::Int

    function RunData(name::String, type::Symbol)
        return new(
            name,
            type,
            ModelData(choicemap(), [], []),
            Dict(),
            Dict(),
            0
        )
    end

    function RunData(name::String, type::Symbol, model::ModelData, mcmcs::Dict{Int,MCMCRun}, mean_rewards::Dict{Int,Float64}, index::Int)
        return new(name, type, model, mcmcs, mean_rewards, index)
    end
end

# function convert_to_rundata(ts::PPTSRun)::RunData
#     model_data = ModelData(
#         ts.model.choices,
#         ts.model.policies,
#         ts.model.start_states
#     )
#     mcmcs = Dict(
#         k => MCMCRun(
#             v.scores,
#             v.acceptance,
#             v.params
#         ) for (k, v) in ts.models
#     )
#     mean_rewards = ts.mean_rewards
#     index = maximum(keys(ts.models))
#     RunData(
#         ts.name,
#         ts.model.type,
#         model_data,
#         mcmcs,
#         mean_rewards,
#         index
#     )
# end


function save_run_jld(run_data::RunData, file_path::String="../../../data/mcmc/runs/")
    open("$(file_path)$(run_data.name).jld", "w") do io
        serialize(io, run_data)
    end
end

function load_jld(filename::String)
    open(filename, "r") do io
        return deserialize(io)
    end
end


end