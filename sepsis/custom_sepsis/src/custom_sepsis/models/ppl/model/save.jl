module Save
export save_history_jld, save_ts_jld, load_jld

using ..PPModel
using ..History
using ..ThompsonSampling
using Serialization


function save_history_jld(history::PPHistoryRun)
    open("../../../data/mcmc/history/$(history.name).jld", "w") do io
        serialize(io, history)
    end
end

function save_ts_jld(thompson_sampling::PPTSRun)
    open("../../../data/mcmc/ts/$(thompson_sampling.name).jld", "w") do io
        serialize(io, thompson_sampling)
    end
end

function load_jld(filename::String)
    open(filename, "r") do io
        return deserialize(io)
    end
end

end