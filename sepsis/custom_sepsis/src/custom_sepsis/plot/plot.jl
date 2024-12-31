module Plot
export DQNType, colors_dict, label_dict, moving_avg, add_dqn!, colors, load_dqn_from_json, random_mean, TSType, load_ts_from_json, HistoryType, load_history_from_json

using ..SepsisTypes
using ..Sepsis
using ..Simple
using ..Softmax
using ..Smart
using ..ValueIteration
using ..PPModel
using ..Save
using ..PPLRun
using ..Plot
using Colors
using JSON3
using Serialization
using Statistics
using CairoMakie

cols = distinguishable_colors(6, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
colors = map(col -> (red(col), green(col), blue(col)), cols)

colors_dict = Dict(
    :Simple => colors[1],
    :Medium => colors[4],
    :None => colors[6],
    :Softmax => colors[3],
    :SimplePPL => colors[2],
    :DQN => colors[5],
)
label_dict = Dict(
    :Simple => "SimpleDBN",
    :Medium => "MediumDBN",
    :None => "FullDBN",
    :Softmax => "SoftmaxPPL",
    :SimplePPL => "SimplePPL",
    :DQN => "DQN"
)

struct TSType
    # policies::Dict{Int, Vector{Policy}}
    mean_rewards::Dict{Int,Float64}
    name::String
    info::Dict
end
function load_ts_from_json(file_path::String)::TSType
    json_data = JSON3.read(file_path)
    mean_rewards = Dict{Int,Float64}(
        parse(Int, string(k)) => Float64(v) for (k, v) in json_data["mean_rewards"]
    )
    name = json_data["info"]["name"]
    # info = Dict(json_data["info"])
    info = Dict(
        string(k) => string(v) for (k, v) in json_data["info"]
    )
    return TSType(mean_rewards, name, info)
end

struct HistoryType
    # policies::Dict{Int, Vector{Policy}}
    mean_rewards::Dict{Int,Vector{Float64}}
    name::String
    info::Dict
end

function load_history_from_json(file_path::String)::HistoryType
    json_data = JSON3.read(file_path)
    mean_rewards = Dict{Int,Vector{Float64}}(
        parse(Int, string(k)) => Float64.(v) for (k, v) in json_data["mean_rewards"]
    )
    name = json_data["info"]["name"]
    # info = Dict(json_data["info"])
    info = Dict(
        string(k) => string(v) for (k, v) in json_data["info"]
    )
    return HistoryType(mean_rewards, name, info)
end


struct DQNType
    # policies::Dict{Int, Vector{Policy}}
    mean_rewards::Vector{Float64}
    std_rewards::Vector{Float64}
    name::String
    info::Dict
end

function moving_avg(data, window_size)
    half_window = div(window_size, 2)

    # Pad the data with the edge values to handle borders
    padded_data = vcat(data[1:half_window], data, data[end-half_window:end])

    # Compute the moving average using a sliding window
    smoothed = [mean(padded_data[i:i+window_size-1]) for i in 1:length(data)]

    return smoothed
end
function add_dqn!(ax, dqn, window_size)
    smoothed = moving_avg(dqn.mean_rewards, window_size)

    lines!(ax, 1:length(smoothed), smoothed, color=colors_dict[:DQN],)
    lines!(ax, 1:length(dqn.mean_rewards), dqn.mean_rewards, color=(colors_dict[:DQN], 0.2), label="Mean Reward of 50 DQN runs")
    # band!(ax, 1:length(smoothed_std), smoothed .- smoothed_std, smoothed .- smoothed_std, color=(colors_dict[:DQN], 0.2))
end


function load_dqn_from_json(file_path)
    json_data = JSON3.read(file_path)
    mean_rewards = [Float64(rew) for rew in json_data["mean_rewards"]]
    std_rewards = [Float64(rew) for rew in json_data["std_rewards"]]
    name = json_data["name"]
    info = Dict(
        string(k) => string(v) for (k, v) in json_data["info"]
    )
    return DQNType(mean_rewards, std_rewards, name, info)
end
random_mean = -0.6662000000000002 #mean([sepsis_gym.evaluate_policy(sepsis_gym.random_policy(), 1000) for i in 1:100])


end