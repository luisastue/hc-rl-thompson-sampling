module Plot
export DQNType, colors_dict, label_dict, moving_avg, add_dqn!, colors, load_dqn_from_json, random_mean, TSType, load_ts_from_json, HistoryType, load_history_from_json, load_rewards_from_json

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


function create_variants(color::RGB)
    variant1 = RGB(clamp(red(color) * 0.9, 0.0, 1.0),  # Darker
        clamp(green(color) * 0.9, 0.0, 1.0),
        clamp(blue(color) * 0.9, 0.0, 1.0))
    variant2 = RGB(clamp(red(color) * 1.05, 0.0, 1.0),  # Brighter
        clamp(green(color) * 1.05, 0.0, 1.0),
        clamp(blue(color) * 1.05, 0.0, 1.0))
    variant3 = RGB(clamp(red(color) * 0.65, 0.0, 1.0),  # Halve RGB components
        clamp(green(color) * 0.65, 0.0, 1.0),
        clamp(blue(color) * 0.65, 0.0, 1.0))

    return (variant1, variant2, variant3)
end
variables = [:Simple, :Medium, :None, :Softmax, :SimplePPL, :DQN, :QLearning]
cols = distinguishable_colors(length(variables), [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
colors = map(create_variants, cols)

colors_dict = Dict(
    :Simple => colors[1][2],
    :Simple100 => colors[1][2],
    :Simple1 => colors[1][2],
    :Medium => colors[4][2],
    :Medium100 => colors[4][2],
    :Medium1 => colors[4][2],
    :None => colors[6][2],
    :None100 => colors[6][2],
    :None1 => colors[6][2],
    :Softmax => colors[3][2],
    :Softmax100 => colors[3][2],
    :Softmax1 => colors[3][2],
    :SimplePPL => colors[2][2],
    :SimplePPL100 => colors[2][2],
    :SimplePPL1 => colors[2][2],
    :DQN_SE => colors[5][1],
    :DQN_AR => colors[5][2],
    :DQN_CR => colors[5][3],
    :QLearning => colors[7][2],
)
label_dict = Dict(
    :Simple => "SimpleDBN",
    :Simple100 => "SimpleDBN_TS100",
    :Simple1 => "SimpleDBN_TS1",
    :Medium => "MediumDBN",
    :Medium100 => "MediumDBN_TS100",
    :Medium1 => "MediumDBN_TS1",
    :None => "FullDBN",
    :None100 => "FullDBN_TS100",
    :None1 => "FullDBN_TS1",
    :Softmax => "SoftmaxPPL",
    :Softmax100 => "SoftmaxPPL_TS100",
    :Softmax1 => "SoftmaxPPL_TS1",
    :SimplePPL => "SimplePPL",
    :SimplePPL100 => "SimplePPL_TS100",
    :SimplePPL1 => "SimplePPL_TS1",
    :DQN_SE => "DQN_SE",
    :DQN_AR => "DQN_AR",
    :DQN_CR => "DQN_CR",
    :QLearning => "QLearning",
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

struct MeanRewardsType
    mean_rewards::Vector{Float64}
    individual_runs::Vector{Vector{Float64}}
    smoothed_mean::Vector{Float64}
    smoothed_std::Vector{Float64}
    name::String
    info::Dict
end
function load_rewards_from_json(file_path)
    json_data = JSON3.read(file_path)
    mean_rewards = [Float64(rew) for rew in json_data["mean_rewards"]]
    individual_runs = [[Float64(rew) for rew in r] for r in json_data["individual_runs"]]
    smoothed_mean = [Float64(rew) for rew in json_data["smoothed_mean"]]
    smoothed_std = [Float64(rew) for rew in json_data["smoothed_std"]]
    name = json_data["name"]
    info = Dict(
        string(k) => string(v) for (k, v) in json_data["info"]
    )
    return MeanRewardsType(mean_rewards, individual_runs, smoothed_mean, smoothed_std, name, info)
end

function moving_avg(data, window_size)
    half_window = div(window_size, 2)

    # Pad the data with the edge values to handle borders
    padded_data = vcat(data[1:half_window], data, data[end-half_window:end])

    # Compute the moving average using a sliding window
    smoothed = [mean(padded_data[i:i+window_size-1]) for i in 1:length(data)]

    return smoothed
end
function add_dqn!(ax, dqn, window_size, show_avg=true)
    lines!(ax, 1:length(dqn.mean_rewards), dqn.mean_rewards, color=(colors_dict[:DQN], 0.2), label="Mean Reward of 50 DQN runs")
    if show_avg
        smoothed = moving_avg(dqn.mean_rewards, window_size)
        lines!(ax, 1:length(smoothed), smoothed, color=colors_dict[:DQN],)
        return smoothed
    end
    # band!(ax, 1:length(smoothed_std), smoothed .- smoothed_std, smoothed .- smoothed_std, color=(colors_dict[:DQN], 0.2))
end

random_mean = -0.6662000000000002 #mean([sepsis_gym.evaluate_policy(sepsis_gym.random_policy(), 1000) for i in 1:100])


end