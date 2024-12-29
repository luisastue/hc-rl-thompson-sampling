include("model/sepsis_types.jl")
include("model/sepsis_gen.jl")
include("model/simple.jl")
include("model/softmax.jl")
include("model/smart.jl")
include("value_iter.jl")
include("model/pp_model.jl")
include("evaluate/history.jl")
include("evaluate/thompson_sampling.jl")
include("model/save.jl")

using .SepsisTypes
using .Sepsis
using .Simple
using .Softmax
using .Smart
using .ValueIteration
using .PPModel
using .History
using .ThompsonSampling
using .Save
using Colors

vars = 1:6
cols = distinguishable_colors(length(vars), [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
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