# maze.jl

# Required libraries
using Random
using StatsBase
using Gen

##### Types #####

# Enumerations for Buttons, Directions, and Fields in the Maze
@enum Button a = 1 b = 2 c = 3 d = 4
@enum Direction up = 1 down = 2 left = 3 right = 4
@enum Field empty = 0 obstacle = 1 goal = 2

# Structure for representing positions in the maze
struct Pos
    x::Int
    y::Int
end

# Custom types for Policy, Maze, and Controller
const Policy = Dict{Pos,Button}
const Maze = Dict{Pos,Field}
const FIELDS = [empty, obstacle, goal]
struct BeliefMaze
    probs::Dict{Pos,Float64}
    goal::Pos
end


const DIRECTIONS = [up, down, left, right]
const DirectionProbabilities = Tuple{Float64,Float64,Float64,Float64} # up, down, left, right
const Controller = Dict{Button,DirectionProbabilities}


# Environment structure to hold controller, movement probabilities, and maze
struct Environment
    controller::Controller
    maze::Maze
end

# Structure to hold an episode's rewards and visited positions
struct Episode
    policy::Policy
    rewards::Vector{Int}
    visited::Vector{Pos}
end

##### Maze and Environment Generators #####

# Function to generate a random maze
function generate_maze(n::Int)::Maze
    maze = Maze()
    for x in 1:n
        for y in 1:n
            maze[Pos(x, y)] = empty
        end
    end
    for i in 1:rand(1:2*n)
        x = rand(1:n)
        y = rand(1:n)
        maze[Pos(x, y)] = obstacle
    end
    x = rand(1:n)
    y = rand(1:n)
    maze[Pos(x, y)] = goal
    return maze
end

# Function to generate the environment
function generate_deterministic_environment(n::Int)::Tuple{Environment,Pos}
    buttons = shuffle([a, b, c, d])
    controller = Dict(buttons .=> [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])
    maze = generate_maze(n)
    start = collect(maze)[findfirst(x -> x[2] == empty, collect(maze))][1]
    return Environment(controller, maze), start
end

function generate_nondet_environment(n::Int)::Tuple{Environment,Pos}
    buttons = shuffle([a, b, c, d])
    controller = Dict(buttons .=> [(0.8, 0, 0.1, 0.1), (0, 0.8, 0.1, 0.1), (0.1, 0.1, 0.8, 0), (0.1, 0.1, 0, 0.8)])
    maze = generate_maze(n)
    start = collect(maze)[findfirst(x -> x[2] == empty, collect(maze))][1]
    return Environment(controller, maze), start
end

##### Movement Logic #####

# Function to get the new position based on direction
function get_new_pos(pos::Pos, direction::Direction)
    new_pos = if direction == up
        Pos(pos.x, pos.y + 1)
    elseif direction == down
        Pos(pos.x, pos.y - 1)
    elseif direction == left
        Pos(pos.x - 1, pos.y)
    elseif direction == right
        Pos(pos.x + 1, pos.y)
    end
    return new_pos
end

# Function to get the new position based on direction
function get_new_pos(pos::Pos, maze::Maze, direction::Direction)
    new_pos = get_new_pos(pos, direction)

    if haskey(maze, new_pos) && maze[new_pos] != obstacle
        return new_pos
    else
        return pos
    end
end


##### Utility Functions #####

# Visualization function to print the maze with policy
function print_maze(maze::Maze, position::Pos=nothing)
    n = sqrt(length(maze))
    for y in reverse(1:n)
        row = ""
        for x in 1:n
            pos = Pos(x, y)
            if pos == position
                row *= " X"  # Start
            elseif haskey(maze, pos) && maze[pos] == obstacle
                row *= " #"
            elseif haskey(maze, pos) && maze[pos] == goal
                row *= " G"
            elseif haskey(maze, pos) && maze[pos] == empty
                row *= " ."
            end
        end
        println(row)
    end
end

function print_maze(environment::Environment, position::Pos, policy=Policy(), numbers=Dict{Pos,Float64}())
    direction_symbols = Dict(
        up => "↑",
        down => "↓",
        left => "←",
        right => "→"
    )

    n = sqrt(length(environment.maze))
    for y in reverse(1:n)
        row = ""
        for x in 1:n
            pos = Pos(x, y)
            if pos == position
                row *= " X"  # Start
            elseif haskey(environment.maze, pos) && environment.maze[pos] == obstacle
                row *= " #"
            elseif haskey(environment.maze, pos) && environment.maze[pos] == goal
                row *= " G"
            elseif haskey(environment.maze, pos) && environment.maze[pos] == empty
                row *= " ."
            end

            if haskey(policy, pos)
                probs = environment.controller[policy[pos]]
                direction = DIRECTIONS[argmax(probs)]
                row *= direction_symbols[direction]
            end

            if haskey(numbers, pos)
                nr = round(numbers[pos], digits=1)
                row *= string(nr) * " "
            else
                row *= "  "
            end
        end
        println(row)
        println()
    end
end

# function print_choices(environment::Environment, trace)
#     choices = get_choices(trace)

#     n = sqrt(length(environment.maze))
#     for y in reverse(1:n)
#         row = ""
#         for x in 1:n
#             pos = Pos(x, y)
#             for b in [a,b,c,d]
#                 visited = [has_value(choices, i => pos => b => :new_pos) for i in 1:100]
#                 if any(visited)
#                     pos_idx = findfirst(visited)
#                     row *= " " * string(pos_idx)
#                 elseif pos == position
#                     row *= " X"  # Start
#                 elseif haskey(environment.maze, pos) && environment.maze[pos] == obstacle
#                     row *= " #"
#                 elseif haskey(environment.maze, pos) && environment.maze[pos] == goal
#                     row *= " G"
#                 elseif haskey(environment.maze, pos) && environment.maze[pos] == empty
#                     row *= " ."
#                 end
#         end
#         println(row)
#     end
# end

function random_policy(n)::Policy
    policy = Policy()
    for x in 1:n
        for y in 1:n
            policy[Pos(x, y)] = rand([a, b, c, d])
        end
    end
    return policy
end


using CairoMakie

# Function to plot the BeliefMaze
function plot_belief_maze(belief_maze::BeliefMaze)
    probs = belief_maze.probs
    # Extract positions and corresponding opacities from the model
    positions = collect(keys(probs))

    # Determine the size of the grid by finding the max x and y coordinates
    max_x = maximum(pos -> pos.x, positions)
    max_y = maximum(pos -> pos.y, positions)

    # Create an empty matrix for the opacity values (assuming all positions in the grid are valid)
    opacity_grid = fill(1.0, (max_x, max_y))

    # Fill the opacity grid with values from the model
    for (pos, opacity) in probs
        opacity_grid[pos.y, pos.x] = 1 - opacity
    end

    # Plot the grid using CairoMakie
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())

    heatmap!(ax, opacity_grid', colormap=:Greys, colorrange=(0.0, 1.0))
    scatter!(ax, [belief_maze.goal.x], [belief_maze.goal.y], color=:red, markersize=10)

    # Show the plot
    return fig
end

# Function to plot the evolution of a field's value over time
function plot_field_evolution(belief_mazes::Vector{BeliefMaze}, position::Pos)
    # Extract the value of the given position over time
    values_over_time = [get(model.probs, position, NaN) for model in belief_mazes]

    # Time steps corresponding to each BeliefMaze
    time_steps = 1:length(belief_mazes)

    # Create the figure and axis for the plot
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Field Value",
        title="Evolution of Field Value at Position $position")

    # Plot the values over time
    lines!(ax, time_steps, values_over_time, color=:blue, linewidth=2)

    # Show the plot
    display(fig)
end

function plot_evolution(values::Vector{Int})

    # Time steps corresponding to each BeliefMaze
    time_steps = 1:length(values)

    # Create the figure and axis for the plot
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Value",
        title="Evolution over time")

    # Plot the values over time
    lines!(ax, time_steps, values, color=:blue, linewidth=2)

    # Show the plot
    display(fig)
end

using CairoMakie

# Function to plot the evolution of all fields over time in a grid
function plot_grid_field_evolution(belief_mazes::Vector{BeliefMaze})
    # Extract all unique positions (fields) from the BeliefMazes
    positions = belief_mazes[1].probs |> keys |> collect

    # Determine the grid size (assuming square-like grid for simplicity)
    max_x = maximum(pos -> pos.x, positions)
    max_y = maximum(pos -> pos.y, positions)

    # Create a figure with a grid layout
    fig = Figure(resolution=(1000, 1000))
    grid = fig[1, 1] = GridLayout(max_x, max_y, alignmode=Outside(2))

    # Loop through each position and plot its value over time in a separate subplot
    for pos in positions
        # Extract the values of the current position over time
        values_over_time = [get(model.probs, pos, NaN) for model in belief_mazes]

        # Time steps corresponding to each BeliefMaze
        time_steps = 1:length(belief_mazes)

        # Create an axis for the subplot in the appropriate position
        ax = Axis(grid[max_y-pos.y+1, pos.x], title="$pos", xlabel="Time", ylabel="Value")
        ylims!(ax, -0.1, 1.1)

        # Plot the values over time for this specific position
        lines!(ax, time_steps, values_over_time, color=:blue, linewidth=2)
    end

    # Show the grid of plots
    display(fig)
end

# Function to plot the density evolution for each position in a grid of subplots
function plot_density_maze(belief_mazes::Vector{BeliefMaze})
    # Extract all unique positions (fields) from the BeliefMazes
    positions = belief_mazes[1].probs |> keys |> collect

    # Determine the grid size (assuming square-like grid for simplicity)
    max_x = maximum(pos -> pos.x, positions)
    max_y = maximum(pos -> pos.y, positions)

    # Create a figure with a grid layout
    fig = Figure(resolution=(3000, 3000))
    grid = fig[1, 1] = GridLayout(max_x, max_y, alignmode=Outside(2))

    # Loop through each position and plot its density over time in a separate subplot
    for pos in positions
        # Extract the values (densities) of the current position over time
        densities = [get(model.probs, pos, NaN) for model in belief_mazes]

        # Time steps corresponding to each BeliefMaze
        time_steps = 1:length(belief_mazes)

        # Create an axis for the subplot in the appropriate position
        ax = Axis(grid[max_y-pos.y+1, pos.x], title="$pos", xlabel="Time", ylabel="Density")

        # Plot the density evolution over time for this specific position
        density!(ax, densities, color=:blue)
    end

    # Show the grid of density plots
    display(fig)
end
