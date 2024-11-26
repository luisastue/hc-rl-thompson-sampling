function get_reward(pos::Pos, maze::Maze)::Int
    if maze[pos] == goal
        return 100
    else
        return -1
    end
end;

function value_iteration(maze::Maze, belief_controller::Controller, discount::Float64, eps::Float64)
    # Initialize state values for all positions in the maze
    state_values = Dict{Pos,Float64}()
    for pos in keys(maze)
        state_values[pos] = 0.0
    end

    policy = Dict{Pos,Button}()

    actions = [a, b, c, d]

    delta = eps + 1  # Arbitrary large number to start
    while delta > eps
        delta = 0

        # Loop through each position in the maze
        for pos in keys(maze)
            if maze[pos] == goal || maze[pos] == obstacle
                continue  # Skip terminal or invalid states
            end

            old_value = state_values[pos]
            action_values = []

            # Compute expected values for each action
            for action in actions
                direction_probabilities = belief_controller[action]
                for (idx, p) in enumerate(direction_probabilities)
                    direction = DIRECTIONS[idx]
                    new_pos = get_new_pos(pos, maze, direction)
                    reward = get_reward(new_pos, maze)
                    action_value = p * (reward + discount * state_values[new_pos])
                    push!(action_values, action_value)
                end
            end

            # Update the state value with the best action's value
            state_values[pos] = maximum(action_values)
            delta = max(delta, abs(old_value - state_values[pos]))
        end
    end

    # Extract policy
    for pos in keys(maze)
        # if maze[pos] == goal || maze[pos] == obstacle
        #     continue
        # end

        action_values = []
        for action in actions
            direction_probabilities = belief_controller[action]
            outcomes = []
            for (idx, p) in enumerate(direction_probabilities)
                direction = DIRECTIONS[idx]
                new_pos = get_new_pos(pos, maze, direction)
                reward = get_reward(new_pos, maze)
                action_value = p * (reward + discount * state_values[new_pos])
                push!(outcomes, action_value)
            end
            push!(action_values, maximum(outcomes))
        end

        # Select the best action
        best_action_idx = argmax(action_values)
        policy[pos] = actions[best_action_idx]
    end

    return policy, state_values
end;

function belief_value_iteration(belief_maze::BeliefMaze, belief_controller::Controller, discount::Float64, eps::Float64)
    # Initialize state values for all positions in the maze
    state_values = Dict{Pos,Float64}()
    for pos in keys(belief_maze.probs)
        state_values[pos] = 0.0
    end

    policy = Dict{Pos,Button}()

    actions = [a, b, c, d]

    delta = eps + 1  # Arbitrary large number to start
    while delta > eps
        delta = 0

        # Loop through each position in the maze
        for pos in keys(belief_maze.probs)
            if belief_maze.goal == pos
                continue  # Skip terminal or invalid states
            end
            old_value = state_values[pos]
            action_values = []

            # Compute expected values for each action
            for action in actions
                direction_probabilities = belief_controller[action]
                for (idx, p) in enumerate(direction_probabilities)
                    direction = DIRECTIONS[idx]
                    # case new_pos is empty
                    new_pos = get_new_pos(pos, direction)
                    new_pos = haskey(belief_maze.probs, new_pos) ? new_pos : pos
                    reward = belief_maze.goal == new_pos ? 100 : -1
                    direction_value_if_empty = p * (reward + discount * state_values[new_pos]) * belief_maze.probs[new_pos]

                    # case new_pos is obstacle
                    reward = belief_maze.goal == pos ? 100 : -1
                    direction_value_if_obstacle = p * (reward + discount * state_values[pos]) * (1 - belief_maze.probs[pos])

                    push!(action_values, direction_value_if_empty + direction_value_if_obstacle)
                end
            end

            # Update the state value with the best action's value
            state_values[pos] = maximum(action_values)
            delta = max(delta, abs(old_value - state_values[pos]))
        end
    end

    # Extract policy
    for pos in keys(belief_maze.probs)

        action_values = []
        for action in actions
            direction_probabilities = belief_controller[action]
            outcomes = []
            for (idx, p) in enumerate(direction_probabilities)
                direction = DIRECTIONS[idx]

                # case new_pos is empty
                new_pos = get_new_pos(pos, direction)
                new_pos = haskey(belief_maze.probs, new_pos) ? new_pos : pos
                reward = belief_maze.goal == new_pos ? 100 : -1
                direction_value_if_empty = p * (reward + discount * state_values[new_pos]) * belief_maze.probs[new_pos]

                # case new_pos is obstacle
                reward = belief_maze.goal == pos ? 100 : -1
                direction_value_if_obstacle = p * (reward + discount * state_values[pos]) * (1 - belief_maze.probs[pos])

                push!(outcomes, direction_value_if_empty + direction_value_if_obstacle)
            end
            push!(action_values, maximum(outcomes))
        end

        # Select the best action
        best_action_idx = argmax(action_values)
        policy[pos] = actions[best_action_idx]
    end

    return policy, state_values
end;
