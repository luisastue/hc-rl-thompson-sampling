# Gen Functions
using Gen

@dist function labeled_categorical(labels, probs)
    index = categorical(probs)
    labels[index]
end;

@gen function simulate_action(maze::Maze, controller::Controller, pos::Pos, button::Button)::Pos
    probs = controller[button]
    possible_targets = [get_new_pos(pos, maze, dir) for dir in DIRECTIONS]
    target = {button => :new_pos} ~ labeled_categorical(possible_targets, [probs[1], probs[2], probs[3], probs[4]])
    return target
end;

@gen function simulate_episode(maze::Maze, controller::Controller, start::Pos, episode_length::Int, policy::Policy)
    pos = start
    playing = true
    visited = [pos]
    rewards = []
    for t in 1:episode_length
        button = policy[pos]
        new_pos = {t => pos} ~ simulate_action(maze, controller, pos, button)
        if playing
            pos = new_pos
            push!(visited, pos)
            if maze[pos] == goal
                reward = {t => :reward} ~ labeled_categorical([100], [1.0])
            else
                reward = {t => :reward} ~ labeled_categorical([-1], [1.0])
            end

            push!(rewards, reward)
        end
    end
    return Episode(policy, rewards, visited)
end;

@gen function simulate_belief_action(maze::BeliefMaze, controller::Controller, pos::Pos, button::Button)::Pos
    target_positions = [get_new_pos(pos, dir) for dir in DIRECTIONS]
    possible_targets = [haskey(maze.probs, target_pos) ? target_pos : pos for target_pos in target_positions]
    contr_probs = [controller[button][i] for i in 1:4]
    maze_probs = [maze.probs[poss_targ] for poss_targ in possible_targets]
    target = {button => :new_pos} ~ labeled_categorical([possible_targets..., pos], [contr_probs .* maze_probs..., 1 - sum(contr_probs .* maze_probs)])
    return target
end;

@gen function simulate_belief_episode(maze::BeliefMaze, controller::Controller, start::Pos, episode_length::Int, policy::Policy)
    pos = start
    playing = true
    visited = [pos]
    rewards = []
    for t in 1:episode_length
        button = policy[pos]
        new_pos = {t => pos} ~ simulate_belief_action(maze, controller, pos, button)
        if playing
            pos = new_pos
            push!(visited, pos)
            if maze.goal == pos
                reward = {t => :reward} ~ labeled_categorical([100], [1.0])
            else
                reward = {t => :reward} ~ labeled_categorical([-1], [1.0])
            end

            push!(rewards, reward)
        end
    end
    return Episode(policy, rewards, visited)
end;
