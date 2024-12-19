module SepsisTypes
export Action, Level, SUPER_LOW, LOW, NORMAL, HIGH, SUPER_HIGH, State, Policy, Parameters, get_env_params, STATES, ACTIONS, to_state, to_action, to_policy, HR_LEVELS, BP_LEVELS, O2_LEVELS, GLU_LEVELS

struct Action
    abx::Bool
    vaso::Bool
    vent::Bool
end

@enum Level begin
    SUPER_LOW = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1
    SUPER_HIGH = 2
end

HR_LEVELS = [LOW, NORMAL, HIGH]
BP_LEVELS = [LOW, NORMAL, HIGH]
O2_LEVELS = [LOW, NORMAL]
GLU_LEVELS = [SUPER_LOW, LOW, NORMAL, HIGH, SUPER_HIGH]

struct State
    hr::Level
    bp::Level
    o2::Level
    glu::Level
    diabetic::Bool
    abx::Bool
    vaso::Bool
    vent::Bool

    function State(hr::Level, bp::Level, o2::Level, glu::Level, diabetic::Bool, abx::Bool, vaso::Bool, vent::Bool)
        # Validate hr and bp (only LOW, NORMAL, HIGH are allowed)
        @assert hr in HR_LEVELS "hr must be LOW, NORMAL, or HIGH"
        @assert bp in BP_LEVELS "bp must be LOW, NORMAL, or HIGH"

        # Validate o2 (only LOW, NORMAL are allowed)
        @assert o2 in O2_LEVELS "o2 must be LOW or NORMAL"
        @assert glu in GLU_LEVELS

        # No restrictions for glu; it can take any Level
        return new(hr, bp, o2, glu, diabetic, abx, vaso, vent)
    end
end

function to_state(state::Tuple)::State
    hr, bp, o2, glu, diabetic, abx, vaso, vent = state
    return State(Level(hr), Level(bp), Level(o2), Level(glu), diabetic, abx, vaso, vent)
end
function to_action(action::Tuple)::Action
    return Action(action...)
end

function to_policy(policy::Dict{Any,Any})::Policy
    pol = Policy()
    for (state, action) in policy
        pol[to_state(state)] = to_action(action)
    end
    return pol
end

const Policy = Dict{State,Action}

const Parameters = Dict{Tuple,Vector{Float64}}

end