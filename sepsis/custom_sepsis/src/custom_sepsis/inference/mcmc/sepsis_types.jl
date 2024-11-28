module SepsisTypes
export Action, Level, SUPER_LOW, LOW, NORMAL, HIGH, SUPER_HIGH, State, Policy, EnvParameters, get_env_params, STATES, ACTIONS, to_state, to_action, to_policy

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
        @assert hr in (LOW, NORMAL, HIGH) "hr must be LOW, NORMAL, or HIGH"
        @assert bp in (LOW, NORMAL, HIGH) "bp must be LOW, NORMAL, or HIGH"

        # Validate o2 (only LOW, NORMAL are allowed)
        @assert o2 in (LOW, NORMAL) "o2 must be LOW or NORMAL"

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

struct EnvParameters
    abx_on_hr_H_N::Float64
    abx_on_bp_H_N::Float64
    abx_withdrawn_hr_N_H::Float64
    abx_withdrawn_bp_N_H::Float64
    vent_on_o2_L_N::Float64
    vent_withdrawn_o2_N_L::Float64
    nond_vaso_on_bp_L_N::Float64
    nond_vaso_on_bp_N_H::Float64
    diab_vaso_on_bp_L_N::Float64
    diab_vaso_on_bp_L_H::Float64
    diab_vaso_on_bp_N_H::Float64
    diab_vaso_on_glu_up::Float64
    nond_vaso_withdrawn_bp_N_L::Float64
    nond_vaso_withdrawn_bp_H_N::Float64
    diab_vaso_withdrawn_bp_N_L::Float64
    diab_vaso_withdrawn_bp_H_N::Float64
    fluct::Float64
    diab_fluct_glu::Float64
end

function get_env_params(p::EnvParameters)::Vector{Float64}
    return [getfield(p, f) for f in fieldnames(EnvParameters)]
end

end