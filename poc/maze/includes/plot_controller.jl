using CairoMakie, Distributions

function plot_controller_traces(traces)
    last_few_traces = traces #[end-100:end]

    # Extract probabilities for each button from the traces
    button_a_probs = [get_choices(trace)[:ctrl=>a] for trace in last_few_traces]
    button_b_probs = [get_choices(trace)[:ctrl=>b] for trace in last_few_traces]
    button_c_probs = [get_choices(trace)[:ctrl=>c] for trace in last_few_traces]
    button_d_probs = [get_choices(trace)[:ctrl=>d] for trace in last_few_traces]

    # Create a figure with 4 rows, each row containing 4 line plots for the probabilities
    fig = Figure(size=(1200, 800))  # Replaced `resolution` with `size`

    # For button A, plot all 4 probabilities with y-axis limit [0, 1]
    for i in 1:4
        ax_a = Axis(fig[1, i], title="Button A: Probability $(DIRECTIONS[i])", xlabel="Step", ylabel="Probability", limits=(nothing, (0, 1)))
        lines!(ax_a, 1:length(button_a_probs), [p[i] for p in button_a_probs], color=:blue)
    end

    # For button B, plot all 4 probabilities with y-axis limit [0, 1]
    for i in 1:4
        ax_b = Axis(fig[2, i], title="Button B: Probability $(DIRECTIONS[i])", xlabel="Step", ylabel="Probability", limits=(nothing, (0, 1)))
        lines!(ax_b, 1:length(button_b_probs), [p[i] for p in button_b_probs], color=:red)
    end

    # For button C, plot all 4 probabilities with y-axis limit [0, 1]
    for i in 1:4
        ax_c = Axis(fig[3, i], title="Button C: Probability $(DIRECTIONS[i])", xlabel="Step", ylabel="Probability", limits=(nothing, (0, 1)))
        lines!(ax_c, 1:length(button_c_probs), [p[i] for p in button_c_probs], color=:green)
    end

    # For button D, plot all 4 probabilities with y-axis limit [0, 1]
    for i in 1:4
        ax_d = Axis(fig[4, i], title="Button D: Probability $(DIRECTIONS[i])", xlabel="Step", ylabel="Probability", limits=(nothing, (0, 1)))
        lines!(ax_d, 1:length(button_d_probs), [p[i] for p in button_d_probs], color=:orange)
    end

    # Display the figure
    display(fig)
end
