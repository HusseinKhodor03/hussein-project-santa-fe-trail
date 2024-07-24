import matplotlib.pyplot as plt

# Enable interactive mode for live plotting
plt.ion()


def plot(scores, mean_scores, time_steps, mean_time_steps):
    # Clear the current figure to prepare for new plots
    plt.clf()

    # Get the current figure manager to manipulate the plot window
    manager = plt.get_current_fig_manager()
    manager.resize(1024, 768)
    manager.set_window_title("Santa Fe Training Plot")

    # Create the first subplot for scores
    plt.subplot(2, 1, 1)
    plt.title("Scores", fontsize=14)
    plt.xlabel("Number of Runs")
    plt.ylabel("Score")
    plt.plot(scores, label="Scores", color="blue")
    plt.plot(mean_scores, label="Mean Scores", color="orange")
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.legend()
    plt.grid()

    # Create the second subplot for time steps
    plt.subplot(2, 1, 2)
    plt.title("Time Steps", fontsize=14)
    plt.xlabel("Number of Runs")
    plt.ylabel("Time Steps")
    plt.plot(time_steps, label="Time Steps", color="green")
    plt.plot(mean_time_steps, label="Mean Time Steps", color="yellow")
    plt.text(len(time_steps) - 1, time_steps[-1], str(time_steps[-1]))
    plt.text(
        len(mean_time_steps) - 1,
        mean_time_steps[-1],
        f"{mean_time_steps[-1]:.2f}",
    )
    plt.legend()
    plt.grid()

    # Adjust the layout to prevent overlapping of subplots and display the plot
    plt.tight_layout(h_pad=2)
    plt.show(block=False)
    plt.pause(0.1)
