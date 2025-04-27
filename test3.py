# Previous code remains the same until create_animation...

import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from typing import Optional, Tuple, List, Union # Added for type hints
import os # Added for path handling

# Define type aliases for clarity
Position = np.ndarray # Type alias for a position vector
Path = List[Position] # Type alias for a list of positions
PathArray = np.ndarray # Type alias for path represented as numpy array

# DrunkardWalk class definition (no changes needed here)
class DrunkardWalk:
    """
    A class for simulating and analyzing random walks (drunkard walks) in multiple dimensions.

    This implementation handles 1D, 2D, and 3D random walks with various step distributions
    and provides analytical tools to study diffusion properties.
    """

    def __init__(self, dimensions: int = 2, step_type: str = "uniform", seed: Optional[int] = None):
        """
        Initialize the random walk simulator.

        Parameters:
        -----------
        dimensions : int
            Number of dimensions for the walk (1, 2, or 3)
        step_type : str
            Type of step distribution ("uniform", "gaussian", or "levy")
        seed : int or None
            Random seed for reproducibility
        """
        if dimensions not in [1, 2, 3]:
            raise ValueError("Dimensions must be 1, 2, or 3")

        if step_type not in ["uniform", "gaussian", "levy"]:
            raise ValueError("Step type must be 'uniform', 'gaussian', or 'levy'")

        self.dimensions: int = dimensions
        self.step_type: str = step_type
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.current_position: Position = np.zeros(dimensions)
        self.path: Path = [np.copy(self.current_position)] # Store path as list of numpy arrays

    def reset(self) -> None:
        """Reset the walker to the origin."""
        self.current_position = np.zeros(self.dimensions)
        self.path = [np.copy(self.current_position)]

    def _generate_step(self) -> Position:
        """Generate a random step based on the specified distribution."""
        if self.step_type == "uniform":
            # Uniform step: randomly choose one of the 2*dimensions possible unit steps
            step = np.zeros(self.dimensions)
            axis = self.rng.integers(0, self.dimensions)
            direction = self.rng.choice([-1, 1]) # Simpler way to get -1 or 1
            step[axis] = direction
            return step

        elif self.step_type == "gaussian":
            # Gaussian step: step size follows a normal distribution (mean=0, std=1)
            return self.rng.normal(0, 1, self.dimensions)

        elif self.step_type == "levy":
            # Lévy flight: using a Cauchy distribution (alpha=1 symmetric stable distribution)
            # This results in occasional very large steps (heavy tails).
            return self.rng.standard_cauchy(self.dimensions)
        else:
             # This case should not be reached due to __init__ check, but added for safety
            raise ValueError(f"Unknown step_type: {self.step_type}")

    def step(self) -> None:
        """Take a single step and update the current position and path."""
        step = self._generate_step()
        self.current_position += step
        self.path.append(np.copy(self.current_position))

    def walk(self, steps: int) -> Position:
        """
        Perform a random walk for the specified number of steps.

        Parameters:
        -----------
        steps : int
            Number of steps to take

        Returns:
        --------
        np.ndarray
            The final position after the walk
        """
        if steps < 0:
             raise ValueError("Number of steps cannot be negative.")
        for _ in range(steps):
            self.step()
        return self.current_position

    def get_path_array(self) -> PathArray:
        """Return the path as a numpy array."""
        return np.array(self.path)

    def displacement(self) -> float:
        """Calculate the final displacement (Euclidean distance) from the origin."""
        return np.linalg.norm(self.current_position)

    def analyze_msd(self, trials: int = 50, max_steps: int = 1000, log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze the mean square displacement (MSD) over multiple trials.

        Parameters:
        -----------
        trials : int
            Number of independent trials to average over (default: 50)
        max_steps : int
            Maximum number of steps to simulate in each trial
        log_scale : bool
            Whether to sample step counts on a logarithmic scale for plotting (more efficient for diffusion analysis)

        Returns:
        --------
        tuple
            (step_counts, msd_values) arrays for plotting
        """
        if max_steps <= 0:
             raise ValueError("max_steps must be positive for MSD analysis.")
        if trials <= 0:
             raise ValueError("Number of trials must be positive.")

        if log_scale:
            # Sample points logarithmically to efficiently analyze diffusion behavior
            # Ensure at least 2 points and include max_steps
            num_points = min(50, max_steps) # Limit number of points for efficiency
            step_counts = np.unique(np.logspace(0, np.log10(max_steps), num_points, dtype=int))
            if len(step_counts) > 0 and step_counts[0] == 0 and max_steps > 0 :
                 if len(step_counts) > 1: step_counts = step_counts[1:] # Remove 0 if other steps exist
                 else: step_counts = np.array([1]) # If only 0, replace with 1
            if len(step_counts) == 0 or step_counts[-1] < max_steps: # Make sure the last step is included
                step_counts = np.append(step_counts, max_steps)
                step_counts = np.unique(step_counts) # Remove potential duplicates
            if len(step_counts) > 0 and step_counts[0] == 0 : step_counts = step_counts[1:] # Final check for step 0
            if len(step_counts) == 0: # Handle edge case where max_steps was 0 or 1
                 step_counts = np.array([max_steps]) if max_steps > 0 else np.array([1])

        else:
            step_counts = np.arange(1, max_steps + 1)

        if len(step_counts) == 0:
            print("Warning: No steps selected for MSD analysis.")
            return np.array([]), np.array([])


        # Store squared displacements for each trial at each step_count
        all_squared_displacements = np.zeros((trials, len(step_counts)))
        actual_max_steps = step_counts[-1] # The true maximum steps needed per trial

        original_seed = self.rng # Store original RNG state if needed, though default_rng handles seeding well

        for trial in tqdm(range(trials), desc="Analyzing diffusion"):
            # Create a new walker for each trial to ensure independence (if needed, or reset carefully)
            # Using the same instance but resetting is usually fine if state is fully reset.
            temp_walker = DrunkardWalk(dimensions=self.dimensions, step_type=self.step_type, seed=None) # Use independent seeds

            # Simulate one full walk for this trial up to the maximum step count needed
            # Pre-allocate path array for this trial for efficiency
            trial_path = np.zeros((actual_max_steps + 1, self.dimensions))
            for i in range(actual_max_steps):
                temp_walker.step()
                trial_path[i+1] = temp_walker.current_position

            # Extract squared displacements at the desired step counts
            # step_counts are 1-based, trial_path is 0-indexed (index 0 is origin)
            positions_at_steps = trial_path[step_counts]
            squared_displacements_trial = np.sum(np.square(positions_at_steps), axis=1)

            all_squared_displacements[trial, :] = squared_displacements_trial

        # Average over trials
        msd_values = np.mean(all_squared_displacements, axis=0)

        # Check for NaNs or Infs which might occur with Levy flights if displacements get huge
        if np.any(np.isnan(msd_values)) or np.any(np.isinf(msd_values)):
            print("\nWarning: NaNs or Infs encountered in MSD calculation. This might happen with heavy-tailed distributions (like Levy) if displacements become extremely large. Consider reducing max_steps or increasing trials.")
            # Optional: handle or filter NaNs/Infs if needed for plotting
            # msd_values = np.nan_to_num(msd_values, nan=0.0, posinf=0.0, neginf=0.0)


        return step_counts, msd_values

    def plot_path(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot the random walk path.

        Parameters:
        -----------
        show : bool
            Whether to display the plot
        save_path : str or None
            File path to save the plot, or None to not save
        """
        path_array = self.get_path_array()
        if len(path_array) <= 1:
            print("Warning: Path has 0 or 1 steps, cannot plot.")
            return

        fig = plt.figure(figsize=(10, 8)) # Create the figure once

        if self.dimensions == 1:
            ax = fig.add_subplot(111)
            steps_axis = np.arange(len(path_array))
            ax.plot(steps_axis, path_array[:, 0], '-o', alpha=0.6, markersize=3)
            ax.set_xlabel('Step')
            ax.set_ylabel('Position')
            ax.set_title(f'1D Random Walk ({self.step_type})')
            ax.grid(True)

        elif self.dimensions == 2:
            ax = fig.add_subplot(111)
            ax.plot(path_array[:, 0], path_array[:, 1], '-', alpha=0.7, label='Path')
            ax.plot(path_array[0, 0], path_array[0, 1], 'go', label='Start', markersize=10)
            ax.plot(path_array[-1, 0], path_array[-1, 1], 'ro', label='End', markersize=10)
            ax.axis('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'2D Random Walk ({self.step_type})')
            ax.legend(loc='upper right') # Explicit legend location
            ax.grid(True)

        elif self.dimensions == 3:
            # Use the figure created above
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], alpha=0.7, label='Path')
            ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], c='g', s=100, label='Start', depthshade=True)
            ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], c='r', s=100, label='End', depthshade=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Random Walk ({self.step_type})')

            # --- Improved 3D Axis Scaling ---
            x_min, x_max = path_array[:, 0].min(), path_array[:, 0].max()
            y_min, y_max = path_array[:, 1].min(), path_array[:, 1].max()
            z_min, z_max = path_array[:, 2].min(), path_array[:, 2].max()
            # Calculate the ranges and midpoint
            max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max()
            # Handle case where range is zero (e.g., single point)
            if max_range == 0: max_range = 1.0
            mid_x = (x_max + x_min) * 0.5
            mid_y = (y_max + y_min) * 0.5
            mid_z = (z_max + z_min) * 0.5
            # Set cubic limits centered on the midpoint with a margin
            margin_factor = 1.2 # Use 20% margin
            ax.set_xlim(mid_x - max_range/2 * margin_factor, mid_x + max_range/2 * margin_factor)
            ax.set_ylim(mid_y - max_range/2 * margin_factor, mid_y + max_range/2 * margin_factor)
            ax.set_zlim(mid_z - max_range/2 * margin_factor, mid_z + max_range/2 * margin_factor)
            # --- End Improved Scaling ---

            ax.legend(loc='upper right') # Explicit legend location


        fig.tight_layout()

        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            except Exception as e:
                 print(f"Error saving plot to {save_path}: {e}")


        if show:
            plt.show()
        else:
            plt.close(fig) # Close the specific figure

# run_ensemble_simulation function (no changes needed here)
def run_ensemble_simulation(n_walkers: int, steps: int, dimensions: int = 2, step_type: str = "uniform", seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate multiple independent random walkers and analyze their collective behavior.

    Parameters:
    -----------
    n_walkers : int
        Number of walkers in the ensemble
    steps : int
        Number of steps each walker takes
    dimensions : int
        Number of dimensions for the walks
    step_type : str
        Type of step distribution
    seed : int or None
        Base random seed for generating walker seeds

    Returns:
    --------
    tuple
        (final_positions, displacements, mean_displacement)
    """
    if n_walkers <= 0:
        raise ValueError("Number of walkers must be positive.")
    if steps < 0:
        raise ValueError("Number of steps cannot be negative.")

    master_rng = np.random.default_rng(seed)
    # Generate independent seeds for each walker more robustly using SeedSequence
    seed_sequence = np.random.SeedSequence(master_rng.integers(2**31))
    walker_seeds = seed_sequence.spawn(n_walkers)

    walkers = [DrunkardWalk(dimensions=dimensions, step_type=step_type, seed=s)
               for s in walker_seeds]

    final_positions = np.zeros((n_walkers, dimensions))
    displacements = np.zeros(n_walkers)

    # Using parallel processing here could speed things up significantly for large n_walkers
    # (e.g., using multiprocessing.Pool), but keeping it sequential for simplicity.
    for i, walker in enumerate(tqdm(walkers, desc="Simulating walkers")):
        final_pos = walker.walk(steps)
        # Handle potential overflows if steps are huge (especially Levy)
        if np.any(np.isinf(final_pos)) or np.any(np.isnan(final_pos)):
             print(f"\nWarning: Walker {i} encountered Inf/NaN position. Displacement calculation may be affected.")
             final_positions[i] = np.nan # Mark as NaN
             displacements[i] = np.nan
        else:
            final_positions[i] = final_pos
            displacements[i] = walker.displacement()

    # Calculate mean displacement ignoring potential NaNs
    mean_displacement = np.nanmean(displacements)
    return final_positions, displacements, mean_displacement

# plot_ensemble_results function (no changes needed here)
def plot_ensemble_results(final_positions: np.ndarray, displacements: np.ndarray, dimensions: int) -> None:
    """Plot results from an ensemble simulation."""
    # Filter out NaN values that might occur from overflows
    valid_mask = ~np.isnan(displacements) & ~np.any(np.isnan(final_positions), axis=1) # Also check positions
    valid_displacements = displacements[valid_mask]
    valid_final_positions = final_positions[valid_mask, :]
    num_valid = len(valid_displacements)
    num_total = len(displacements)
    if num_valid < num_total:
        print(f"Warning: Plotting results for {num_valid}/{num_total} walkers due to NaN values.")
    if num_valid == 0:
        print("Error: No valid walker data to plot.")
        return


    fig = plt.figure(figsize=(16, 8))

    # Plot 1: Final positions
    ax1 = fig.add_subplot(1, 2, 1, projection='3d' if dimensions == 3 else None)
    plot_alpha = max(0.05, min(0.7, 500 / num_valid)) if num_valid > 0 else 0.7 # Adjust alpha based on points
    plot_size = max(1, min(5, 50 / np.sqrt(num_valid))) if num_valid > 0 else 5 # Adjust size

    if dimensions == 1:
        ax1.hist(valid_final_positions.flatten(), bins=min(50, max(10, num_valid // 10)), alpha=0.7)
        ax1.set_xlabel('Final Position')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Final Positions ({num_valid} Walkers)')
        ax1.grid(True, axis='y')
    elif dimensions == 2:
        ax1.scatter(valid_final_positions[:, 0], valid_final_positions[:, 1], alpha=plot_alpha, s=plot_size)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')
        ax1.set_title(f'Final Positions ({num_valid} Walkers)')
        ax1.grid(True)
    elif dimensions == 3:
        ax1.scatter(valid_final_positions[:, 0], valid_final_positions[:, 1], valid_final_positions[:, 2], alpha=plot_alpha, s=plot_size)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Final Positions ({num_valid} Walkers)')
        # Equal aspect ratio is hard in 3D, auto-scaling is usually better
        # ax1.set_aspect('equal')

    # Plot 2: Displacement histogram
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(valid_displacements, bins=min(50, max(10, num_valid // 10)), alpha=0.7)
    ax2.set_xlabel('Displacement from Origin')
    ax2.set_ylabel('Count')
    mean_disp = np.mean(valid_displacements)
    std_disp = np.std(valid_displacements)
    ax2.set_title(f'Displacement Distribution ({num_valid} Walkers)\nMean: {mean_disp:.3f}, Std: {std_disp:.3f}')
    ax2.grid(True, axis='y')

    fig.tight_layout()
    plt.show()

# ==============================================
# Modified create_animation function starts here
# ==============================================
def create_animation(walker_path: PathArray, dimensions: int, step_type: str, fps: int = 30, show_animation: bool = True, save_path: Optional[str] = None, blit: bool = True) -> Optional[animation.FuncAnimation]:
    """
    Create and potentially save/show an animation of the random walk.

    Parameters:
    -----------
    walker_path : np.ndarray
        Array of walker positions at each step (shape: [num_steps+1, dimensions])
    dimensions : int
        Number of dimensions (1, 2, or 3)
    step_type : str
        Type of step distribution (for title)
    fps : int
        Frames per second for the animation
    show_animation : bool
        Whether to display the animation interactively
    save_path : str or None
        Path to save the animation file (e.g., 'walk.mp4')
    blit : bool
        Whether to use blitting for optimization (can cause issues, especially 3D)


    Returns:
    --------
    matplotlib.animation.FuncAnimation or None
        The animation object, or None if path is too short.
    """
    num_frames = len(walker_path)
    if num_frames <= 1:
        print("Warning: Path too short for animation.")
        return None

    fig = plt.figure(figsize=(10, 8))
    # Use a simple static title - updating title in animation is slow
    static_title = f'{dimensions}D Random Walk ({step_type}, {num_frames-1} steps)'

    # --- Axis limit margin ---
    margin_factor = 1.2 # Use 20% margin beyond data range

    if dimensions == 1:
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], '-o', alpha=0.7, markersize=3, label='Path') # Path line
        point, = ax.plot([], [], 'ro', markersize=8, label='Current') # Current position marker
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, ha='left', va='top') # Step counter text

        # Determine fixed axes limits
        steps_axis = np.arange(num_frames)
        y_min, y_max = walker_path[:, 0].min(), walker_path[:, 0].max()
        y_range = y_max - y_min if y_max > y_min else 1.0 # Handle single point case
        y_mid = (y_max + y_min) * 0.5
        ax.set_xlim(0, num_frames -1)
        ax.set_ylim(y_mid - y_range/2 * margin_factor, y_mid + y_range/2 * margin_factor)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            ax.set_xlabel('Step')
            ax.set_ylabel('Position')
            ax.set_title(static_title)
            ax.grid(True)
            # Place legend, avoid top-left where text is
            ax.legend(loc='upper right')
            time_text.set_text('')
            return line, point, time_text # Return all animated artists

        def update(frame):
            line.set_data(steps_axis[:frame+1], walker_path[:frame+1, 0])
            point.set_data([frame], [walker_path[frame, 0]])
            time_text.set_text(f'Step: {frame}')
            return line, point, time_text

    elif dimensions == 2:
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], '-', alpha=0.7, label='Path') # Path line
        point, = ax.plot([], [], 'ro', markersize=8, label='Current') # Current position
        start, = ax.plot([], [], 'go', markersize=10, label='Start') # Start marker
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, ha='left', va='top') # Step counter text

        # Determine fixed axes limits
        x_min, x_max = walker_path[:, 0].min(), walker_path[:, 0].max()
        y_min, y_max = walker_path[:, 1].min(), walker_path[:, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        x_mid = (x_max + x_min) * 0.5
        y_mid = (y_max + y_min) * 0.5
        ax.set_xlim(x_mid - x_range/2 * margin_factor, x_mid + x_range/2 * margin_factor)
        ax.set_ylim(y_mid - y_range/2 * margin_factor, y_mid + y_range/2 * margin_factor)
        ax.axis('equal') # Keep aspect ratio equal

        def init():
            line.set_data([], [])
            point.set_data([], [])
            start.set_data([walker_path[0, 0]], [walker_path[0, 1]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(static_title)
            ax.grid(True)
            ax.legend(loc='upper right') # Explicitly set legend location
            time_text.set_text('')
            return line, point, start, time_text

        def update(frame):
            # Optimize update by only updating necessary data
            line.set_data(walker_path[:frame+1, 0], walker_path[:frame+1, 1])
            point.set_data([walker_path[frame, 0]], [walker_path[frame, 1]])
            time_text.set_text(f'Step: {frame}')
            return line, point, start, time_text

    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], '-', alpha=0.7, label='Path') # Path line
        point = ax.scatter([], [], [], color='r', s=50, label='Current', depthshade=True)
        start = ax.scatter([], [], [], color='g', s=100, label='Start', depthshade=True)
        # Place text in 3D using text2D
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, ha='left', va='top')

        # --- Determine fixed cubic axes limits ---
        x_min, x_max = walker_path[:, 0].min(), walker_path[:, 0].max()
        y_min, y_max = walker_path[:, 1].min(), walker_path[:, 1].max()
        z_min, z_max = walker_path[:, 2].min(), walker_path[:, 2].max()
        # Calculate the ranges and midpoint
        ranges = np.array([x_max-x_min, y_max-y_min, z_max-z_min])
        max_range = ranges.max()
        # Handle case where range is zero (e.g., single point or straight line)
        if max_range == 0: max_range = 1.0
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        # Set cubic limits centered on the midpoint with margin
        ax.set_xlim(mid_x - max_range/2 * margin_factor, mid_x + max_range/2 * margin_factor)
        ax.set_ylim(mid_y - max_range/2 * margin_factor, mid_y + max_range/2 * margin_factor)
        ax.set_zlim(mid_z - max_range/2 * margin_factor, mid_z + max_range/2 * margin_factor)
        # --- End axis limits ---

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point._offsets3d = ([], [], []) # Reset scatter data
            start._offsets3d = ([walker_path[0, 0]], [walker_path[0, 1]], [walker_path[0, 2]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(static_title)
            ax.legend(loc='upper right') # Explicitly set legend location
            time_text.set_text('')
            # Blitting is often problematic with 3D scatters, return necessary artists
            return line, point, start, time_text

        def update(frame):
            # Update line efficiently
            line.set_data(walker_path[:frame+1, 0], walker_path[:frame+1, 1])
            line.set_3d_properties(walker_path[:frame+1, 2])
            # Update scatter plot position
            point._offsets3d = ([walker_path[frame, 0]], [walker_path[frame, 1]], [walker_path[frame, 2]])
            time_text.set_text(f'Step: {frame}')
            # Return all artists that change
            return line, point, start, time_text

    else:
        raise ValueError("Animation supports only 1, 2, or 3 dimensions.")

    # Create the animation
    try:
        # Disable blit for 3D as it's often unreliable with scatter updates
        blit_effective = blit if dimensions != 3 else False
        print(f"Creating animation (blit={'ON' if blit_effective else 'OFF'})...")
        anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                      init_func=init, blit=blit_effective, interval=max(1, 1000//fps),
                                      repeat=False)

        # Save the animation if a path is provided
        if save_path:
            try:
                # Ensure the directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir: os.makedirs(save_dir, exist_ok=True) # Only make dirs if path includes one
                # Choose a writer (FFMpeg is common)
                writer = FFMpegWriter(fps=fps)
                print(f"Saving animation to {save_path} (using {writer.__class__.__name__}, this may take a while)...")
                anim.save(save_path, writer=writer, dpi=150) # Lower dpi for faster save?
                print(f"Animation successfully saved.")
            except FileNotFoundError:
                print("\nError: 'ffmpeg' command not found.")
                print("Please install ffmpeg to save animations.")
                print("  - On Debian/Ubuntu: sudo apt install ffmpeg")
                print("  - On macOS (using Homebrew): brew install ffmpeg")
                print("  - On Windows: Download from ffmpeg.org and add to PATH.\n")
            except Exception as e:
                print(f"\nError saving animation: {e}")
                print("Ensure ffmpeg is installed correctly and working.")
                print(f"Try running with --no-blit if blitting was enabled ({blit_effective}).\n")


        if show_animation:
            print("Showing animation...")
            plt.show()
        else:
            # If not showing and not saving, animation object isn't used.
            # If saving, we need to keep the figure open until save completes.
            # Closing only if not shown seems correct.
            if not save_path:
                 plt.close(fig) # Close the figure if not shown AND not saved

        # Explicitly clear figure memory after showing/saving
        plt.close(fig)
        return anim

    except Exception as e:
        print(f"Error creating animation object: {e}")
        plt.close(fig) # Ensure figure is closed on error
        return None
# ============================================
# End of modified create_animation function
# ============================================


# plot_msd function (no changes needed here)
def plot_msd(steps: np.ndarray, msd: np.ndarray, step_type: str, dimensions: int, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plots the Mean Square Displacement (MSD) vs steps."""
    if len(steps) == 0 or len(msd) == 0:
        print("Warning: No data provided for MSD plot.")
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.loglog(steps, msd, 'o-', label=f'Simulated MSD ({step_type})')

    # Plot theoretical lines for comparison
    # Normal diffusion (Brownian motion like): MSD ~ t
    # Scale theoretical lines to match the first data point roughly (if available)
    scale_factor = (msd[0]/steps[0]) if steps[0] > 0 else (msd[1]/steps[1] if len(steps)>1 and steps[1]>0 else 1.0)

    ax.loglog(steps, steps * scale_factor, 'k--', alpha=0.7, label=r'MSD $\propto t$ (Normal)')

    # Super-diffusion example (relevant for Levy flights): MSD ~ t^alpha with alpha > 1
    if step_type == 'levy' or np.any(msd > steps * scale_factor * 1.1): # Show super-diffusion lines if relevant
         valid_steps = steps[steps > 0]
         if len(valid_steps) > 0:
             power_1_5 = np.power(valid_steps.astype(np.float64), 1.5) * scale_factor
             ax.loglog(valid_steps, power_1_5, 'r--', alpha=0.7, label=r'MSD $\propto t^{1.5}$')


    ax.set_xlabel('Number of Steps (t)')
    ax.set_ylabel('Mean Square Displacement (MSD)')
    ax.set_title(f'Diffusion Analysis ({dimensions}D, {step_type})')
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5) # Grid on both major and minor ticks in log scale
    fig.tight_layout()

    if save_path:
         try:
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MSD plot saved to {save_path}")
         except Exception as e:
            print(f"Error saving MSD plot to {save_path}: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# main function (no changes needed here, uses updated create_animation)
def main():
    """Main function to parse arguments and run the simulation."""

    parser = argparse.ArgumentParser(
        description='Simulate and analyze Drunkard Walks (Random Walks) in 1D, 2D, or 3D.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
        )
    parser.add_argument('-d', '--dimensions', type=int, default=2, choices=[1, 2, 3],
                        help='Number of spatial dimensions for the walk.')
    parser.add_argument('-n', '--steps', type=int, default=1000,
                        help='Number of steps for each walker.')
    parser.add_argument('-w', '--walkers', type=int, default=1,
                        help='Number of walkers to simulate (ensemble simulation if > 1).')
    parser.add_argument('-t', '--step-type', type=str, default='uniform',
                        choices=['uniform', 'gaussian', 'levy'],
                        help='Type of step distribution: "uniform" (unit step along random axis), '
                             '"gaussian" (step components from N(0,1)), '
                             '"levy" (step components from Cauchy distribution).')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility. If None, simulation is not repeatable.')
    parser.add_argument('--animate', action='store_true',
                        help='Show an animation of the walk (only for single walker).')
    parser.add_argument('--record-animation', type=str, default=None, metavar='FILE.mp4',
                        help='Save animation of the walk to a file (e.g., walk.mp4). Implies --walkers=1.')
    parser.add_argument('--save-plot', type=str, default=None, metavar='FILE.png',
                        help='Save the final path plot (single walker) or ensemble plot to a file.')
    parser.add_argument('--analyze-msd', action='store_true',
                        help='Perform and plot Mean Square Displacement (MSD) analysis.')
    parser.add_argument('--msd-trials', type=int, default=50,
                        help='Number of trials for MSD analysis.')
    parser.add_argument('--msd-max-steps', type=int, default=None,
                        help='Maximum steps for MSD analysis (defaults to main --steps value).')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for animation.')
    parser.add_argument('--no-log-steps', action='store_true',
                         help='Use linear step sampling for MSD analysis instead of logarithmic.')
    parser.add_argument('--no-blit', action='store_true',
                         help='Disable blitting optimization in animation (may fix rendering issues).')


    args = parser.parse_args()

    # --- Input Validation ---
    if args.steps <= 0:
        parser.error("Number of steps must be positive.")
    if args.walkers <= 0:
        parser.error("Number of walkers must be positive.")
    if args.msd_trials <= 0:
         parser.error("Number of MSD trials must be positive.")
    if args.fps <= 0:
         parser.error("Animation FPS must be positive.")

    if (args.animate or args.record_animation) and args.walkers > 1:
        print("Warning: Animation is only available for a single walker. Setting walkers to 1.")
        args.walkers = 1
    # If recording, force walkers to 1
    if args.record_animation and args.walkers > 1:
         print("Warning: Recording animation forces walkers to 1.")
         args.walkers = 1


    # Default MSD max steps to main steps if not specified
    msd_max_steps = args.msd_max_steps if args.msd_max_steps is not None else args.steps
    if args.analyze_msd and msd_max_steps <= 0:
        parser.error("--msd-max-steps must be positive if specified.")


    # --- Simulation ---
    start_time = time.time()

    if args.walkers == 1:
        # --- Single Walker Simulation ---
        print(f"Simulating 1 walker for {args.steps} steps in {args.dimensions}D ({args.step_type})...")
        walker = DrunkardWalk(dimensions=args.dimensions,
                             step_type=args.step_type,
                             seed=args.seed)

        # Simulate the walk fully first
        # We always need the full path for animation or just plotting the final path
        walker.walk(args.steps)
        walker_path_array = walker.get_path_array() # Get path after full walk
        print(f"Simulation finished. Final position: {walker.current_position}")
        print(f"Final displacement from origin: {walker.displacement():.3f}")


        # --- Animation ---
        anim_obj = None # Keep track of animation object
        if args.animate or args.record_animation:
            anim_obj = create_animation(
                walker_path=walker_path_array,
                dimensions=args.dimensions,
                step_type=args.step_type,
                fps=args.fps,
                show_animation=args.animate, # Only show if --animate is true
                save_path=args.record_animation,
                blit=not args.no_blit
            )
        # --- Plot final path if not animating/recording OR if saving plot explicitly ---
        elif args.save_plot: # Only plot separately if not animating and save_plot requested
             print("Plotting final path...")
             walker.plot_path(show=False, save_path=args.save_plot) # Don't show, just save
        elif not args.analyze_msd: # If nothing else is happening, show the plot
             print("Plotting final path...")
             walker.plot_path(show=True, save_path=None)


        # --- MSD Analysis (Single Walker - potentially noisy) ---
        if args.analyze_msd:
             print(f"\nAnalyzing MSD over {args.msd_trials} trials up to {msd_max_steps} steps...")
             # Note: analyze_msd runs its own simulations
             msd_steps, msd_values = walker.analyze_msd(
                 trials=args.msd_trials,
                 max_steps=msd_max_steps,
                 log_scale=not args.no_log_steps
             )
             print("MSD analysis finished.")

             msd_save_path = None
             # Derive MSD save path from --save-plot if provided
             if args.save_plot:
                 base, ext = os.path.splitext(args.save_plot)
                 msd_save_path = f"{base}_msd{ext}"
             # Derive MSD save path from --record-animation if provided and --save-plot wasn't
             elif args.record_animation and not args.save_plot:
                  base, ext = os.path.splitext(args.record_animation)
                  msd_save_path = f"{base}_msd.png" # Save as png

             plot_msd(msd_steps, msd_values, args.step_type, args.dimensions, save_path=msd_save_path, show=True) # Always show MSD plot for now


    else:
        # --- Ensemble Simulation ---
        print(f"Simulating ensemble of {args.walkers} walkers for {args.steps} steps in {args.dimensions}D ({args.step_type})...")
        final_positions, displacements, mean_disp = run_ensemble_simulation(
            n_walkers=args.walkers,
            steps=args.steps,
            dimensions=args.dimensions,
            step_type=args.step_type,
            seed=args.seed
        )

        print(f"\nEnsemble simulation finished.")
        # Check how many valid results we got
        num_valid = np.sum(~np.isnan(displacements))
        if num_valid == 0:
             print("Error: All walkers resulted in invalid (NaN/Inf) positions or displacements.")
        else:
            print(f"Mean displacement (from {num_valid} valid walkers): {mean_disp:.3f}")
            # Theoretical expectation for simple random walk (uniform/gaussian steps) is sqrt(steps * dimensions_factor)
            if args.step_type == 'uniform':
                 expected_rms_disp = np.sqrt(args.steps)
                 print(f"Theoretical RMS displacement (uniform steps) ≈ sqrt(steps) = {expected_rms_disp:.3f}")
            elif args.step_type == 'gaussian':
                expected_rms_disp = np.sqrt(args.dimensions * args.steps)
                print(f"Theoretical RMS displacement (gaussian steps) ≈ sqrt(d*steps) = {expected_rms_disp:.3f}")

            # Plot ensemble results
            plot_ensemble_results(final_positions, displacements, args.dimensions)
            if args.save_plot:
                try:
                    # Ensure directory exists
                    save_dir = os.path.dirname(args.save_plot)
                    if save_dir: os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
                    print(f"Ensemble plot saved to {args.save_plot}")
                except Exception as e:
                    print(f"Error saving ensemble plot to {args.save_plot}: {e}")


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    # Explicitly clean up plots at the very end if any are lingering
    plt.close('all')


if __name__ == "__main__":
    main()