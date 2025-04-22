import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

class DrunkardWalk:
    """
    A class for simulating and analyzing random walks (drunkard walks) in multiple dimensions.
    
    This implementation handles 1D, 2D, and 3D random walks with various step distributions
    and provides analytical tools to study diffusion properties.
    """
    
    def __init__(self, dimensions=2, step_type="uniform", seed=None):
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
            
        self.dimensions = dimensions
        self.step_type = step_type
        self.rng = np.random.default_rng(seed)
        self.current_position = np.zeros(dimensions)
        self.path = [np.copy(self.current_position)]
        
    def reset(self):
        """Reset the walker to the origin."""
        self.current_position = np.zeros(self.dimensions)
        self.path = [np.copy(self.current_position)]
        
    def _generate_step(self):
        """Generate a random step based on the specified distribution."""
        if self.step_type == "uniform":
            # Uniform step: randomly choose one of the 2*dimensions possible unit steps
            step = np.zeros(self.dimensions)
            axis = self.rng.integers(0, self.dimensions)
            direction = 2 * self.rng.integers(0, 2) - 1  # Either -1 or 1
            step[axis] = direction
            return step
            
        elif self.step_type == "gaussian":
            # Gaussian step: step size follows a normal distribution
            return self.rng.normal(0, 1, self.dimensions)
            
        elif self.step_type == "levy":
            # Lévy flight: occasionally take very large steps
            # Uses a Cauchy distribution (heavy-tailed)
            return self.rng.standard_cauchy(self.dimensions)
    
    def step(self):
        """Take a single step and update the current position."""
        step = self._generate_step()
        self.current_position += step
        self.path.append(np.copy(self.current_position))
        
    def walk(self, steps):
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
        for _ in range(steps):
            self.step()
        return self.current_position
        
    def get_path_array(self):
        """Return the path as a numpy array."""
        return np.array(self.path)
        
    def displacement(self):
        """Calculate the displacement from the origin."""
        return np.linalg.norm(self.current_position)
        
    def analyze_msd(self, trials=10, max_steps=1000, log_scale=True):
        """
        Analyze the mean square displacement (MSD) over multiple trials.
        
        Parameters:
        -----------
        trials : int
            Number of trials to average over
        max_steps : int
            Maximum number of steps per trial
        log_scale : bool
            Whether to sample steps on a logarithmic scale (for efficiency)
            
        Returns:
        --------
        tuple
            (steps, msd) arrays for plotting
        """
        if log_scale:
            # Sample points logarithmically to efficiently analyze diffusion behavior
            steps = np.unique(np.logspace(0, np.log10(max_steps), 50, dtype=int))
        else:
            steps = np.arange(1, max_steps + 1)
            
        msd = np.zeros(len(steps))
        
        for trial in tqdm(range(trials), desc="Analyzing diffusion"):
            self.reset()
            positions = []
            step_count = 0
            
            for target_step in steps:
                steps_to_take = target_step - step_count
                for _ in range(steps_to_take):
                    self.step()
                step_count = target_step
                positions.append(np.copy(self.current_position))
                
            squared_displacements = np.sum(np.square(positions), axis=1)
            msd += squared_displacements
            
        msd /= trials
        return steps, msd

    def plot_path(self, show=True, save_path=None):
        """
        Plot the random walk path.
        
        Parameters:
        -----------
        show : bool
            Whether to display the plot
        save_path : str or None
            File path to save the plot, or None to not save
        """
        path = self.get_path_array()
        
        plt.figure(figsize=(10, 8))
        
        if self.dimensions == 1:
            steps = np.arange(len(path))
            plt.plot(steps, path, '-o', alpha=0.6, markersize=3)
            plt.xlabel('Step')
            plt.ylabel('Position')
            plt.title('1D Random Walk')
            
        elif self.dimensions == 2:
            plt.plot(path[:, 0], path[:, 1], '-', alpha=0.7)
            plt.plot(path[0, 0], path[0, 1], 'go', label='Start', markersize=10)
            plt.plot(path[-1, 0], path[-1, 1], 'ro', label='End', markersize=10)
            plt.axis('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('2D Random Walk')
            plt.legend()
            
        elif self.dimensions == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.7)
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='g', s=100, label='Start')
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='r', s=100, label='End')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Random Walk')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()

def run_ensemble_simulation(n_walkers, steps, dimensions=2, step_type="uniform", seed=None):
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
        Random seed
        
    Returns:
    --------
    tuple
        (final_positions, displacements, mean_displacement)
    """
    if seed is not None:
        # Create different seeds for each walker
        seeds = np.random.default_rng(seed).integers(0, 10000, size=n_walkers)
    else:
        seeds = [None] * n_walkers
    
    walkers = [DrunkardWalk(dimensions=dimensions, step_type=step_type, seed=s) 
               for s in seeds]
    
    # Run all walkers
    final_positions = np.zeros((n_walkers, dimensions))
    displacements = np.zeros(n_walkers)
    
    for i, walker in enumerate(tqdm(walkers, desc="Simulating walkers")):
        walker.walk(steps)
        final_positions[i] = walker.current_position
        displacements[i] = walker.displacement()
    
    mean_displacement = np.mean(displacements)
    return final_positions, displacements, mean_displacement

def plot_ensemble_results(final_positions, displacements, dimensions):
    """Plot results from an ensemble simulation."""
    plt.figure(figsize=(16, 8))
    
    # Plot 1: Final positions
    plt.subplot(1, 2, 1)
    if dimensions == 1:
        plt.hist(final_positions.flatten(), bins=30, alpha=0.7)
        plt.xlabel('Final Position')
        plt.ylabel('Count')
    elif dimensions == 2:
        plt.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
    elif dimensions == 3:
        ax = plt.subplot(1, 2, 1, projection='3d')
        ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.title('Final Positions')
    
    # Plot 2: Displacement histogram
    plt.subplot(1, 2, 2)
    plt.hist(displacements, bins=30, alpha=0.7)
    plt.xlabel('Displacement from Origin')
    plt.ylabel('Count')
    plt.title(f'Displacement Distribution (Mean: {np.mean(displacements):.2f})')
    
    plt.tight_layout()
    plt.show()

def create_animation(walker_steps, dimensions, fps=30, show_animation=True, save_path=None):
    """
    Create and save an animation of the random walk.
    
    Parameters:
    -----------
    walker_steps : list
        List of walker positions at each step
    dimensions : int
        Number of dimensions (1, 2, or 3)
    fps : int
        Frames per second for the animation
    show_animation : bool
        Whether to display the animation
    save_path : str or None
        Path to save the animation file (must end with .mp4)
        
    Returns:
    --------
    matplotlib.animation.Animation
        The animation object
    """
    steps = np.array(walker_steps)
    
    fig = plt.figure(figsize=(10, 8))
    
    if dimensions == 1:
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], '-o', alpha=0.7, markersize=3)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        def init():
            x_range = np.arange(len(steps))
            y_min, y_max = steps.min(), steps.max()
            margin = max(1, (y_max - y_min) * 0.1)
            ax.set_xlim(0, len(steps))
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlabel('Step')
            ax.set_ylabel('Position')
            ax.set_title('1D Random Walk')
            return line, point
            
        def update(frame):
            line.set_data(range(frame+1), steps[:frame+1])
            point.set_data([frame], [steps[frame]])
            return line, point
            
    elif dimensions == 2:
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], '-', alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=8)
        start, = ax.plot([], [], 'go', markersize=10, label='Start')
        
        x_min, x_max = steps[:, 0].min(), steps[:, 0].max()
        y_min, y_max = steps[:, 1].min(), steps[:, 1].max()
        
        # Add some margin
        margin_x = max(1, (x_max - x_min) * 0.1)
        margin_y = max(1, (y_max - y_min) * 0.1)
        
        def init():
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('2D Random Walk')
            ax.grid(True)
            start.set_data([steps[0, 0]], [steps[0, 1]])
            ax.legend()
            return line, point, start
            
        def update(frame):
            line.set_data(steps[:frame+1, 0], steps[:frame+1, 1])
            point.set_data([steps[frame, 0]], [steps[frame, 1]])
            return line, point, start
            
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], '-', alpha=0.7)
        point = ax.scatter([], [], [], color='r', s=50)
        start = ax.scatter([], [], [], color='g', s=100, label='Start')
        
        x_min, x_max = steps[:, 0].min(), steps[:, 0].max()
        y_min, y_max = steps[:, 1].min(), steps[:, 1].max()
        z_min, z_max = steps[:, 2].min(), steps[:, 2].max()
        
        # Add some margin
        margin_x = max(1, (x_max - x_min) * 0.1)
        margin_y = max(1, (y_max - y_min) * 0.1)
        margin_z = max(1, (z_max - z_min) * 0.1)
        
        def init():
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)
            ax.set_zlim(z_min - margin_z, z_max + margin_z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Random Walk')
            start._offsets3d = ([steps[0, 0]], [steps[0, 1]], [steps[0, 2]])
            ax.legend()
            return line, point, start
            
        def update(frame):
            line.set_data(steps[:frame+1, 0], steps[:frame+1, 1])
            line.set_3d_properties(steps[:frame+1, 2])
            point._offsets3d = ([steps[frame, 0]], [steps[frame, 1]], [steps[frame, 2]])
            return line, point, start
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(steps),
                                  init_func=init, blit=True, interval=1000/fps,
                                  repeat=False)
    
    # Save the animation if a path is provided
    if save_path:
        # Make sure we're using a supported codec
        writer = FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    if show_animation:
        plt.show()
    else:
        plt.close()
        
    return anim

def main():
    """Main function to parse arguments and run the simulation."""

    parser = argparse.ArgumentParser(description='Drunkard Walk Simulation')
    parser.add_argument('--dimensions', type=int, default=2, choices=[1, 2, 3],
                        help='Number of dimensions (1, 2, or 3)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of steps to take')
    parser.add_argument('--walkers', type=int, default=1,
                        help='Number of walkers (for ensemble simulation)')
    parser.add_argument('--step-type', type=str, default='uniform', 
                        choices=['uniform', 'gaussian', 'levy'],
                        help='Type of step distribution')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--animate', action='store_true',
                        help='Animate the walk (only for single walker)')
    parser.add_argument('--save', type=str, default=None,
                        help='File path to save the plot or animation')
    parser.add_argument('--analyze-msd', action='store_true',
                        help='Analyze mean square displacement')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the animation')
    parser.add_argument('--record-animation', action='store_true',
                        help='Record the animation to a file')
    
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    
    if args.walkers <= 0:
        raise ValueError("Walkers must be a positive integer.")
    
    if args.animate and args.walkers > 1:
        print("Animation is only available for a single walker. Setting walkers to 1.")
        args.walkers = 1
    
    # Add default save path for animation if recording is requested but no path provided
    if args.record_animation and not args.save:
        args.save = f"drunkard_walk_{args.dimensions}d_{args.step_type}_{args.steps}_steps.mp4"
        print(f"No save path provided, will save animation to {args.save}")
    
    # Run simulation based on arguments
    if args.walkers == 1:
        walker = DrunkardWalk(dimensions=args.dimensions, 
                             step_type=args.step_type, 
                             seed=args.seed)
        
        if args.animate or args.record_animation:
            # Run the simulation first to collect all positions
            for _ in tqdm(range(args.steps), desc="Simulating walk"):
                walker.step()
                
            walker_steps = walker.get_path_array()
            
            # Create the animation
            create_animation(
                walker_steps, 
                args.dimensions, 
                fps=args.fps, 
                show_animation=args.animate,
                save_path=args.save if args.record_animation else None
            )
            
            print(f"Final position: {walker.current_position}")
            print(f"Displacement from origin: {walker.displacement():.2f}")
        else:
            # Non-animated
            walker.walk(args.steps)
            walker.plot_path(save_path=args.save)
            print(f"Final position: {walker.current_position}")
            print(f"Displacement from origin: {walker.displacement():.2f}")
            
        if args.analyze_msd:
            steps, msd = walker.analyze_msd(trials=10, max_steps=args.steps)
            
            plt.figure(figsize=(10, 6))
            plt.loglog(steps, msd, 'o-')
            plt.loglog(steps, steps, 'k--', label='Linear (normal diffusion)')
            if args.step_type == 'levy':
                plt.loglog(steps, np.power(steps, 1.5), 'r--', label='Super-diffusion')
            plt.xlabel('Steps')
            plt.ylabel('Mean Square Displacement')
            plt.title('Diffusion Analysis')
            plt.legend()
            plt.grid(True)
            plt.show()
            
    else:
        # Ensemble simulation
        start_time = time.time()
        final_positions, displacements, mean_disp = run_ensemble_simulation(
            args.walkers, args.steps, args.dimensions, args.step_type, args.seed)
        end_time = time.time()
        
        print(f"Simulated {args.walkers} walkers for {args.steps} steps in {end_time - start_time:.2f} seconds")
        print(f"Mean displacement: {mean_disp:.2f}")
        print(f"Expected displacement (√n): {np.sqrt(args.steps):.2f}")
        
        plot_ensemble_results(final_positions, displacements, args.dimensions)

if __name__ == "__main__":
    main()
