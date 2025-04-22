import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from tqdm import tqdm
import time

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
        self.rng = np.random.RandomState(seed)
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
            axis = self.rng.randint(0, self.dimensions)
            direction = 2 * self.rng.randint(0, 2) - 1  # Either -1 or 1
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
        seeds = np.random.RandomState(seed).randint(0, 10000, size=n_walkers)
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
                        help='File path to save the plot')
    parser.add_argument('--analyze-msd', action='store_true',
                        help='Analyze mean square displacement') 
    parser.add_argument('--save-animation', type=str, default=None,
                        help='File path to save the animation (e.g., "walk.mp4" or "walk.gif")')
    
    args = parser.parse_args()
    
    if args.animate and args.walkers > 1:
        print("Animation is only available for a single walker. Setting walkers to 1.")
        args.walkers = 1
    
    # Run simulation based on arguments
    if args.walkers == 1:
        walker = DrunkardWalk(dimensions=args.dimensions, 
                             step_type=args.step_type, 
                             seed=args.seed)
        
        if args.animate and args.dimensions == 2:
            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.grid(True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('2D Random Walk Animation')
            
            line, = ax.plot([], [], '-o', alpha=0.7, markersize=2)
            point, = ax.plot([], [], 'ro', markersize=8)
            
            # Initialize an empty path
            path = [np.zeros(2)]
            
            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point
            
            def update(frame):
                walker.step()
                path.append(np.copy(walker.current_position))
                
                # Convert path to array for plotting
                path_array = np.array(path)
                
                # Update plot data
                line.set_data(path_array[:, 0], path_array[:, 1])
                point.set_data([path_array[-1, 0]], [path_array[-1, 1]])
                
                # Adjust limits if necessary
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                if path_array[-1, 0] < x_min or path_array[-1, 0] > x_max or \
                path_array[-1, 1] < y_min or path_array[-1, 1] > y_max:
                    ax.set_xlim(min(x_min, path_array[-1, 0] - 10), max(x_max, path_array[-1, 0] + 10))
                    ax.set_ylim(min(y_min, path_array[-1, 1] - 10), max(y_max, path_array[-1, 1] + 10))
                
                return line, point
            
            ani = FuncAnimation(fig, update, frames=args.steps, 
                                init_func=init, blit=True, interval=20, 
                                repeat=False)  # Add this parameter
            
            
            # Save animation if requested
            if args.save_animation:
                print(f"Saving animation to {args.save_animation}...")
                if args.save_animation.endswith('.mp4'):
                    writer = animation.FFMpegWriter(fps=30)
                    ani.save(args.save_animation, writer=writer)
                elif args.save_animation.endswith('.gif'):
                    writer = animation.PillowWriter(fps=15)
                    ani.save(args.save_animation, writer=writer)
                else:
                    print("Animation format not recognized. Use .mp4 or .gif extension.")
            
            plt.show()
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
