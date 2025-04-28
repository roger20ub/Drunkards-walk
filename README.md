# Drunkard Walk Simulation

This Python script simulates and analyzes random walks (also known as drunkard walks) in one, two, or three dimensions. It supports different step distributions and provides tools for visualizing paths, running ensemble simulations, analyzing mean square displacement (MSD), and creating animations.

## Features

* **Dimensionality:** Simulate walks in 1D, 2D, or 3D.
* **Step Types:**
    * `uniform`: Unit steps along a randomly chosen axis.
    * `gaussian`: Step components drawn from a standard normal distribution (N(0, 1)).
    * `levy`: Step components drawn from a Cauchy distribution, simulating heavy-tailed Lévy flights.
* **Single Walker Simulation:** Simulate and visualize the path of a single drunkard.
* **Ensemble Simulation:** Run multiple independent walkers to study collective behavior, analyze final positions, and calculate mean displacement.
* **Mean Square Displacement (MSD) Analysis:** Calculate and plot the MSD over time for both single walkers (as squared displacement) and ensembles (as average MSD). Supports logarithmic scaling of steps for efficient analysis.
* **Path Plotting:** Generate static plots of the walker's path.
* **Animation:** Create animated visualizations of the walk (for a single walker).
* **Saving Output:** Save plots and animations to files.
* **Reproducibility:** Option to set a random seed for reproducible simulations.
* **Progress Tracking:** Uses `tqdm` to show progress bars for simulations and analyses.

## Installation

1.  **Clone the repository or download the script:**
    ```bash
    # If using git
    git clone https://github.com/roger20ub/Drunkards-walk.git
    cd drunkard-walk-simulation
    # If just downloading the file, save it as drunkard-walk-simulation.py
    ```

2.  **Install dependencies:**
    You need Python 3 and the following libraries:
    * `numpy`
    * `matplotlib`
    * `tqdm`

    Install them using pip:
    ```bash
    pip install numpy matplotlib tqdm
    ```

3.  **For saving animations (optional):**
    To save animations as MP4 files, you need `ffmpeg`. Install it using your system's package manager:
    * **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
    * **macOS (using Homebrew):** `brew install ffmpeg`
    * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Usage

Run the script from the command line.

```bash
python drunkard-walk-simulation.py [options]
