if step_counts[0] == 0 and max_steps > 0 : step_counts[0] = 1 # Ensure first step is at least 1
            if step_counts[-1] < max_steps: # Make sure the last step is included
                step_counts = np.append(step_counts, max_steps)
                step_counts = np.unique(step_counts) # Remove potential duplicates