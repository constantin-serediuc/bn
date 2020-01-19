from stable_baselines import results_plotter
import matplotlib.pyplot as plt
# Helper from the library
results_plotter.plot_results(['/home/constantin/Desktop/projects/disertation/rl_logs'], 1e5, results_plotter.X_TIMESTEPS, "Test")
plt.show()
