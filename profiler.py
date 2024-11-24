import pstats

# Use the absolute path for the profiling file
stats = pstats.Stats("/Users/haydentedrake/Desktop/UROP2024/pbh_mass/output.prof")

# Sort by time and save results to a file
stats.sort_stats("time")
with open("/Users/haydentedrake/Desktop/UROP2024/pbh_mass/profile_results.txt", "w") as f:
    stats.stream = f
    stats.print_stats()
