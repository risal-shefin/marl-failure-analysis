import os
import re
import argparse
import matplotlib.pyplot as plt

# File path
parser = argparse.ArgumentParser(description="Plot correlation barplots from results file.")
parser.add_argument("--file_path", type=str, help="Path to the correlation analysis results file")
parser.add_argument("--save_ext", type=str, default=None, help="File extension for saving the plot")
parser.add_argument("--title", type=str, default="Correlation Barplots", help="Title for the plot")
args = parser.parse_args()
file_path = args.file_path


# Read file
with open(file_path, "r") as f:
    text = f.read()

# Regex to extract percentages
shapley_matches = re.findall(r"Shapley Values vs Reward Drop Correlation:[\s\S]*?Index \d.*?= ([\d.]+)%", text)
edge_matches = re.findall(r"Edge Scores vs Reward Drop Correlation:[\s\S]*?Index \d.*?= ([\d.]+)%", text)

# Convert to float
shapley_values = [float(x) for x in shapley_matches]
edge_values = [float(x) for x in edge_matches]

# Rank labels
ranks = ["Rank1", "Rank2", "Rank3"]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Left plot - Shapley
axes[0].bar(ranks, shapley_values, color="skyblue")
axes[0].set_title(f"Shapley vs Reward Drop [{args.title}]")
axes[0].set_ylabel("Correlation (%)")

# Right plot - Edge Scores
axes[1].bar(ranks, edge_values, color="salmon")
axes[1].set_title(f"Edge Scores vs Reward Drop [{args.title}]")

# Adjust layout
plt.tight_layout()

# Save plot in same folder
out_dir = os.path.dirname(file_path)
if args.save_ext:
    out_path = os.path.join(out_dir, f"{args.save_ext}_correlation_barplots.png")
else:
    out_path = os.path.join(out_dir, "correlation_barplots.png")
plt.savefig(out_path, dpi=300)

print(f"Plot saved at: {out_path}")
