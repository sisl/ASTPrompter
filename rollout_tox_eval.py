import json
import numpy as np
import matplotlib.pyplot as plt
from toxicity.dxf import call_detoxify

from tqdm import tqdm

# Load the data
MODEL_NAME = "rollout_data_gpt2_defense_gpt2_best_convokit_old"

with open(f"output/{MODEL_NAME}.json", "r") as f:
    data = json.load(f)

# Loop through each rollout and get the toxicity scores. Add to list
toxicity_scores = []

for rollout in tqdm(data["rollouts"], desc="Getting toxicity scores"):
# for rollout in data["rollouts"][0:2]:
    attacker_rollout = rollout["attacker_rollout"]
    baseline_rollout = rollout["baseline_rollout"]
    prompt = rollout["prompt"]

    # Concatenate the rollout into a single string
    attacker_rollout = " ".join(attacker_rollout)
    baseline_rollout = " ".join(baseline_rollout)

    toxicity_scores.append({
        "prompt_id": rollout["prompt_id"],
        "attacker": call_detoxify(attacker_rollout),
        "baseline": call_detoxify(baseline_rollout)
    })

# Scatter plot of toxicity scores between attacker and baseline

attacker_scores = [float(x["attacker"]["toxicity"]) for x in toxicity_scores]
baseline_scores = [float(x["baseline"]["toxicity"]) for x in toxicity_scores]

# Plot values for each id
plt.scatter(range(len(attacker_scores)), attacker_scores, label="Attacker")
plt.scatter(range(len(baseline_scores)), baseline_scores, label="Baseline")
plt.xlabel("Prompt ID")
plt.ylabel("Toxicity Score")
plt.legend()
plt.show()

print(f'Attacker - Mean: {np.mean(attacker_scores)}, Std: {np.std(attacker_scores)}')
print(f'Baseline - Mean: {np.mean(baseline_scores)}, Std: {np.std(baseline_scores)}')

# Save figure
plt.savefig(f"output/{MODEL_NAME}_toxicity_scores.png", dpi=300, bbox_inches="tight")