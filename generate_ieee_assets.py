import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

# Create folder for plots
os.makedirs('results/ieee_plots', exist_ok=True)

# Set global font for IEEE style (Times New Roman usually, but we will use a clean serif)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# --- DATA ---
datasets = ['Dataset-1\n(DefenseDroid)', 'Dataset-2\n(Mendeley)', 'Dataset-3\n(Lopez)']
baseline_features = [137, 105, 54]
bpso_features = [130, 27, 50]

baseline_acc = [91.65, 67.98, 93.75]
bpso_acc = [91.65, 67.32, 97.50]

x = np.arange(len(datasets))
width = 0.35

# ==========================================
# CHART 1: Feature Reduction Comparison
# ==========================================
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, baseline_features, width, label='Baseline (GBFI)', color='#4C72B0', edgecolor='black')
plt.bar(x + width/2, bpso_features, width, label='Proposed (BPSO Swarm)', color='#DD8452', edgecolor='black')

plt.ylabel('Number of Selected Features', fontsize=12)
plt.title('Feature Set Size Reduction', fontsize=14, fontweight='bold')
plt.xticks(x, datasets, fontsize=10)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(baseline_features):
    plt.text(i - width/2, v + 3, str(v), ha='center', fontweight='bold', fontsize=10)
for i, v in enumerate(bpso_features):
    plt.text(i + width/2, v + 3, str(v), ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/ieee_plots/Chart1_Feature_Reduction.png', dpi=300)
plt.close()

# ==========================================
# CHART 2: Accuracy Comparison
# ==========================================
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, baseline_acc, width, label='Baseline (GBFI)', color='#4C72B0', edgecolor='black')
plt.bar(x + width/2, bpso_acc, width, label='Proposed (BPSO Swarm)', color='#55A868', edgecolor='black')

plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, datasets, fontsize=10)
plt.ylim(50, 105)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(baseline_acc):
    plt.text(i - width/2, v + 1, f"{v}%", ha='center', fontweight='bold', fontsize=10)
for i, v in enumerate(bpso_acc):
    plt.text(i + width/2, v + 1, f"{v}%", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/ieee_plots/Chart2_Accuracy.png', dpi=300)
plt.close()

# ==========================================
# CHART 3: METHODOLOGY FLOWCHART
# ==========================================
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

# Define boxes
boxes = [
    {"text": "Android Permission Dataset\n(Full Feature Set)", "xy": (0.5, 0.9), "color": "#E8F4F8"},
    {"text": "Data Preprocessing\n(Cleaning & Train/Test Split)", "xy": (0.5, 0.75), "color": "#E8F4F8"},
    {"text": "Initialization: GBFI-Seeded or Random 80/20\n(Hard Limit Constraint applied)", "xy": (0.5, 0.58), "color": "#FFF2CC"},
    {"text": "Evaluate Particle Fitness\n(RF Accuracy - Feature Penalty)", "xy": (0.5, 0.41), "color": "#FFF2CC"},
    {"text": "Update Personal Best (pBest)\nand Global Best (gBest)", "xy": (0.5, 0.24), "color": "#FFF2CC"},
    {"text": "Final Optimized Feature Subset\n(Swarm Output)", "xy": (0.5, 0.07), "color": "#D5E8D4"}
]

# Draw boxes
for box in boxes:
    rect = patches.FancyBboxPatch((box["xy"][0] - 0.35, box["xy"][1] - 0.05), 0.7, 0.1,
                                  boxstyle="round,pad=0.02",
                                  edgecolor="black", facecolor=box["color"], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(box["xy"][0], box["xy"][1], box["text"], ha="center", va="center", fontsize=11, fontweight='bold')

# Draw arrows
for i in range(len(boxes) - 1):
    start_y = boxes[i]["xy"][1] - 0.05
    end_y = boxes[i+1]["xy"][1] + 0.05
    ax.annotate("", xy=(0.5, end_y), xytext=(0.5, start_y),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

# Add BPSO Loop Arrow — clean right-angle path, strictly left of boxes
# Route: left edge of box-5 (0.15,0.24) → go left to x=0.06 →
#        go up to y=0.58 → turn right into left edge of box-3 (0.15,0.58)
loop_x = [0.15, 0.06, 0.06, 0.15]
loop_y = [0.24, 0.24, 0.58, 0.58]
ax.plot(loop_x, loop_y, color="#D35400", lw=1.5, clip_on=False)
# Arrowhead only at the entry point into box-3
ax.annotate("", xy=(0.15, 0.58), xytext=(0.08, 0.58),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#D35400",
                            mutation_scale=12))
# Label well clear of boxes (x=0.02)
ax.text(0.02, 0.41, "Iterative\nSwarm\nOptimization\nLoop",
        ha="center", va="center", fontsize=9, color="#D35400",
        fontweight="bold", rotation=90)

plt.title("Proposed Hybrid Swarm Intelligence Methodology", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/ieee_plots/Chart3_Methodology_Flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("All IEEE-ready charts and the methodology flowchart have been successfully generated in results/ieee_plots/!")
