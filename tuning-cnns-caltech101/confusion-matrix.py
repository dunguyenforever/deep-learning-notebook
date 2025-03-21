import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the accuracy array
accuracy_array = np.array([
    [0.0, 0.0, 0.0],  # CNN-1 results
    [0.0, 0.0, 0.0],  # CNN-2 results
    [0.0, 0.0, 0.0]   # CNN-3 results
])

# Define row names
row_names = [
    "CNN-1: 2 Conv Layers (with MaxPooling) and 2 Fully Connected Layers",
    "CNN-2: 3 Conv Layers (with MaxPooling) and 2 Fully Connected Layers",
    "CNN-3: 4 Conv Layers (with MaxPooling) and 2 Fully Connected Layers"
]

# Define column names
column_names = ["Data Aug: RandHorizontalFlip, RandRotation(30), Resize(64,64), Dropout(0.5), Adam(weight_decay=2e-4,alpha=0.99)", "Hyperparam Combo 2", "Hyperparam Combo 3"]

# Create a DataFrame
df = pd.DataFrame(accuracy_array, index=row_names, columns=column_names)

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_frame_on(False)  # Remove frame
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Create table
table = ax.table(cellText=df.round(2).values, 
                 colLabels=df.columns,
                 rowLabels=df.index,
                 cellLoc='center', 
                 loc='center')

# Apply color gradient
for i in range(len(row_names)):
    for j in range(len(column_names)):
        val = accuracy_array[i, j]
        color = plt.cm.Blues(val)  # Use colormap to assign color based on accuracy
        table[i+1, j].set_facecolor(color)

# Adjust layout and show plot
plt.title("Test Accuracy for Different CNN Models", fontsize=12)
plt.show()
