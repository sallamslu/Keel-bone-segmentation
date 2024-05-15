import os
import pandas as pd
import matplotlib.pyplot as plt

# list to store data from folds
dataframes = []

# paths to folds
folder_paths = ['output_fold_1', 'output_fold_2', 'output_fold_3', 'output_fold_4', 'output_fold_5']

# read from paths to folders
for folder_path in folder_paths:
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            filepath = os.path.join(folder_path, file)
            df = pd.read_csv(filepath)
            dataframes.append(df)

# combine data from folds
combined_df = pd.concat(dataframes)

# give colors for each fold
colors = ['b', 'g', 'r', 'c', 'm']

# creat the a figure with two subplots of loss and dice
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# plot training and validation loss
for fold, (df, color) in enumerate(zip(dataframes, colors), start=1):
    axes[0].plot(df['epoch'], df['loss'], linestyle='--', color=color)
    axes[0].plot(df['epoch'], df['val_loss'], linestyle='-', color=color)

# legend for folds in the loss subplot
for color, fold in zip(colors, range(1, 6)):
    axes[0].plot([], [], color=color, label=f'Fold {fold}')
# Add legend for line styles in the loss subplot
axes[0].plot([], [], linestyle='--', color='k', label='Training')
axes[0].plot([], [], linestyle='-', color='k', label='Validation')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and validation loss across folds')
axes[0].legend()
axes[0].grid(True)

# plot training and validation dice coefficient
for fold, (df, color) in enumerate(zip(dataframes, colors), start=1):
    axes[1].plot(df['epoch'], df['dice_coef'], linestyle='--', color=color)
    axes[1].plot(df['epoch'], df['val_dice_coef'], linestyle='-', color=color)

# add legend for folds in the dice coefficient subplot
for color, fold in zip(colors, range(1, 6)):
    axes[1].plot([], [], color=color, label=f'Fold {fold}')
# Add legend for line styles in the dice coefficient subplot
axes[1].plot([], [], linestyle='--', color='k', label='Training')
axes[1].plot([], [], linestyle='-', color='k', label='Validation')

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice coefficient')
axes[1].set_title('Training and validation dice coefficient across folds')
axes[1].legend()
axes[1].grid(True)

# Save the combined plot
plt.tight_layout()
plt.savefig('training_validation_combined_plot.jpg', format='jpeg', dpi=300)
plt.close()

