import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# a helper function to examine the numerical features
def analyze_numerical(df, features):
    num_features = len(features)
    rows = num_features // 2 + num_features % 2  # Calculate rows needed
    cols = 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Main figure creation
        fig, axes = plt.subplots(rows, cols, figsize=(40, 60))  # Increase the figsize values
        axes = axes.flatten()  # For easy iteration if not 2 columns neatly

        for i in range(len(axes)):
            if i < len(features):
                col = features[i]
                # Plotting on the subplot axes
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])  
                axes[i].set_title(f"Distribution of {col}")
            else:
                # Remove axis for empty plots
                fig.delaxes(axes[i])
            
        plt.tight_layout() 
        plt.show()