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
        
        
# a helper function to examine low cardinality categorical features
def visualize_categorical(X_train, low_cardinality, figsize=(20, 10)):
    # Calculate the number of rows needed for subplots
    n_rows = len(low_cardinality) // 2
    if len(low_cardinality) % 2: n_rows += 1

    # visualize the histogram of the categorical variables
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(low_cardinality):
        # print the normalize value counts
        print(X_train[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
        print("\n")
        
        # Order bars based on their count
        order = X_train[col].value_counts().index
        
        # Create horizontal bars and order them based on their count
        sns.countplot(data=X_train, y=col, ax=axes[i], color='skyblue', order=order)
        
        axes[i].set_title(f"Distribution of {col}")

    # Remove the extra subplot if the number of categories is odd
    if len(low_cardinality) % 2:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()
    
    # visualize the distribution for large 
def plot_top_10_frequencies(X_train, large_cat_var, figsize=(20, 20)):
    top_10_values = {}
    for feature in large_cat_var:
        top_10_values[feature] = X_train[feature].value_counts().head(10)

    # Calculate the number of rows needed for subplots
    n_rows = len(large_cat_var) // 2
    if len(large_cat_var) % 2: n_rows += 1

    # Create a subplot grid for the histogram plots
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    # Flatten the axes array if there's more than one row
    if n_rows > 1:
        axes = axes.flatten()

    # Plot the frequency of the top 10 values for each categorical feature on the corresponding subplot
    for i, feature in enumerate(top_10_values.keys()):
        ax = axes[i]  # Select the appropriate subplot
        top_10_values[feature].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f"Top 10 Frequencies of {feature}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels for better readability

    # Remove the extra subplot if the number of features is odd
    if len(large_cat_var) % 2:
        fig.delaxes(axes[-1])

    # Adjust layout and spacing
    plt.tight_layout()
 

def plot_boxplot_and_histogram(df, numerical_features, loan_status):
    # Calculate the number of rows needed for subplots
    n_rows = len(numerical_features)
    fig, axes = plt.subplots(n_rows, 2, figsize=(30, 60))  # Change figsize to a larger size

    for i, col in enumerate(numerical_features):
        # Boxplot on the left
        sns.boxplot(data=df, x=loan_status, y=col, ax=axes[i, 0])
        axes[i, 0].set_title(f"Boxplot of {col} vs {loan_status}")

        # Histogram on the right
        sns.histplot(data=df, x=col, hue=loan_status, ax=axes[i, 1], kde=True)
        axes[i, 1].set_title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()