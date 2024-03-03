import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


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
    

# a helper function to plot the percentage crosstab for categorical features
def plot_percentage_crosstab(X_train, y_train, low_cardinality):
    num_vars = len(low_cardinality)
    num_cols = num_vars // 2 + num_vars % 2 

    # two rows
    fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(8 * num_cols, 10))
    axes = axes.flatten()

    # plot heatmap
    for i, var in enumerate(low_cardinality):
        crosstab = pd.crosstab(X_train[var], y_train, normalize='columns') * 100

        order = X_train[var].value_counts().index

        sns.heatmap(crosstab.loc[order], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
        axes[i].set_title(f'Percentage Crosstab of {var} vs Target')
        axes[i].set_ylabel(var)
        axes[i].set_xlabel('Target')

    # hide unused subplots
    for j in range(num_vars, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
# a helper class for feature engineering 
class DataPreprocessor:
    def __init__(self):
        pass
    # drop some useless numerical cols 
    def drop_useless_numerical_cols(self, df):
        
        useless_cols = ['chargeoff_within_12_mths', 
                        'collections_12_mths_ex_med', 
                        'delinq_amnt', 
                        'tax_liens', 
                        'policy_code', 
                        'acc_now_delinq', 
                        'total_rec_late_fee',
                        'application_type',
                        'url', 
                        'pymnt_plan',
                        'id',
                        'member_id',
                        'zip_code', 
                        'title',
                        'pub_rec',
                        'mths_since_last_record',
                        'last_pymnt_d',
                        'loan_amnt'] # highly correlated with funded amnt]
        
        # Only keep columns that exist in the DataFrame
        cols_to_drop = [col for col in useless_cols if col in df.columns]    
        df = df.drop(cols_to_drop, axis=1)
        return df
    # --------------------------------
    # numerical feature engineering
    ## 1. convert interest rate cols to numerical from categorical
    def convert_interest_rate_to_numerical(self, df):
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
        df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)
        return df
    ## 2. emp_length
    def emp_length_fe(self, df):
        # convert emp_length to numerical: 11 types - 10+ years, 1 year, 2 years, 3 years, 4 years, 5 years, 6 years, 7 years, 8 years, 9 years, < 1 year -> Convert to 10:10, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 0:0
        emp_length_dict = {'10+ years': 10, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '< 1 year': 0}
        df['emp_length'] = df['emp_length'].map(emp_length_dict)
        return df
    ## 3.mths_since_last_delinq
    def mths_since_last_delinq_fe(self, df):
        # assume that if the record is missing, it means the person has never been delinquent
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].isna().astype(int)
        return df
    ## 4 pub_rec_bankruptcies
    def pub_rec_bankruptcies_fe(self, df):
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(lambda x: 0 if pd.isna(x) or x == 0.0 else 1)
        return df
    # --------------------------------
    # categorical feature engineering 
    ## 1. Term 
    def term_fe(self, df):
        # convert 36 months to 0 and 60 months to 1
        df['term'] = df['term'].apply(lambda x: 0 if x == '36 months' else 1)
        return df
    
    ## 2. grade 
    def grade_feature_engineering(self, df):
        # convert grade to numerical grade: 7 grades - A to G -> Convert to A:0, B:1, C:2, D:3, E:4, F:5, G:6
        grade_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_dict)
        return df
    
    ## 3. Home ownership
    def home_ownership_fe(self, df):
        # convert home ownership to numerical: 6 types - MORTGAGE, RENT, OWN, OTHER, NONE, ANY -> Convert to MORTGAGE:0, RENT:1, OWN:2, OTHER:3, NONE:4
        home_ownership_dict = {'MORTGAGE': 0, 'RENT': 1, 'OWN': 2, 'OTHER': 3, 'NONE': 4}
        df['home_ownership'] = df['home_ownership'].map(home_ownership_dict)
        return df
    
    ## 4. verification_status
    def verification_status_fe(self, df):
        # convert verification_status to numerical: 3 types - Not Verified, Verified, Source Verified -> Convert to Not Verified:0, Verified:1, Source Verified:2
        verification_status_dict = {'Not Verified': 0, 'Verified': 1, 'Source Verified': 2}
        df['verification_status'] = df['verification_status'].map(verification_status_dict)
        return df
    
    ## 6. earliest_cr_line
    def earliest_cr_line_fe(self, df):
    # convert to number of months from earliest credit line to the date of loan application
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
        df['issue_d'] = pd.to_datetime(df['issue_d'])  # Ensure 'issue_d' is also in datetime format
        df['earliest_cr_line'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 30
        return df
        
    ## 7. desc 
    def desc_fe(self, df):
        # convert desc to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1
        df['desc'] = df['desc'].isna().astype(int)
        return df
    
    ## 8. emp_title
    def emp_title_fe(self, df):
        # convert emp_title to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1 (job less vs having a job)
        df['emp_title'] = df['emp_title'].isna().astype(int)
        return df
    
    ## 9.mths_since_last_delinq
    def mths_since_last_delinq_fe(self, df):
        # convert mths_since_last_delinq to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].isna().astype(int)
        return df
    
    ## 10. last_credit_pull_d
    def last_credit_pull_d_fe(self, df):
        # if it is this month (Sep 2016), convert to 1, else convert to 0 (long ago)
        df['last_credit_pull_d'] = df['last_credit_pull_d'].apply(lambda x: 1 if x == 'Sep-2016' else 0)
        return df
    
    ## 11. next_pymnt_d
    def next_pymnt_d_fe(self, df):
        # if it is next month (Oct 2016), convert to 0, else convert to 1
        df['next_pymnt_d'] = df['next_pymnt_d'].apply(lambda x: 1 if x == 'Oct-2016' else 0)
        return df
    
    ## 12. issue_d
    def months_between_issue_d_and_current_date(self, df):
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['issue_d'] = (pd.to_datetime('2016-09-01') - df['issue_d']).dt.days / 30
        return df
    # 13. converting the target variable to numerical
    def convert_target_to_numerical(self, y):
        y = y.apply(lambda x: 1 if x == 'default' else 0)
        return y
    
    # --------------------------------
    
    # Feature engineering pipeline 
    def feature_engineering_pipeline(self, df):
        df = self.drop_useless_numerical_cols(df)
        df = self.convert_interest_rate_to_numerical(df)
        df = self.emp_length_fe(df)
        df = self.mths_since_last_delinq_fe(df)
        df = self.pub_rec_bankruptcies_fe(df)
        df = self.term_fe(df)
        df = self.grade_feature_engineering(df)
        df = self.home_ownership_fe(df)
        df = self.verification_status_fe(df)
        df = self.earliest_cr_line_fe(df)
        df = self.desc_fe(df)
        df = self.emp_title_fe(df)
        df = self.mths_since_last_delinq_fe(df)
        df = self.last_credit_pull_d_fe(df)
        df = self.next_pymnt_d_fe(df)
        df = self.months_between_issue_d_and_current_date(df)
        return df
        

# Create a function to automate the pipeline (preq: preprocessor)
def train_model(algorithm, preprocessor, X_train, y_train ):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', algorithm)])
    model = pipeline.fit(X_train, y_train)
    return model

# create a function to get prediciton 
def get_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob

# create a class to store eval metrics 
class PerformanceMetrics:
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.roc_auc = roc_auc_score(y_true, y_pred_prob)
        self.f_1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def print_performance_metrics(self, model_name = None):
        print(f'Performance Metrics of the model {model_name}: \n')
        print(f'Accuracy: {self.accuracy:.4f}')
        print(f'Precision: {self.precision:.4f}')
        print(f'Recall: {self.recall:.4f}')
        print(f'ROC AUC: {self.roc_auc:.4f}')
        print(f'F1 Score: {self.f_1:.4f}')