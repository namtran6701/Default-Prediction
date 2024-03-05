import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# a helper function to examine the numerical features
def analyze_numerical(df, features):
    num_features = len(features)
    rows = num_features // 2 + num_features % 2  # Calculate rows needed
    cols = 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Main figure creation
        fig, axes = plt.subplots(rows, cols, figsize=(
            40, 60))  # Increase the figsize values
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
    if len(low_cardinality) % 2:
        n_rows += 1

    # visualize the histogram of the categorical variables
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(low_cardinality):
        # print the normalize value counts
        print(X_train[col].value_counts(normalize=True).mul(
            100).round(2).astype(str) + '%')
        print("\n")

        # Order bars based on their count
        order = X_train[col].value_counts().index

        # Create horizontal bars and order them based on their count
        sns.countplot(data=X_train, y=col,
                      ax=axes[i], color='skyblue', order=order)

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
    if len(large_cat_var) % 2:
        n_rows += 1

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
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', labelrotation=45)

    # Remove the extra subplot if the number of features is odd
    if len(large_cat_var) % 2:
        fig.delaxes(axes[-1])

    # Adjust layout and spacing
    plt.tight_layout()


def plot_boxplot_and_histogram(df, numerical_features, loan_status):
    # Calculate the number of rows needed for subplots
    n_rows = len(numerical_features)
    # Change figsize to a larger size
    fig, axes = plt.subplots(n_rows, 2, figsize=(30, 60))

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
    fig, axes = plt.subplots(nrows=2, ncols=num_cols,
                             figsize=(8 * num_cols, 10))
    axes = axes.flatten()

    # plot heatmap
    for i, var in enumerate(low_cardinality):
        crosstab = pd.crosstab(
            X_train[var], y_train, normalize='columns') * 100

        order = X_train[var].value_counts().index

        sns.heatmap(crosstab.loc[order], annot=True,
                    fmt=".2f", cmap="YlGnBu", ax=axes[i])
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
                        'loan_amnt']  # highly correlated with funded amnt]

        # Only keep columns that exist in the DataFrame
        cols_to_drop = [col for col in useless_cols if col in df.columns]
        df = df.drop(cols_to_drop, axis=1)
        return df
    # --------------------------------
    # numerical feature engineering
    # 1. convert interest rate cols to numerical from categorical

    def convert_interest_rate_to_numerical(self, df):
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
        df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)
        return df
    # 2. emp_length

    def emp_length_fe(self, df):
        # convert emp_length to numerical: 11 types - 10+ years, 1 year, 2 years, 3 years, 4 years, 5 years, 6 years, 7 years, 8 years, 9 years, < 1 year -> Convert to 10:10, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 0:0
        emp_length_dict = {'10+ years': 10, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                           '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '< 1 year': 0}
        df['emp_length'] = df['emp_length'].map(emp_length_dict)
        return df
    # 3.mths_since_last_delinq

    def mths_since_last_delinq_fe(self, df):
        # assume that if the record is missing, it means the person has never been delinquent
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].isna().astype(int)
        return df
    # 4 pub_rec_bankruptcies

    def pub_rec_bankruptcies_fe(self, df):
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(
            lambda x: 0 if pd.isna(x) or x == 0.0 else 1)
        return df
    # --------------------------------
    # categorical feature engineering
    # 1. Term

    def term_fe(self, df):
        # convert 36 months to 0 and 60 months to 1
        df['term'] = df['term'].apply(lambda x: 0 if x == '36 months' else 1)
        return df

    # 2. grade
    def grade_feature_engineering(self, df):
        # convert grade to numerical grade: 7 grades - A to G -> Convert to A:0, B:1, C:2, D:3, E:4, F:5, G:6
        grade_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_dict)
        return df

    # 3. Home ownership
    def home_ownership_fe(self, df):
        # convert home ownership to numerical: 6 types - MORTGAGE, RENT, OWN, OTHER, NONE, ANY -> Convert to MORTGAGE:0, RENT:1, OWN:2, OTHER:3, NONE:4
        home_ownership_dict = {'MORTGAGE': 0,
                               'RENT': 1, 'OWN': 2, 'OTHER': 3, 'NONE': 4}
        df['home_ownership'] = df['home_ownership'].map(home_ownership_dict)
        return df

    # 4. verification_status
    def verification_status_fe(self, df):
        # convert verification_status to numerical: 3 types - Not Verified, Verified, Source Verified -> Convert to Not Verified:0, Verified:1, Source Verified:2
        verification_status_dict = {
            'Not Verified': 0, 'Verified': 1, 'Source Verified': 2}
        df['verification_status'] = df['verification_status'].map(
            verification_status_dict)
        return df

    # 6. earliest_cr_line
    def earliest_cr_line_fe(self, df):
        # convert to number of months from earliest credit line to the date of loan application
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
        # Ensure 'issue_d' is also in datetime format
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['earliest_cr_line'] = (
            df['issue_d'] - df['earliest_cr_line']).dt.days / 30
        return df

    # 7. desc
    def desc_fe(self, df):
        # convert desc to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1
        df['desc'] = df['desc'].isna().astype(int)
        return df

    # 8. emp_title
    def emp_title_fe(self, df):
        # convert emp_title to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1 (job less vs having a job)
        df['emp_title'] = df['emp_title'].isna().astype(int)
        return df

    # 9.mths_since_last_delinq
    def mths_since_last_delinq_fe(self, df):
        # convert mths_since_last_delinq to numerical: 2 types - NaN, not NaN -> Convert to NaN:0, not NaN:1
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].isna().astype(int)
        return df

    # 10. last_credit_pull_d
    def last_credit_pull_d_fe(self, df):
        # if it is this month (Sep 2016), convert to 1, else convert to 0 (long ago)
        df['last_credit_pull_d'] = df['last_credit_pull_d'].apply(
            lambda x: 1 if x == 'Sep-2016' else 0)
        return df

    # 11. next_pymnt_d
    def next_pymnt_d_fe(self, df):
        # if it is next month (Oct 2016), convert to 0, else convert to 1
        df['next_pymnt_d'] = df['next_pymnt_d'].apply(
            lambda x: 1 if x == 'Oct-2016' else 0)
        return df

    # 12. issue_d
    def months_between_issue_d_and_current_date(self, df):
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['issue_d'] = (pd.to_datetime('2016-09-01') -
                         df['issue_d']).dt.days / 30
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
def train_model(algorithm, preprocessor, X_train, y_train):
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
        self.f_1 = 2 * (self.precision * self.recall) / \
            (self.precision + self.recall)

    def print_performance_metrics(self, model_name=None):
        print(f'Performance Metrics of the model {model_name}: \n')
        print(f'Accuracy: {self.accuracy:.4f}')
        print(f'Precision: {self.precision:.4f}')
        print(f'Recall: {self.recall:.4f}')
        print(f'ROC AUC: {self.roc_auc:.4f}')
        print(f'F1 Score: {self.f_1:.4f}')

# create a function to store peformance metrics in a table


def create_metrics_dataframe(performance_metrics, index_name):
    data = {
        'accuracy': [performance_metrics.accuracy],
        'precision': [performance_metrics.precision],
        'recall': [performance_metrics.recall],
        'roc_auc': [performance_metrics.roc_auc],
        'f1_score': [performance_metrics.f_1]
    }
    df = pd.DataFrame(data)
    df.index = [index_name]
    return df

# function for model tuning

# Random Forest random search


def tune_rf_rs(rf_pipeline, X_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

    max_features = ['auto', 'sqrt', 'log2']

    max_depth = max_depth = [int(x) for x in np.linspace(5, 120, num=12)]

    min_samples_split = [10, 15, 20, 30, 50]

    min_samples_leaf = [5, 10, 15]

    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'classifier__n_estimators': n_estimators,
                   'classifier__max_features': max_features,
                   'classifier__max_depth': max_depth,
                   'classifier__min_samples_split': min_samples_split,
                   'classifier__min_samples_leaf': min_samples_leaf,
                   'classifier__bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    random_search_rf = RandomizedSearchCV(estimator=rf_pipeline,
                                          param_distributions=random_grid,
                                          n_iter=15, cv=3,
                                          verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    random_search_rf.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search_rf.best_params_

    return best_params

# Random Forest grid search (define the search space after the random search results)


def tune_rf_gs(rf_pipeline, X_train, y_train):

    # Create the grid
    grid_search_space = {'classifier__n_estimators': [900, 1000, 1100],
                         'classifier__max_features': ['sqrt'],
                         'classifier__max_depth': [32, 36, 40],
                         'classifier__min_samples_split': [50, 60],
                         'classifier__min_samples_leaf': [5, 8],
                         'classifier__bootstrap': [False]}
    # Use the grid to search for best hyperparameters
    grid_search_rf = GridSearchCV(
        estimator=rf_pipeline, param_grid=grid_search_space, cv=3, verbose=2, n_jobs=-1)

    # Fit the grid search model
    grid_search_rf.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search_rf.best_params_

    return best_params

# XGBoost random search


def tune_xgb_rs(xgb_pipeline, x, y):
    # Define the hyperparameter grid
    xgb_grid = {
        'classifier__n_estimators': [100, 200, 500, 1000],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__gamma': [0.5, 1, 1.5, 2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'classifier__reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    }

    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=xgb_grid,
        n_iter=30,
        cv=3,
        verbose=2,
        random_state=6701,
        n_jobs=-1,
        scoring='roc_auc'
    )

    # Fit the random search model
    random_search_xgb.fit(x, y)

    # Get the best parameters
    best_params = random_search_xgb.best_params_

    return best_params

# XGBoost grid search (define the search space after the random search results)


def tune_xgb_gs(xgb_pipeline, X_resampled, y_resampled):
    # Create the grid
    xgb_grid = {
        'classifier__n_estimators': [950, 1000, 1050],
        'classifier__learning_rate': [0.01, 0.001],
        'classifier__max_depth': [10, 12],
        'classifier__min_child_weight': [5, 7],
        'classifier__gamma': [2, 3],
        'classifier__subsample': [0.6, 0.65],
        'classifier__colsample_bytree': [0.8, 0.85],
        'classifier__reg_alpha': [0, 0.0001],
        'classifier__reg_lambda': [1, 1.5]
    }

    # Use the grid to search for best hyperparameters
    grid_search_xgb = GridSearchCV(
        estimator=xgb_pipeline, param_grid=xgb_grid, cv=3, verbose=2, n_jobs=-1, scoring='roc_auc')

    # Fit the grid search model
    grid_search_xgb.fit(X_resampled, y_resampled)

    # Get the best parameters
    best_params = grid_search_xgb.best_params_

    return best_params

# Stacking Classifier


def create_stacking_classifier(preprocessor):

    # Define base estimators for the stacker
    base_estimators = [
        ('gbm', GradientBoostingClassifier(n_estimators=30,
         learning_rate=1.0, max_depth=3, random_state=42)),
        ('xgb', XGBClassifier(ase_score=None, booster=None, callbacks=None,
                              colsample_bylevel=None, colsample_bynode=None,
                              colsample_bytree=0.8, device=None, early_stopping_rounds=None,
                              enable_categorical=False, eval_metric=None, feature_types=None,
                              gamma=2, grow_policy=None, importance_type=None,
                              interaction_constraints=None, learning_rate=0.01, max_bin=None,
                              max_cat_threshold=None, max_cat_to_onehot=None,
                              max_delta_step=None, max_depth=12, max_leaves=None,
                              min_child_weight=5, monotone_constraints=None)),
        ('rf', RandomForestClassifier(bootstrap=False,
                                      max_depth=40,
                                      min_samples_leaf=5,
                                      min_samples_split=50,
                                      n_estimators=1100,
                                      random_state=0)),
        ('lr', LogisticRegression(random_state=42))
    ]

    # Final estimator on top
    final_estimator = LogisticRegression()

    # Create stacking classifier
    stacking_classifier = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1
    )

    # Create stacked pipeline
    stacked_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', stacking_classifier)])

    return stacked_pipeline

# Metrics visualization

# 1. Confusion Matrix


def plot_confusion_matrix(y_true, y_pred_prob, threshold, model_name):
    """
    Plot the confusion matrix at a given threshold.

    Parameters:
    - y_true: The true target values.
    - y_pred_prob: The predicted target values.
    - threshold: The threshold to use for classification.
    - model_name: The name of the model.

    Returns:
    - The confusion matrix.
    """
    y_pred = y_pred_prob > threshold

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix at 5% FPR for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 2. ROC Curve


def plot_roc_curve(pred_prob, model_names, y_test):
    for i, pred_prob in enumerate(pred_prob):

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(y_test, pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC: {roc_auc:.2f})')

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Precision Recall Curve

def plot_threshold_precision_recall(y_true, y_pred_prob, title='Threshold-Precision-Recall Curve'):
    """
    Plot the relationship between the threshold and precision, and threshold and recall.

    Parameters:
    - y_true: The true target values.
    - y_pred_prob: The predicted target values.

    Returns:
    - The threshold-precision-recall curve.
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = [precision_score(y_true, y_pred_prob > t) for t in thresholds]
    recalls = [recall_score(y_true, y_pred_prob > t) for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, marker='o',
             linestyle='--', color='blue', label='Precision')
    plt.plot(thresholds, recalls, marker='o',
             linestyle='--', color='orange', label='Recall')
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# plot roc and precision recall curve together


def plot_roc_pr_curves(y_true, y_pred_prob, fpr_percentile):
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(
        y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    # Find the index for FPR
    idx = next(i for i, x in enumerate(fpr) if x >= fpr_percentile / 100)
    # Fix the above line to iterate through fpr array elements

    # Find the closest threshold in the PR curve to the one identified in the ROC curve analysis
    roc_threshold = thresholds_roc[idx]
    closest_threshold_index = np.argmin(np.abs(thresholds_pr - roc_threshold))
    selected_precision = precision[closest_threshold_index]
    selected_recall = recall[closest_threshold_index]

    # Create a figure with two plots side by side
    plt.figure(figsize=(14, 6))

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.plot(fpr[idx], tpr[idx], 'ro', label=f'~{fpr_percentile}% FPR')
    plt.annotate(f'FPR ~{fpr_percentile}%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]),
                 textcoords="offset points", xytext=(40, 10), ha='center')
    plt.axvline(x=fpr[idx], color='r', linestyle='--',
                label=f'{fpr_percentile}% FPR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Plot PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.plot(selected_recall, selected_precision, 'ro')
    plt.annotate(f'Threshold={roc_threshold:.2f}\nPrecision={selected_precision:.2f}\nRecall={selected_recall:.2f}',
                 (selected_recall, selected_precision),
                 textcoords="offset points",
                 xytext=(-10, 20),
                 ha='center')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

# Feature Importance

# 1. Logistic Regression Coefficients


def plot_lr_fi(lr_pipeline):
    # Get feature names
    feature_names = lr_pipeline.named_steps['preprocessor'].get_feature_names_out(
    )

    # Get feature importance coefficients
    feature_importance = lr_pipeline.named_steps['classifier'].coef_

    # Create DataFrame for feature importance
    feature_importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': feature_importance[0]})

    # Calculate absolute importance
    feature_importance_df['abs_importance'] = feature_importance_df['importance'].abs(
    )

    # Sort DataFrame by absolute importance
    feature_importance_df = feature_importance_df.sort_values(
        'abs_importance', ascending=False)

    # Create binary column for positive/negative importance
    feature_importance_df['positive'] = feature_importance_df['importance'] > 0

    # Remove prefix from feature names
    feature_importance_df['feature'] = feature_importance_df['feature'].str.replace(
        'num__', '')
    feature_importance_df['feature'] = feature_importance_df['feature'].str.replace(
        'cat__', '')

    # Plot feature importance for Logistic Regression
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Features Coefficients in Logistic Regression')

    # Create list of colors based on sign of coefficients
    colors = ['green' if x >=
              0 else 'red' for x in feature_importance_df['importance'][:10]]

    # Create horizontal bar plot
    plt.barh(feature_importance_df['feature'][:10],
             feature_importance_df['abs_importance'][:10], color=colors)
    plt.xlabel('Absolute Coefficient')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()

    # Create legend
    red_patch = mpatches.Patch(color='red', label='Negative Values')
    green_patch = mpatches.Patch(color='green', label='Positive Values')
    plt.legend(handles=[red_patch, green_patch])

    plt.show()


# 2. Random Forest Feature Importance
def plot_tree_fi(rf_pipeline):
    # Get feature names
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out(
    )

    # Get feature importance
    feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_

    # Create DataFrame for feature importance
    feature_importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': feature_importance})

    # Sort DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(
        'importance', ascending=False)

    # Plot feature importance for Random Forest
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Features Importance in Random Forest')

    # Create horizontal bar plot
    plt.barh(feature_importance_df['feature'][:10],
             feature_importance_df['importance'][:10], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()

    plt.show()

# Partial Dependence Plot

# Numerical Features PDP


def pdp_plot_numeric(ax, X_train, var, sample_n, pipeline):
    pdp_values = pd.DataFrame(X_train[var].sort_values().sample(
        frac=0.1).unique(), columns=[var])
    pdp_sample = X_train.sample(sample_n).drop(var, axis=1)

    pdp_cross = pdp_sample.merge(pdp_values, how='cross')
    pdp_cross['pred'] = pipeline.predict_proba(pdp_cross)[:, 1]

    sns.lineplot(ax=ax, x=f"{var}", y='pred', data=pdp_cross)
    ax.set_title(f"Partial Dependence Plot: {var}")
    ax.set_ylabel('Predicted Probability')
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax.grid(True)

# Categorical Features PDP


def pdp_plot_categorical(ax, X_train, var, sample_n, pipeline):

    pdp_values = pd.DataFrame(X_train[var].sort_values().sample(
        frac=0.1).unique(), columns=[var])
    pdp_sample = X_train.sample(sample_n).drop(var, axis=1)

    pdp_cross = pdp_sample.merge(pdp_values, how='cross')
    pdp_cross['pred'] = pipeline.predict_proba(pdp_cross)[:, 1]
    mean_pred = pdp_cross['pred'].mean()
    pdp_cross['pred'] = pdp_cross['pred'].apply(lambda x: x - mean_pred)
    sns.barplot(ax=ax, x='pred', y=f"{var}",
                ci=None,
                data=pdp_cross,
                estimator="mean")
    ax.set_title(f"Partial Dependence Plot: {var}")
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel(var)
    ax.tick_params(axis='y', rotation=45)  # Rotate y-axis labels
    ax.grid(True)

# Local Breakdown interaction


def plot_local_breakdown_interactions(top_10_tp, pipeline_explainer):
    for index, row in top_10_tp.iterrows():
        local_breakdown_exp = pipeline_explainer.predict_parts(
            top_10_tp.iloc[index],
            type='break_down_interactions',
            label=f"record:{index}, prob:{row['pred_proba']:.3f}")

        local_breakdown_exp.plot()
