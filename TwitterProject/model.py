import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint
from joblib import dump

# Load the real and fake user data
real_users = pd.read_csv("C:\\Users\\VINAYA\\OneDrive\\Desktop\\TwitterProjectn\\TwitterProject\\realusers.csv")
fake_users = pd.read_csv("C:\\Users\\VINAYA\\OneDrive\\Desktop\\TwitterProjectn\\TwitterProject\\fakeusers.csv")

# Assign labels: 1 for real users and 0 for fake users
real_users['label'] = 1
fake_users['label'] = 0

# Combine the datasets
data = pd.concat([real_users, fake_users], ignore_index=True)

# Preprocessing
# Map 'lang' values to codes
lang_list = list(enumerate(np.unique(data['lang'])))
lang_dict = {name: i for i, name in lang_list}
data['lang_code'] = data['lang'].map(lambda lang: lang_dict[lang]).astype(int)

# Feature Engineering
data['followers_friends_ratio'] = data['followers_count'] / (data['friends_count'] + 1)
data['statuses_favourites_ratio'] = data['statuses_count'] / (data['favourites_count'] + 1)

# Select features and target variable
X = data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang_code', 'followers_friends_ratio', 'statuses_favourites_ratio']]
y = data['label']

# One-hot encode the lang_code column
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['lang_code']])
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(['lang_code']))
X = pd.concat([X.drop(columns=['lang_code']), X_encoded_df], axis=1)

# Save the encoder
dump(encoder, 'encoder.joblib')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simpler parameter distribution
param_dist = {
    'n_estimators': randint(100, 200),
    'max_features': ['sqrt'],
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 3),
    'bootstrap': [True]
}

# Instantiate the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=param_dist,
                                   n_iter=25,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=2,
                                   random_state=42)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best estimator from random search
best_rf_clf = random_search.best_estimator_

# Perform cross-validation
cv_scores = cross_val_score(best_rf_clf, X, y, cv=3)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Train the best model on the full training data
best_rf_clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
dump(best_rf_clf, 'random_forest_model.joblib')
