import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('heartdiseases.csv')
data = pd.get_dummies(data, columns=['Gender', 'Education', 'Smoker'], drop_first=True)

X = data.drop('Heart_Disease', axis=1)
y = data['Heart_Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)   

predictions = log_reg.predict(X_test)
print("Predictions:", predictions)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

predictions_prob = log_reg.predict_proba(X_test)[:,1]
print("Prediction Probabilities:", predictions_prob)

y_test = y_test.map({'Yes': 1, "No": 0})

fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
auc = roc_auc_score(y_test, predictions_prob)

plt.plot(fpr, tpr, label=f'AUC: {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

odds_ratios = np.exp(log_reg.coef_)
print("Odds Ratios:", odds_ratios)
