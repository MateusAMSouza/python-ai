import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the spam dataset
spam = pd.read_csv('data/spambase.csv')

# Display basic info and heatmap
spam.info()
plt.figure(figsize=(40, 32))
sns.heatmap(spam.corr(), annot=True, cmap='icefire').set_title('Correlation Heatmap')
plt.show()

# Extract top features
correlacoes = spam.corr().loc[:, 'class'].drop('class')
maiores_correlacoes = correlacoes.abs().nlargest(10)
print(f"Top correlations with the target class:\n{maiores_correlacoes}")

# Select features for classification
x = spam[['word_freq_free', 'word_freq_your']]
y = spam['class']

# Train the logistic regression model
lr = LogisticRegression()
scores = cross_val_score(lr, x, y, cv=10, scoring='accuracy')
print('Mean Accuracy:', scores.mean())

# Plot scatter
def plot_scatter():
    plt.scatter(x['word_freq_free'][y == 0], x['word_freq_your'][y == 0], alpha=.4, label='Not Spam')
    plt.scatter(x['word_freq_free'][y == 1], x['word_freq_your'][y == 1], alpha=.4, label='Spam')
    plt.legend()
    plt.xlabel('Word Frequency: Free')
    plt.ylabel('Word Frequency: Your')
plot_scatter()

# Test with new data
df_emails = pd.DataFrame([[0.85, 0.15], [0.2, 0.6]], columns=["word_freq_free", "word_freq_your"])
lr.fit(x, y)
print('Classifications:\n', lr.predict(df_emails))
print('Probabilities:\n', lr.predict_proba(df_emails))
