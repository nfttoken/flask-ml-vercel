import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
# Adjust the column names and target as needed

# Load dataset
# Use header from CSV
cols = ['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv('api/adult.csv')

# Encode categorical columns
cat_cols = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country']
for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

df['income'] = (df['income'].str.strip() == '>50K').astype(int)

X = df.drop(['income'], axis=1)
y = df['income']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model trained and saved as model.pkl')
