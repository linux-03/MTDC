# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tpot import TPOTClassifier

# Load the Forex data
df = pd.read_csv('./data/EURCHF-2000-2020-15m.csv')

# Convert 'DATE_TIME' to datetime
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

# Ensure the DataFrame is sorted by date
df = df.sort_values('DATE_TIME')

# Create example features (you may need to adjust based on your specific needs)
df['price_change'] = df['CLOSE'].pct_change() * 100
df['price_diff'] = df['CLOSE'] - df['OPEN']
df['volatility'] = (df['HIGH'] - df['LOW']) / df['OPEN'] * 100
df['DC_event_price'] = df['price_change']
df['DC_event_time'] = df['DATE_TIME'].diff().dt.total_seconds().fillna(0)
df['Speed'] = df['price_change'] / df['DC_event_time']
df['Previous_DC_event_price'] = df['DC_event_price'].shift(1)
df['Previous_OS'] = (df['price_change'].shift(1) > 0).astype(int)
df['Flash_event'] = (df['DC_event_time'] == 0).astype(int)

# Label example: 1 if the price increased, 0 if the price decreased or stayed the same
df['Label'] = (df['price_change'] > 0).astype(int)

# Drop rows with NaN values created by shifting
df.dropna(inplace=True)

# Select relevant columns for the model
df = df[['DC_event_price', 'DC_event_time', 'Speed', 'Previous_DC_event_price', 'Previous_OS', 'Flash_event', 'Label']]

# Separate features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use TPOT for AutoML
tpot = TPOTClassifier(generations=2, population_size=10, verbosity=2, random_state=42, max_time_mins=5, max_eval_time_mins=1)
tpot.fit(X_train, y_train)

# Make predictions
y_pred = tpot.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Export the best pipeline 
tpot.export('best_pipeline.py')
