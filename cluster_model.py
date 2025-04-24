import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv('data//hotel_booking_cancellation.csv')

# Convert 'arrival_date_month' to numerical values
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['arrival_date_month'] = df['arrival_date_month'].apply(lambda x: months.index(x) + 1)

# Convert boolean columns to 0/1
bool_col = ['is_repeated_guest', 'room_mismatch']
df[bool_col] = df[bool_col].astype(int)

# One-hot encode categorical variables
cat_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type', 'continent']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Standardize numerical and boolean columns
num_cols = [
    'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'total_stay', 'total_guests', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

# Combine all numeric data
X_features = num_cols + bool_col + [col for col in df_encoded.columns if col not in num_cols + bool_col]
X = df_encoded[X_features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the clustering model (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save the scaler and model
joblib.dump(scaler, 'model/cluster_scaler.pkl')
joblib.dump(kmeans, 'model/cluster_model.pkl')

print("Clustering model and scaler saved successfully.")
