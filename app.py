import mysql.connector
import joblib
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'hotel_prediction'
}

# Load trained model
model = joblib.load("model/model.pkl")

# Load clustering model and scaler
cluster_model = joblib.load("model/cluster_model.pkl")
cluster_scaler = joblib.load("model/cluster_scaler.pkl")

# Mapping for categorical inputs
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

category_mappings = {
    'hotel': ['City Hotel', 'Resort Hotel'],
    'meal': ['BB', 'FB', 'HB', 'SC', 'Undefined'],
    'market_segment': ['Aviation', 'Complementary', 'Corporate', 'Direct', 'Groups',
                       'Offline TA/TO', 'Online TA', 'Undefined'],
    'distribution_channel': ['Corporate', 'Direct', 'GDS', 'TA/TO', 'Undefined'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Contract', 'Group', 'Transient', 'Transient-Party'],
    'continent': ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America', 'Unknown']
}

# Function to one-hot encode categorical inputs
def one_hot_encode_input(form_data, category_mappings):
    one_hot_encoded = {}

    for field, categories in category_mappings.items():
        selected_value = form_data[field]
        for category in categories:
            col_name = f"{field}_{category}"
            one_hot_encoded[col_name] = 1 if selected_value == category else 0

    return one_hot_encoded

# Columns used in the model
input_features = ['lead_time', 'arrival_date_month', 'arrival_date_week_number',
                  'arrival_date_day_of_month', 'is_repeated_guest', 'previous_cancellations',
                  'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'adr',
                  'required_car_parking_spaces', 'total_of_special_requests', 'room_mismatch', 'total_guests',
                  'total_stay', 'hotel_City Hotel', 'hotel_Resort Hotel', 'meal_BB', 'meal_FB', 'meal_HB',
                  'meal_SC', 'meal_Undefined', 'market_segment_Aviation', 'market_segment_Complementary',
                  'market_segment_Corporate', 'market_segment_Direct', 'market_segment_Groups',
                  'market_segment_Offline TA/TO', 'market_segment_Online TA', 'market_segment_Undefined',
                  'distribution_channel_Corporate', 'distribution_channel_Direct', 'distribution_channel_GDS',
                  'distribution_channel_TA/TO', 'distribution_channel_Undefined', 'deposit_type_No Deposit',
                  'deposit_type_Non Refund', 'deposit_type_Refundable', 'customer_type_Contract',
                  'customer_type_Group', 'customer_type_Transient', 'customer_type_Transient-Party',
                  'continent_Africa', 'continent_Asia', 'continent_Europe', 'continent_North America',
                  'continent_Oceania', 'continent_South America', 'continent_Unknown']



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        hotel_name = request.form['hotel_name']
        email = request.form['email']
        password = request.form['password']
        hotel_type = request.form['hotel_type']

        # Store in database
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (hotel_name, email, password, hotel_type) 
                VALUES (%s, %s, %s, %s)    
            """, (hotel_name, email, password, hotel_type))     #%s prevents SQL injection attacks
            conn.commit()
            cursor.close()
            conn.close()

            # Store hotel type in session for later use. So it can be used even in another route
            session['hotel_type'] = hotel_type
            return redirect(url_for('login'))
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))

            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session['user_id'] = user['id']
                session['hotel_type'] = user['hotel_type']
                return redirect(url_for('prediction_form'))
            else:
                return "Invalid credentials"
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('login.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction_form():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        form_data = request.form.to_dict()

        # Convert date to month, week number, and day of month
        arrival_date = datetime.strptime(form_data['arrival_date'], '%Y-%m-%d')
        form_data['arrival_date_month'] = arrival_date.strftime('%B')  # Full month name
        form_data['arrival_date_week_number'] = arrival_date.isocalendar()[1]
        form_data['arrival_date_day_of_month'] = arrival_date.day

        # Add the hotel type from session
        form_data['hotel'] = session['hotel_type']

        # Convert appropriate fields to integers or floats
        num_fields = ['lead_time', 'previous_cancellations', 'previous_bookings_not_canceled',
                      'booking_changes', 'days_in_waiting_list', 'adr',
                      'required_car_parking_spaces', 'total_of_special_requests',
                      'total_guests', 'total_stay']

        for field in num_fields:
            form_data[field] = float(form_data[field])

        # Convert booleans (checkboxes)
        form_data['is_repeated_guest'] = 1 if form_data.get('is_repeated_guest') == 'on' else 0
        form_data['room_mismatch'] = 1 if form_data.get('room_mismatch') == 'on' else 0

        # Convert month name to number
        form_data['arrival_date_month'] = months.index(form_data['arrival_date_month']) + 1

        # Separate numerical and boolean fields
        numerical_data = {key: form_data[key] for key in form_data if key not in category_mappings}

        # One-hot encode categorical fields
        one_hot_data = one_hot_encode_input(form_data, category_mappings)

        # Combine into final input dict
        final_input_dict = {**numerical_data, **one_hot_data}

        # Create DataFrame
        input_df = pd.DataFrame([final_input_dict])

        # Reorder columns to match the model's expected order
        input_df = input_df[input_features]

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO predictions (
                    user_id, arrival_date, lead_time, meal, market_segment, distribution_channel, 
                    is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, booking_changes, 
                    deposit_type, days_in_waiting_list, customer_type, adr, required_car_parking_spaces, 
                    total_of_special_requests, room_mismatch, continent, total_guests, total_stay, 
                    predicted_cancellation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                session['user_id'],
                form_data['arrival_date'],
                form_data['lead_time'],
                form_data['meal'],
                form_data['market_segment'],
                form_data['distribution_channel'],
                form_data['is_repeated_guest'],
                form_data['previous_cancellations'],
                form_data['previous_bookings_not_canceled'],
                form_data['booking_changes'],
                form_data['deposit_type'],
                form_data['days_in_waiting_list'],
                form_data['customer_type'],
                form_data['adr'],
                form_data['required_car_parking_spaces'],
                form_data['total_of_special_requests'],
                form_data['room_mismatch'],
                form_data['continent'],
                form_data['total_guests'],
                form_data['total_stay'],
                int(prediction)
            )

            cursor.execute(insert_query, values)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            return f"Database Insert Error: {str(e)}"



        # --- Clustering Logic ---
        cluster_input = input_df.copy()

        # Ensure cluster_input matches scaler's expected features
        cluster_features = cluster_scaler.feature_names_in_

        # Add missing columns (if any) with 0
        for col in cluster_features:
            if col not in cluster_input.columns:
                cluster_input[col] = 0

        # Reorder columns to match scaler's expected order
        cluster_input = cluster_input[cluster_features]


        print("---------------------------------------------------------------------------------")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(cluster_input)

        # Apply scaling & clustering
        cluster_scaled = cluster_scaler.transform(cluster_input)
        cluster_label = cluster_model.predict(cluster_scaled)[0]

        print(cluster_label)


        # Message based on cancellation prediction
        if prediction == 1:
            result_text = "⚠️ This booking is likely to be canceled."
            cancel_tips = {
                0: [  # Corporate/Business Travelers
                    "🏆 Corporate Loyalty: 'Stay 4 nights, get 5th free' ",
                    "🛎️ Room assignment double-check: Supervisor must verify room type matches booking before check-in",
                    "💸 Compensation policy: If room mismatch occurs, offer 20% discount or free breakfast immediately",
                    "📱 Priority support: Assign dedicated WhatsApp line for corporate clients (<15 min response time)"
                ],
                1: [  # Leisure Travelers
                    "💰 Direct booking incentive: Booking through our website = 5% discount + free welcome drink",
                            #encouraging them to book directly next time
                    "🎟️ Non-refundable experience bundles: 'Add sunset cruise for $50 (normally $75) when booking'",
                    "⏳ Confirm next booking within 24 hours? Offer a free room upgrade as a reward to encourage quick bookings.",
                    "📧 If a guest is interested in rebooking, ask for their preferences and send a personalized itinerary."
                ],
                2: [  # Group Travelers
                    "👔 Group coordinator portal: Provide a login for the group coordinator to manage rooms and payments. Send the link after booking.",
                    "💳 Flexible deposits: Offer 10% initial payment  for groups of 15+ rooms. Add this option in the contract.",
                    "🍽️ Custom meal planning: Send a dietary preference form 14 days before arrival for custom meal planning.",
                    "🔄 Group rescheduling: If a group reschedules (instead of canceling), offer 1 free room for their next booking."
                ]
            }
            recommendation_list = cancel_tips.get(cluster_label, [])
        else:
            result_text = "✅ This booking is not likely to be canceled."
            sat_tips = {
                0: [  # Corporate/Business Travelers
                    "📊 Corporate Reporting: Send a quarterly savings report to travel managers to highlight cost savings.",
                    "☕ Extended Stay Perks: Offer free pressing service or executive lounge access for stays of 5+ nights.",
                    "💻 Productivity Bundle: Provide premium Wi-Fi and a quiet floor for remote workers",
                    "📅 Automated Rebooking: Sync the guest’s calendar to suggest optimal future stay dates",
                    "🧳 Business Traveler Kit: Place adapters, notepads, and stain remover in the room."
                ],
                1: [  # Leisure Travelers
                    "📸 Instagram Package: Offer a free cocktail at the rooftop bar if they tag us in 3 photos.",
                    "🌅 Sunrise Experience: Offer a 6AM yoga/breakfast package at booking.",
                    "🗺️ Neighborhood Guide: Provide a curated self-guided tour map with hidden spots",
                    "🛎️ Surprise Upgrade: Randomly upgrade 5% of leisure bookings to a better room "
                ],
                2: [  # Group Travelers
                    "📷 Group Photo Service: Schedule a photographer during breakfast and deliver prints before checkout.",
                    "🍝 Family-Style Dining: Offer shared platters for groups of 20+ people.",
                    "👕 Group Souvenirs: Provide custom keepsakes (e.g., embroidered caps) for the group",
                    "🗣️ Multilingual Support: Ensure staff speaks the group's primary language.",
                    "🎤 Evening Events: Organize karaoke or bingo nights for large groups"
                ]
            }
            recommendation_list = sat_tips.get(cluster_label, [])

        return render_template("form.html",
                               prediction=result_text,
                               probability=f"{prediction_proba:.2%}",
                               cluster=cluster_label,
                               recommendation_list=recommendation_list)

    return render_template("form.html")


@app.route('/dashboard')
def dashboard():
    # Get the username from the session
    user_id = session.get('user_id')

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    # Fetch hotel name from the users table
    cursor.execute("SELECT hotel_name FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    # Get the hotel name, or set a default if not found
    hotel_name = user['hotel_name'] if user else 'Hotel Name not found'

    # Monthly cancellations
    cursor.execute("""
            SELECT 
                MONTH(arrival_date) AS month,
                YEAR(arrival_date) AS year,
                COUNT(*) AS total_bookings,
                SUM(predicted_cancellation) AS canceled_bookings,
                (SUM(predicted_cancellation) / COUNT(*)) * 100 AS cancellation_rate
            FROM predictions
            WHERE YEAR(arrival_date) = YEAR(CURDATE())
                AND user_id = %s
            GROUP BY YEAR(arrival_date), MONTH(arrival_date)
        """, (user_id,))
    monthly_cancellations = cursor.fetchall()

    # Summary stats
    cursor.execute("""
            SELECT 
                COUNT(*) AS total_non_canceled,
                AVG(lead_time) AS avg_lead_time,
                AVG(adr) AS avg_adr,
                AVG(total_guests) AS avg_total_guests,
                AVG(total_stay) AS avg_total_stay
            FROM predictions
            WHERE YEAR(arrival_date) = YEAR(CURDATE()) 
                AND MONTH(arrival_date) = MONTH(CURDATE())
                AND predicted_cancellation = 0
                AND user_id = %s
        """, (user_id,))
    summary_stats = cursor.fetchone()

    cursor.close()
    conn.close()

    # Convert month numbers to names using previously defined month list
    for row in monthly_cancellations:
        row['month'] = months[row['month'] - 1]

    return render_template('dashboard.html',
                           hotel_name=hotel_name,
                           monthly_cancellations=monthly_cancellations,
                           summary_stats=summary_stats)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)