import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# === Load & Prepare Data ===
@st.cache_data
def load_data():
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()
    launchpads = requests.get("https://api.spacexdata.com/v4/launchpads").json()

    rocket_map = {r['id']: r['name'] for r in rockets}
    launchpad_map = {l['id']: (l['name'], l['latitude'], l['longitude']) for l in launchpads}

    data = []
    for launch in launches:
        if launch.get('success') is None or launch.get('launchpad') is None:
            continue
        pad = launchpad_map.get(launch['launchpad'], ('Unknown', None, None))
        data.append({
            'flight_number': launch['flight_number'],
            'name': launch['name'],
            'date_utc': launch['date_utc'],
            'launch_year': int(launch['date_utc'][:4]),
            'rocket_id': launch['rocket'],
            'rocket_name': rocket_map.get(launch['rocket'], 'Unknown'),
            'launchpad_name': pad[0],
            'latitude': pad[1],
            'longitude': pad[2],
            'success': int(launch['success']),
            'details': launch.get('details', '')
        })
    return pd.DataFrame(data)

df = load_data()

# === Build Model ===
le = LabelEncoder()
df['encoded_rocket'] = le.fit_transform(df['rocket_id'])

X = df[['launch_year', 'encoded_rocket']]
y = df['success']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === Sidebar ===
st.sidebar.title("🚀 SpaceX Launch Dashboard")
st.sidebar.write("Filters & Prediction")

# === Theme Toggle (Light / Dark) ===
mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212 !important;
        color: #ffffff !important;
    }
    .css-1cpxqw2, .css-ffhzg2, .css-10trblm {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# === Filters ===
years = sorted(df['launch_year'].unique())
selected_year = st.sidebar.selectbox("Filter launches by Year", years)

rocket_options = df[['rocket_id', 'rocket_name']].drop_duplicates().reset_index(drop=True)
rocket_names = rocket_options['rocket_name'].tolist()

rocket_name = st.sidebar.selectbox("Select Rocket for Prediction", rocket_names)
input_year = st.sidebar.number_input("Enter Launch Year (Prediction)", min_value=2006, max_value=2030, value=2024)
input_month = st.sidebar.slider("Launch Month", 1, 12, 6)
input_day = st.sidebar.slider("Launch Day", 1, 31, 15)

predict_btn = st.sidebar.button("Predict Success")

# === Main Page ===
st.title("🚀 SpaceX Launch Dashboard")
st.write("Explore SpaceX launch history, visualize launch sites, and predict future launch success probabilities.")

filtered = df[df['launch_year'] == selected_year]

# --- Launch Outcomes Bar Chart ---
st.subheader(f"Launch Outcomes in {selected_year}")
success_counts = filtered['success'].value_counts().rename({1: 'Success', 0: 'Failure'})
st.bar_chart(success_counts)

# --- Launch Trends Over Years ---
st.subheader("📈 Launch Trends Over Years")
launches_per_year = df.groupby('launch_year').size()
success_rate_per_year = df.groupby('launch_year')['success'].mean()

fig, ax1 = plt.subplots(figsize=(10, 4))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Launches', color=color)
ax1.plot(launches_per_year.index, launches_per_year.values, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Success Rate', color=color)
ax2.plot(success_rate_per_year.index, success_rate_per_year.values, color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)

st.pyplot(fig)

# --- Launch Sites Map ---
st.subheader("🌍 Launch Sites Map")
map_data = filtered.dropna(subset=['latitude', 'longitude'])
m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=3)
for _, row in map_data.iterrows():
    popup_text = f"{row['name']} ({row['launchpad_name']})<br>Success: {'Yes' if row['success'] == 1 else 'No'}"
    color = 'green' if row['success'] == 1 else 'red'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        popup=popup_text,
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)
st_folium(m, width=700, height=450)

# --- Feature Importance ---
st.subheader("🧠 Model Feature Importance")
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

# === Prediction Result ===
if predict_btn:
    rocket_id = rocket_options[rocket_options['rocket_name'] == rocket_name]['rocket_id'].values[0]
    encoded_id = le.transform([rocket_id])[0]
    input_data = pd.DataFrame([[input_year, encoded_id]], columns=['launch_year', 'encoded_rocket'])
    proba = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted Success Probability: {proba:.2%}")

# === Fixed info box in corner ===
total_launches = len(df)
success_rate = df['success'].mean() * 100

st.markdown(f"""
<div id="fixed-box" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(30,30,30,0.85);
    border-radius: 12px;
    padding: 15px 25px;
    box-shadow: 0 0 10px 2px #4caf50;
    z-index: 9999;
    font-size: 14px;
    color: #d1d1d1;
    max-width: 250px;">
    <b>Dashboard Summary</b><br>
    Total Launches: {total_launches}<br>
    Overall Success Rate: {success_rate:.1f}%<br>
    Selected Year: {selected_year}<br>
    Rockets: {len(rocket_names)}<br>
</div>
""", unsafe_allow_html=True)

# === Footer ===
st.markdown("<br><hr><p style='text-align:center'>Made with ❤️ by Aaraiz Ali | Data from SpaceX API</p>", unsafe_allow_html=True)
