
import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image
import pytesseract

st.set_page_config(page_title="NBA Value Bet Predictor", layout="wide")
st.title("NBA Player Prediction and Value Bet Detection")

csv_file = st.file_uploader("Upload Cleaned NBA Stats CSV (90 days)", type=["csv"])
jpeg_file = st.file_uploader("Optional: Upload FanDuel Prop JPEG", type=["jpg", "jpeg", "png"])

odds_df = pd.DataFrame()
if jpeg_file:
    image = Image.open(jpeg_file)
    text = pytesseract.image_to_string(image)
    lines = text.split('\n')
    data = []
    for line in lines:
        if any(char.isdigit() for char in line):
            parts = line.split()
            if len(parts) >= 2:
                player = " ".join(parts[:-1])
                prop = parts[-1]
                try:
                    value = float(prop)
                    data.append({"player": player, "line": value})
                except:
                    continue
    odds_df = pd.DataFrame(data)

if csv_file:
    df = pd.read_csv(csv_file)
    features = ["rebounds", "assists", "three_pointers", "minutes", "team_def_rating", "opp_def_rating"]
    if all(f in df.columns for f in features):
        X = df[features]
        y = df["rebounds"] * 0.1 + df["assists"] * 0.5 + df["three_pointers"] * 2 + df["minutes"] * 0.3
        model = GradientBoostingRegressor()
        model.fit(X, y)
        df['predicted_points'] = model.predict(X)
        if not odds_df.empty:
            df = pd.merge(df, odds_df, on='player', how='left')
            df['value_gap'] = df['predicted_points'] - df['line']
            df['value_rating'] = df['value_gap'].apply(lambda x: "High" if abs(x) > 5 else "Medium" if abs(x) > 2 else "Low")
        else:
            df['line'] = None
            df['value_gap'] = None
            df['value_rating'] = "Not Scored"
        show_cols = [col for col in ["player", "team", "opponent", "predicted_points", "line", "value_gap", "value_rating"] if col in df.columns]
        top_players = df.sort_values(by='predicted_points', ascending=False).head(40)
        st.success("Top 30–40 Value Bets Based on Predictions:")
        st.dataframe(top_players[show_cols])
    else:
        st.error("Missing required columns in uploaded CSV.")
