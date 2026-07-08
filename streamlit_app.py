import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page Title
# -------------------------------
st.set_page_config(page_title="Penguins ML App", page_icon="🐧")

st.title("🤖 Machine Learning")
st.info("This is a machine learning application for penguin species prediction.")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("penguins_cleaned.csv")

# Features and Target
x = df.drop("species", axis=1)
y_raw = df["species"]

# -------------------------------
# Data Section
# -------------------------------
with st.expander("📊 Data"):
    st.write("### Raw Data")
    st.dataframe(df)

    st.write("### Features (X)")
    st.dataframe(x)

    st.write("### Target (Y)")
    st.dataframe(y_raw)

# -------------------------------
# Data Visualization
# -------------------------------
with st.expander("📈 Data Visualization"):
    st.scatter_chart(
        data=df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species"
    )

# -------------------------------
# Sidebar Inputs
# -------------------------------
with st.sidebar:
    st.header("Input Features")

    island = st.selectbox(
        "Island",
        ("Biscoe", "Dream", "Torgersen")
    )

    bill_length_mm = st.slider(
        "Bill Length (mm)",
        float(df.bill_length_mm.min()),
        float(df.bill_length_mm.max()),
        float(df.bill_length_mm.mean())
    )

    bill_depth_mm = st.slider(
        "Bill Depth (mm)",
        float(df.bill_depth_mm.min()),
        float(df.bill_depth_mm.max()),
        float(df.bill_depth_mm.mean())
    )

    flipper_length_mm = st.slider(
        "Flipper Length (mm)",
        float(df.flipper_length_mm.min()),
        float(df.flipper_length_mm.max()),
        float(df.flipper_length_mm.mean())
    )

    body_mass_g = st.slider(
        "Body Mass (g)",
        float(df.body_mass_g.min()),
        float(df.body_mass_g.max()),
        float(df.body_mass_g.mean())
    )

    gender = st.selectbox(
        "Gender",
        ("male", "female")
    )

# -------------------------------
# User Input DataFrame
# -------------------------------
input_df = pd.DataFrame({
    "island": [island],
    "bill_length_mm": [bill_length_mm],
    "bill_depth_mm": [bill_depth_mm],
    "flipper_length_mm": [flipper_length_mm],
    "body_mass_g": [body_mass_g],
    "sex": [gender]
})

# Combine with original data
input_penguins = pd.concat([input_df, x], ignore_index=True)

# -------------------------------
# Show Input
# -------------------------------
with st.expander("📝 Input Features"):
    st.write("### Input Penguin")
    st.dataframe(input_df)

    st.write("### Combined Dataset")
    st.dataframe(input_penguins)

# -------------------------------
# One-Hot Encoding
# -------------------------------
df_penguins = pd.get_dummies(
    input_penguins,
    columns=["island", "sex"]
)

# Split input and training data
input_row = df_penguins.iloc[:1]
X = df_penguins.iloc[1:]

# -------------------------------
# Encode Target
# -------------------------------
target_mapper = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2
}

y = y_raw.map(target_mapper)

with st.expander("⚙ Data Preparation"):
    st.write("### Encoded Input")
    st.dataframe(input_row)

    st.write("### Encoded Target")
    st.dataframe(y)

# -------------------------------
# Train Model
# -------------------------------
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X, y)

# -------------------------------
# Prediction
# -------------------------------
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

species = np.array([
    "Adelie",
    "Chinstrap",
    "Gentoo"
])

# -------------------------------
# Prediction Output
# -------------------------------
st.subheader("🐧 Predicted Species")
st.success(species[prediction][0])

# -------------------------------
# Prediction Probabilities
# -------------------------------
st.subheader("Prediction Probability")

df_prediction_proba = pd.DataFrame(
    prediction_proba,
    columns=["Adelie", "Chinstrap", "Gentoo"]
)

st.dataframe(
    df_prediction_proba,
    column_config={
        "Adelie": st.column_config.ProgressColumn(
            "Adelie",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),
        "Chinstrap": st.column_config.ProgressColumn(
            "Chinstrap",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),
        "Gentoo": st.column_config.ProgressColumn(
            "Gentoo",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),
    },
    hide_index=True
)
