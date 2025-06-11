import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

@st.cache_data
def load_data():
    identifiers = r"C:\\Users\\angus123\\Downloads\\ip_addresses_full\\ip_addresses_full\\identifiers.csv"
    time = r"C:\\Users\\angus123\\Downloads\\times\\times\\times_10_minutes.csv"
    relationship = r"C:\\Users\\angus123\\Downloads\\ids_relationship.csv"
    base_dir = r"C:\\Users\\angus123\\Downloads\\ip_addresses_full\\ip_addresses_full\\agg_10_minutes"

    relation_df = pd.read_csv(relationship)
    time_df = pd.read_csv(time)
    time_df.columns = ["id_time", "timestamp"]
    time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])
    identifier_df = pd.read_csv(identifiers)

    inst = relation_df["id_institution"].drop_duplicates().sort_values().head(5)
    filtered = relation_df[relation_df["id_institution"].isin(inst)]
    top_subnets = (
        filtered.drop_duplicates(subset=["id_institution", "id_institution_subnet"])
        .groupby("id_institution")
        .head(4)
    )

    top_subnets = top_subnets.merge(identifier_df, on="id_ip", how="left")
    top_subnets["csv_path"] = top_subnets.apply(
        lambda row: fr"{base_dir}\{row['id_ip_folder']}\{row['id_ip']}.csv", axis=1
    )

    all_frames = []
    for _, row in top_subnets.iterrows():
        path = Path(row["csv_path"])
        if path.exists():
            try:
                df = pd.read_csv(path)
                df = df.merge(time_df, on="id_time", how="left")
                df["id_ip"] = row["id_ip"]
                df["id_institution"] = row["id_institution"]
                df["id_institution_subnet"] = row["id_institution_subnet"]
                all_frames.append(df)
            except:
                continue

    return pd.concat(all_frames, ignore_index=True)

df_all = load_data()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")
df_all = df_all[df_all["timestamp"] >= pd.Timestamp("2024-04-01", tz="UTC")].copy()
df_all = df_all.sort_values("timestamp")

st.title("Google Fiber Forecast & Capacity Dashboard")

institutions = df_all["id_institution"].dropna().unique()
selected_institution = st.selectbox("Filter by Institution (for plots only)", institutions)

df_plot = df_all[df_all["id_institution"] == selected_institution]

subnets = df_plot["id_institution_subnet"].dropna().unique()
subnet_id = st.selectbox("Select Subnet for Forecasting", subnets)

df = df_plot[df_plot["id_institution_subnet"] == subnet_id].copy()

for lag in [1, 2, 3]:
    df[f"n_bytes_lag_{lag}"] = df["n_bytes"].shift(lag)

df["hour"] = df["timestamp"].dt.hour

df = df.dropna()
features = [f"n_bytes_lag_{lag}" for lag in [1, 2, 3]] + ["hour"]
X = df[features]
y = df["n_bytes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df.loc[y_test.index, "n_bytes_pred"] = y_pred

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.header("Forecasting")
st.metric("MAE", f"{mae:,.0f} bytes")
st.metric("RMSE", f"{rmse:,.0f} bytes")

st.subheader("Forecast vs Actual (n_bytes)")
plot_df = df.loc[y_test.index].copy().tail(100)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(plot_df["timestamp"], plot_df["n_bytes"], label="Actual", linewidth=2)
ax.plot(plot_df["timestamp"], plot_df["n_bytes_pred"], label="Predicted", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("n_bytes")
ax.legend()
st.pyplot(fig)

# Scaled Forecasting
subnet_mean = df["n_bytes"].mean()
df["n_bytes_scaled"] = df["n_bytes"] / subnet_mean
for lag in [1, 2, 3]:
    df[f"n_bytes_scaled_lag_{lag}"] = df["n_bytes_scaled"].shift(lag)

features_scaled = [f"n_bytes_scaled_lag_{lag}" for lag in [1, 2, 3]] + ["hour"]
df = df.dropna(subset=features_scaled + ["n_bytes_scaled"])

X_scaled = df[features_scaled]
y_scaled = df["n_bytes_scaled"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_scaled, shuffle=False, test_size=0.2)
model_scaled = GradientBoostingRegressor()
model_scaled.fit(X_train_s, y_train_s)
y_pred_scaled = model_scaled.predict(X_test_s)
df.loc[y_test_s.index, "n_bytes_scaled_pred"] = y_pred_scaled

st.subheader("Forecast vs Actual (Scaled by Subnet Average)")
plot_df_scaled = df.loc[y_test_s.index].copy().tail(100)
fig_scaled, ax_scaled = plt.subplots(figsize=(14, 5))
ax_scaled.plot(plot_df_scaled["timestamp"], plot_df_scaled["n_bytes_scaled"], label="Actual (scaled)", linewidth=2)
ax_scaled.plot(plot_df_scaled["timestamp"], plot_df_scaled["n_bytes_scaled_pred"], label="Predicted (scaled)", linestyle="--")
ax_scaled.set_xlabel("Time")
ax_scaled.set_ylabel("n_bytes / subnet mean")
ax_scaled.legend()
st.pyplot(fig_scaled)

st.subheader("Daily Average TCP vs UDP Packet Ratio")
if "tcp_udp_ratio_packets" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    daily_ratio = df.set_index("timestamp").resample("D")["tcp_udp_ratio_packets"].mean().dropna()
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(daily_ratio.index, daily_ratio.values, label="Daily Avg TCP Ratio", color="purple")
    ax2.set_ylabel("TCP Ratio (0=UDP, 1=TCP)")
    ax2.set_xlabel("Date")
    ax2.set_title("Daily Average TCP vs UDP Packet Ratio")
    ax2.grid(True)
    st.pyplot(fig2)
else:
    st.warning("TCP/UDP ratio data not available for this subnet.")

st.subheader("Daily Average Inbound vs Outbound Packet Ratio")
if "dir_ratio_packets" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    daily_dir_ratio = df.set_index("timestamp").resample("D")["dir_ratio_packets"].mean().dropna()
    fig_dir_daily, ax_dir_daily = plt.subplots(figsize=(14, 4))
    ax_dir_daily.plot(daily_dir_ratio.index, daily_dir_ratio.values, label="Daily Avg Direction Ratio", color="green")
    ax_dir_daily.set_ylabel("Outbound Ratio (0=Inbound, 1=Outbound)")
    ax_dir_daily.set_xlabel("Date")
    ax_dir_daily.set_title("Daily Average Inbound vs Outbound Packet Ratio")
    ax_dir_daily.grid(True)
    st.pyplot(fig_dir_daily)

st.subheader("Forecast Residuals")
df["residual"] = abs(df["n_bytes"] - df["n_bytes_pred"])
fig_resid, ax_resid = plt.subplots(figsize=(14, 4))
ax_resid.plot(df["timestamp"], df["residual"], label="Forecast Error", color="red")
ax_resid.set_ylabel("Residual (bytes)")
ax_resid.set_title("Forecast Error Over Time")
ax_resid.grid(True)
st.pyplot(fig_resid)

st.header("Capacity Summary")
capacity_summary = (
    df_all.groupby(["id_institution", "id_institution_subnet"])
    .agg(
        avg_bytes=("n_bytes", "mean"),
        peak_bytes=("n_bytes", "max"),
        p95_bytes=("n_bytes", lambda x: np.percentile(x, 95)),
        samples=("n_bytes", "count"),
    )
    .reset_index()
)

st.subheader("Top 10 Subnets by Average Usage")
top_capacity = capacity_summary.sort_values("avg_bytes", ascending=False).head(10)
st.dataframe(top_capacity, use_container_width=True)

st.subheader("Average Usage by Institution")
avg_by_inst = df_all.groupby("id_institution")["n_bytes"].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.bar(avg_by_inst["id_institution"].astype(str), avg_by_inst["n_bytes"])
ax3.set_ylabel("Average n_bytes")
ax3.set_xlabel("Institution ID")
ax3.set_title("Average Network Usage by Institution")
st.pyplot(fig3)

st.subheader("Institution Burstiness (95th vs Average)")
burst = df_all.groupby("id_institution").agg(
    avg=("n_bytes", "mean"),
    p95=("n_bytes", lambda x: np.percentile(x, 95))
).reset_index()
burst["burst_ratio"] = burst["p95"] / burst["avg"]
fig_burst, ax_burst = plt.subplots(figsize=(12, 4))
ax_burst.bar(burst["id_institution"].astype(str), burst["burst_ratio"])
ax_burst.set_ylabel("95th / Avg Ratio")
ax_burst.set_title("Burstiness by Institution")
st.pyplot(fig_burst)
