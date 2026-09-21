import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Anomaly Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Log Anomaly Detector")
st.caption("Paste a raw HDFS log line — fields are auto-parsed and spread out.")

COLUMNS = ["LineId", "Date", "Time", "Pid", "Level", "Component", "Content", "EventId", "EventTemplate"]

@st.cache_resource(show_spinner="Training model on HDFS dataset…")
def train():
    df = pd.read_csv("HDFS_2k.log_structured.csv")
    vec = TfidfVectorizer(max_features=500)
    X   = vec.fit_transform(df["Content"].astype(str))
    scaler = StandardScaler(with_mean=False)
    X_s = scaler.fit_transform(X)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_s)
    return vec, scaler, clf

vec, scaler, clf = train()

raw = st.text_input(
    "Paste full log line (comma-separated):",
    placeholder="1,081109,203615,148,INFO,dfs.DataNode$PacketResponder,PacketResponder 1 for block blk_38865049064139660 terminating,E10,PacketResponder <*> for block blk_<*> terminating"
)

if raw.strip():
    # Parse — split by comma but limit to 9 parts (EventTemplate may contain commas)
    parts = raw.strip().split(",", maxsplit=8)
    parts += [""] * (9 - len(parts))   # pad if short
    parsed = dict(zip(COLUMNS, parts))

    st.divider()
    st.subheader("📋 Parsed Log Fields")

    # Row 1: LineId | Date | Time | Pid | Level
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LineId",    parsed["LineId"])
    c2.metric("Date",      parsed["Date"])
    c3.metric("Time",      parsed["Time"])
    c4.metric("Pid",       parsed["Pid"])
    c5.metric("Level",     parsed["Level"])

    # Row 2: Component | EventId | EventTemplate
    c6, c7, c8 = st.columns([2, 1, 3])
    c6.markdown(f"**Component**\n\n`{parsed['Component']}`")
    c7.markdown(f"**EventId**\n\n`{parsed['EventId']}`")
    c8.markdown(f"**EventTemplate**\n\n`{parsed['EventTemplate']}`")

    # Content full width
    st.markdown("**Content**")
    st.info(parsed["Content"] if parsed["Content"] else "—")

    # Predict
    st.divider()
    if st.button("Predict", type="primary", use_container_width=True):
        if not parsed["Content"].strip():
            st.warning("Content field is empty — cannot predict.")
        else:
            X_new = scaler.transform(vec.transform([parsed["Content"].strip()]))
            pred  = clf.predict(X_new)[0]
            if pred == -1:
                st.error("🚨 **Anomaly Detected**")
            else:
                st.success("✅ **Normal Log**")
