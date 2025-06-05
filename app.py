import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit.components.v1 as components

# Page setup
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

# --- CSS for animations and layout ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e0f2f1, #f1f8e9);
        }

        .main-title {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            color: #2e7d32;
            animation: glow 2s ease-in-out infinite alternate;
            margin-top: 0;
        }

        @keyframes glow {
            from {text-shadow: 0 0 10px #4caf50;}
            to {text-shadow: 0 0 20px #81c784, 0 0 30px #66bb6a;}
        }

        @keyframes floatUp {
            0% {transform: translateY(0);}
            100% {transform: translateY(-15px);}
        }

        .emoji-banner {
            font-size: 2.2em;
            text-align: center;
            animation: floatUp 2s ease-in-out infinite alternate;
            margin-top: 10px;
        }

        .section-header {
            font-size: 1.6em;
            margin-top: 30px;
            color: #2e7d32;
            animation: slide-in 1s ease forwards;
        }

        @keyframes slide-in {
            0% {transform: translateY(-30px); opacity: 0;}
            100% {transform: translateY(0); opacity: 1;}
        }

        .stButton>button {
            background-color: #43a047;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #2e7d32;
            transform: scale(1.05);
        }

        .download-section {
            margin-top: 20px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {box-shadow: 0 0 0 0 rgba(67, 160, 71, 0.4);}
            70% {box-shadow: 0 0 0 10px rgba(67, 160, 71, 0);}
            100% {box-shadow: 0 0 0 0 rgba(67, 160, 71, 0);}
        }

        /* About Slide Panel */
        #about-toggle { display: none; }
        #about-panel {
            position: fixed;
            top: 0;
            right: -320px;
            width: 320px;
            height: 100%;
            background: #2e7d32;
            color: white;
            padding: 30px;
            transition: right 0.4s ease;
            z-index: 9999;
            overflow-y: auto;
        }
        #about-toggle:checked + #about-panel {
            right: 0;
        }
        #about-label {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4caf50;
            color: white;
            padding: 12px 18px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10000;
        }
        #about-label:hover {
            background: #388e3c;
        }
    </style>
""", unsafe_allow_html=True)

# --- Slide-In About Panel ---
st.markdown("""
<input type="checkbox" id="about-toggle" />
<label for="about-toggle" id="about-label">About Model</label>
<div id="about-panel">
    <h2 style='margin-top:0;'>About KMeans</h2>
    <p>KMeans is a powerful unsupervised algorithm that groups data into K clusters.</p>
    <ul>
        <li>Best for numerical data</li>
        <li>Each cluster has a centroid</li>
        <li>Minimizes intra-cluster variance</li>
    </ul>
    <p>This app uses KMeans to discover customer segments.</p>
</div>
""", unsafe_allow_html=True)

# --- Title and Emoji Banner ---
st.markdown('<div class="main-title">📊 Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="emoji-banner">🧍‍♂️🧍‍♀️🧑‍💼💰💳💸</div>', unsafe_allow_html=True)

# --- Upload file ---
uploaded_file = st.file_uploader("📁 Upload your customer CSV file", type=["csv"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # 🎈 Streamlit built-in balloons
    st.balloons()

    st.dataframe(dataset.head())

    # --- Feature selection ---
    st.markdown('<div class="section-header">1️⃣ Select Features for Clustering</div>', unsafe_allow_html=True)
    numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.multiselect("Select 2 features", numeric_cols, default=numeric_cols[:2])

    if len(selected_features) != 2:
        st.warning("⚠️ Please select exactly 2 numerical features.")
    else:
        X = dataset[selected_features]

        # --- Choose cluster count ---
        st.markdown('<div class="section-header">2️⃣ Select Number of Clusters</div>', unsafe_allow_html=True)
        n_clusters = st.slider("Select K", 2, 10, 4)

        # --- Train model ---
        with st.spinner("⏳ Clustering in progress..."):
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            centroids = model.cluster_centers_

        dataset['Cluster'] = labels

        # 🎉 Confetti celebration after clustering
        components.html("""
        <canvas id="confetti-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9999;"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
        <script>
            var duration = 3 * 1000;
            var end = Date.now() + duration;
            (function frame() {
                confetti({ particleCount: 5, angle: 60, spread: 55, origin: { x: 0 } });
                confetti({ particleCount: 5, angle: 120, spread: 55, origin: { x: 1 } });
                if (Date.now() < end) requestAnimationFrame(frame);
            })();
        </script>
        """, height=0)

        # --- Visualization ---
        st.markdown('<div class="section-header">3️⃣ Cluster Visualization</div>', unsafe_allow_html=True)
        h = 0.2
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        z = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=plt.cm.Pastel2, origin='lower', aspect='auto')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=100, cmap='viridis', edgecolor='k')
        ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title("Customer Clusters")
        ax.legend()
        st.pyplot(fig)

        # --- Download ---
        st.markdown('<div class="section-header">📥 Download Results</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="download-section card">', unsafe_allow_html=True)
            csv = dataset.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV with Clusters", csv, file_name='clustered_customers.csv', mime='text/csv')
            
