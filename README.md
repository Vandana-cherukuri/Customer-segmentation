# Customer-segmentation
📊 Customer Segmentation App
An interactive and visually rich Streamlit web application for customer segmentation using the KMeans Clustering Algorithm. Upload your customer data, select features, perform clustering, and download the results—all with celebratory visuals and elegant animations!

🚀 Features
✅ Upload your own customer CSV file

🧠 Select two numerical features for clustering

🔢 Choose number of clusters (K)

📈 Visualize clustering results with centroids

🎉 Confetti celebration upon clustering

📥 Download clustered dataset as CSV

📖 Slide-in "About" panel to explain the model

💅 Custom styling, animations, and hover effects

📦 Tech Stack
Streamlit

Pandas

NumPy

Matplotlib

Scikit-learn

HTML/CSS for custom layout and animation

🖼️ App Interface
<!-- Add your own screenshot here -->

🛠️ How to Run the App Locally
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/customer-segmentation-app.git
cd customer-segmentation-app
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy
Edit
streamlit
pandas
numpy
matplotlib
scikit-learn
3. Run the App
bash
Copy
Edit
streamlit run app.py
📁 Input File Format
The app accepts CSV files with numerical features.
Ensure your file has at least two numerical columns to enable clustering.

Example:

CustomerID	Age	Annual Income (k$)	Spending Score (1-100)
1	19	15	39
2	21	15	81

📤 Output
After clustering, you'll receive a downloadable CSV file containing a new Cluster column indicating the cluster each customer belongs to.

ℹ️ About the Model
KMeans is an unsupervised learning algorithm that partitions data into K clusters by minimizing the intra-cluster variance. Each cluster has a centroid that represents the average of the points within it.

🎨 Design & Animation
💚 Animated titles and section headers

📜 Slide-in info panel for explanations

🥳 Balloons and confetti for visual feedback

🎨 Responsive layout with CSS enhancements

🧑‍💻 Author
Your Name
vandana-cherukuri

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
