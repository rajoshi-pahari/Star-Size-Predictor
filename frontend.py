import streamlit as st
import pandas as pd
import requests
import time
import random
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Star Size Predictor 🌟",
    page_icon="🌟",
)

# List of 5 background image URLs (ensure they are direct image links)
background_urls = [
    "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8Z2FsYXh5fGVufDB8MXwwfHx8MA%3D%3D",  # Example image 1
    "https://images.unsplash.com/photo-1515825838458-f2a94b20105a?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fGdhbGF4eXxlbnwwfDF8MHx8fDA%3D",  # Example image 2
    "https://images.unsplash.com/photo-1540449078594-94d6173856c0?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fGdhbGF4eXxlbnwwfDF8MHx8fDA%3D",  # Example image 3
    "https://images.unsplash.com/photo-1728441212127-a13f6fc4fb25?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8bmVidWxhfGVufDB8MHwwfHx8MA%3D%3D",  # Example image 4
    "https://images.unsplash.com/photo-1507908708918-778587c9e563?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzR8fGdhbGF4eXxlbnwwfDF8MHx8fDA%3D",  # Example image 5
]

# Randomly select one background image URL
selected_bg = random.choice(background_urls)

# Inject custom CSS with the randomly selected background
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{selected_bg}") no-repeat center center fixed;
    background-size: cover;
    padding-bottom: 60px; 
    min-height: 100vh;  
    overflow-y: auto;
}}
[data-testid="stSidebar"] {{
    background: rgba(255, 255, 255, 0.7); /* Make sidebar semi-transparent */
}}

.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    z-index: 10;  /* Ensure footer is above other content */
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Now you can proceed with your app content
st.title("Welcome to Star Size Predictor 🌟")

st.markdown("""
    <div class="footer">
        <p>🌟 This project is developed by <b>Rajoshi Pahari</b> as part of the <b>ML4A Training Program</b> at Spartificial. 🚀</p>
    </div>
""", unsafe_allow_html=True)

button_style = """
<style>
button {
    background-color: #808080; /* Grey */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
}
.stButton>button:hover {
        background-color: #6e6e6e;  /* Darker gray when hovered */
        transform: scale(1.1);  /* Slightly enlarge button on hover */
    }
    
    .stButton>button:active {
        background-color: #555555;  /* Even darker gray when clicked */
        transform: scale(0.95);  /* Shrink button when clicked */
    }
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# API Base URL
BASE_URL = "https://star-size-predictor-bk3h.onrender.com/"  # Replace with your FastAPI server URL

st.markdown("Upload your dataset or generate a dataset to predict the size of a star based on its brightness!")

st.markdown(
    "How to use the app:\n"
    "1. If you have a dataset, upload it to quickly generate predictions and use the Plot the Linear Regression button to visualise it.\n"
    "2. If you want to generate a dataset, click on the Generate Dataset button and a dataset will be generated for you.\n"
    "3. Download this dataset and upload it to generate predictions and click on the Plot the Linear Regression button to visualise it.\n"
)

# File Upload Section
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Check if the uploaded file is a valid CSV
    if uploaded_file.type != "text/csv":
        st.error("Please upload a valid CSV file.")
    else:
        try:
            # Read the original dataset
            original_df = pd.read_csv(BytesIO(uploaded_file.getvalue()))  # Use BytesIO for uploaded file

            # Ensure the dataset contains 'Brightness' and 'True Size' columns
            if 'Brightness' not in original_df.columns or 'True Size' not in original_df.columns:
                st.error("The dataset must contain 'Brightness' and 'True Size' columns.")
            else:
                # Send the file to the FastAPI predict endpoint
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{BASE_URL}/predict/", files=files)

                if response.status_code == 200:
                    # Read the predicted data
                    predicted_file = BytesIO(response.content)
                    predicted_df = pd.read_csv(predicted_file)

                    # Clean up column names by stripping whitespace
                    original_df.columns = original_df.columns.str.strip()
                    predicted_df.columns = predicted_df.columns.str.strip()

                    # Display the original and predicted datasets side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Dataset")
                        st.dataframe(original_df)

                    with col2:
                        st.write("Predicted Dataset")
                        st.dataframe(predicted_df)

                else:
                    st.error("Failed to get predictions. Please try again.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
        
# Synthetic Dataset Creation Section
st.header("Create Synthetic Dataset")
rows = st.number_input("Number of stars", min_value=10, max_value=10000, value=100, step=10)

if st.button("Generate Dataset", key="generate_dataset"):
    # Request the dataset from FastAPI
    response = requests.post(f"{BASE_URL}/create_dataset/", params={"rows": rows})
    if response.status_code == 200:
        synthetic_file = BytesIO(response.content)
        synthetic_df = pd.read_csv(synthetic_file)

        # Display the synthetic dataset
        st.subheader("Generated Synthetic Dataset")
        st.dataframe(synthetic_df)

        # Allow the user to download the file
        st.download_button(
            label="Download Dataset",
            data=synthetic_file.getvalue(),
            file_name="synthetic_data.csv",
            mime="text/csv",
        )
    else:
        st.error("Failed to generate dataset. Please try again.")

# Plotting Section
st.header("Plot the Linear Regression")
if st.button("Plot the Linear Regression", key="plot_button"):
    st.write("Generating Linear Regression Plot...")
    
    # Example: Plotting Linear Regression on synthetic data
    # Ensure you have some columns like 'Brightness' and 'True Size' to plot
    if uploaded_file is not None:
    # Read the uploaded file as a DataFrame
        data = pd.read_csv(uploaded_file)

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Create the seaborn lmplot on the created figure
        sns.regplot(x="Brightness", y="True Size", data=data, ax=ax, line_kws={"color": "red"})

        # Set the title and labels
        ax.set_title("Linear Regression: Brightness vs. True Size")
        ax.set_xlabel("Brightness")
        ax.set_ylabel("True Size")

        # Display the plot in Streamlit
        st.pyplot(fig)
    
    else:
        st.error("Please upload a dataset first.")
        
PREDICT_ENDPOINT = "https://star-size-predictor-bk3h.onrender.com/predict/"
PLOT_ENDPOINT = "https://star-size-predictor-bk3h.onrender.com/plot/"
CREATE_DATASET_ENDPOINT = "https://star-size-predictor-bk3h.onrender.com/create_dataset/"
    