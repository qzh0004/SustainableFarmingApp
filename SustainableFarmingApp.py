# -*- coding: utf-8 -*-
"""
@author: Minlu Wang-He
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Custom CSS for consistent button styling
button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50; /* Green color */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 50px;
        width: 100%;
    }
    </style>
"""

def main():
    st.markdown(button_style, unsafe_allow_html=True)  # Apply custom button styling

    st.title("SoilXpert")

    # Upload dataset
    st.markdown("<h3 style='font-size:20px;'>Upload Bacteria Count Dataset (CSV)</h3>", unsafe_allow_html=True)
    dataset_file = st.file_uploader("", type=["csv"])  # Empty label to avoid redundancy
    if dataset_file is not None:
        try:
            df = pd.read_csv(dataset_file)
            st.success("Dataset loaded successfully!")
            st.write("Preview of the bacteria count dataset:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Upload model
    st.markdown("<h3 style='font-size:20px;'>Upload Machine Learning Model (PKL)</h3>", unsafe_allow_html=True)
    model_file = st.file_uploader("", type=["pkl"])  # Empty label to avoid redundancy
    if model_file is not None:
        try:
            model = pickle.load(model_file)
            st.success("Model loaded successfully!")
            st.write("Model details:")
            st.write(model)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # Perform prediction
    if st.button("Prediction Crop Yield"):
        # Check if dataset and model are uploaded
        if dataset_file is None or model_file is None:
            st.error("Please upload both the dataset and the model before performing predictions.")
        else:
            try:
                # Log-normalize the dataset
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_log_normed = df.copy()
                df_log_normed[numeric_cols] = np.log(df_log_normed[numeric_cols] + 1)

                # Perform prediction
                predictions = model.predict(df_log_normed)

                # Visualization
                sample_mean = 0.75
                x_labels = [f"Farm {i+1}" for i in range(len(predictions))]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(x_labels, predictions, color='blue', width=0.4, label='Predicted Yield')
                ax.axhline(y=sample_mean, color='k', linestyle='--', label='Average Farm Yield')
                ax.set_xlabel('Farms')
                ax.set_ylabel('Predicted Yield (Tonne/Hectare)')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

                # Display predictions
                st.success("Predictions completed successfully!")
                st.write(pd.DataFrame(predictions, columns=['Predicted Yield']))
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()