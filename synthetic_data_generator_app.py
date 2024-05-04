import streamlit as st
import pandas as pd
import openai
from io import StringIO

# OpenAI API key (replace 'your_api_key' with your actual OpenAI API key)
openai.api_key = 'your_api_key'

def generate_synthetic_data(input_data, model="text-davinci-003", num_rows=10):
    try:
        # Convert input DataFrame to a single string prompt
        prompt = input_data.to_csv(index=False, header=True)
        
        # Append the instruction to the prompt
        prompt += "\nGenerate additional " + str(num_rows) + " rows based on the pattern above."
        
        # Call the OpenAI API
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=2048,  # Adjust based on needs
            n=1,
            stop=None,
            temperature=0.7
        )
        generated_text = response.choices[0].text.strip()

        # Convert generated text back to DataFrame
        synthetic_data = pd.read_csv(StringIO(generated_text), header=None)
        synthetic_data.columns = input_data.columns
        return synthetic_data
    except Exception as e:
        st.error(f"Failed to generate data: {e}")
        return pd.DataFrame()

def app():
    st.title('Synthetic Data Generator Using OpenAI')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original Data Preview:")
        st.write(data.head())

        # Number of rows to generate
        num_rows = st.number_input('Enter number of synthetic rows to generate:', min_value=1, value=5)
        
        if st.button('Generate Synthetic Data'):
            synthetic_data = generate_synthetic_data(data, num_rows=num_rows)
            st.write("Generated Synthetic Data Preview:")
            st.write(synthetic_data.head())

            # Convert DataFrame to CSV for download
            csv = synthetic_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download synthetic data",
                csv,
                "synthetic_data.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == '__main__':
    app()
