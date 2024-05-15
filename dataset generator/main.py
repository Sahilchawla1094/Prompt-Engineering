import os
import pandas as pd
import requests
import json
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Setup environment
load_dotenv(find_dotenv())

# Verification of API Key using Azure OpenAI
def verify_api_key(api_key, endpoint):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{endpoint}/v1/engines"  # Example endpoint to list available engines
    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error verifying API key with Azure OpenAI: {str(e)}")
        return False

# File loading based on extension
def load_document(file_path):
    from langchain_community import document_loaders
    ext = os.path.splitext(file_path)[-1]
    loader_map = {
        '.pdf': document_loaders.PyPDFLoader,
        '.docx': document_loaders.Docx2txtLoader,
        '.txt': document_loaders.TextLoader,
        '.xlsx': document_loaders.UnstructuredExcelLoader,
        '.csv': document_loaders.CSVLoader,
        '.html': document_loaders.BSHTMLLoader,
    }
    if ext in loader_map:
        return loader_map[ext](file_path).load()
    else:
        st.error("Unsupported file type.")
        return None

# Main function with Streamlit UI
def main():
    st.title('Synthetic Dataset Generator')
    api_provider = st.selectbox('Select your API provider:', ['Select an API', 'Azure OpenAI'])
    api_key = ''
    endpoint = ''
    if api_provider == 'Azure OpenAI':
        api_key = st.text_input('Enter your API Key:', type="password")
        endpoint = st.text_input('Enter your Endpoint URL:')
        if st.button('Verify API Key'):
            if api_key and endpoint and verify_api_key(api_key, endpoint):
                st.success('API Key verified!')
            else:
                st.error('Invalid API Key or Endpoint')

    input_method = st.radio("Choose your input method:", ['Upload a file', 'Enter data manually'])
    columns = ''
    if input_method == "Upload a file":
        uploaded_file = st.file_uploader("Upload your file:", type=['pdf', 'docx', 'txt', 'html', 'csv', 'xlsx'])
        if uploaded_file:
            file_path = os.path.join('./files', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            document = load_document(file_path)
            if document:
                st.success("File processed successfully.")
                columns = str(document)  # Assuming the document loader returns metadata or text that can be useful
    else:
        columns = st.text_area("Enter column names and types separated by commas (e.g., name:str, age:int, salary:float)")

    num_records = st.number_input('Enter the number of records you want to generate:', min_value=1, value=10)
    if st.checkbox('Provide a scenario description'):
        scenario_description = st.text_area("Describe the scenario for which the dataset is being generated:")
        prompt = f"Generate {num_records} records for a dataset with columns: {columns}. Scenario: {scenario_description}"
    else:
        prompt = f"Generate {num_records} records for a dataset with columns: {columns}."

    if st.button('Generate Dataset'):
        st.spinner('Generating dataset... Please wait')
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "davinci",  # Example model, adjust as needed
            "prompt": prompt,
            "max_tokens": 1024
        }
        response = requests.post(f"{endpoint}/v1/completions", headers=headers, json=data)

        try:
            data = json.loads(response.text)
            df = pd.DataFrame(data['choices'][0]['text'])
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Dataset", csv, "synthetic_data.csv", "text/csv", key='download-csv')
        except Exception as e:
            st.error(f"Failed to generate dataset: {e}")

if __name__ == "__main__":
    main()
