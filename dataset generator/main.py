import streamlit as st
import os
import pandas as pd
import requests
from openai import OpenAI
import json
from dotenv import find_dotenv, load_dotenv

# Api keys 
file = './.env' # new path to store new api_keys
keys = find_dotenv(file,raise_error_if_not_found=True)
load_dotenv(keys, override=True)

# Verification
def verify_api_key(api_key, provider):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    if provider == "OpenAI":
        # Example endpoint for OpenAI
        url = "https://api.openai.com/v1/models"
    elif provider == "Google API":
        # Example Google API endpoint
        url = "https://www.googleapis.com/auth/cloud-platform"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    else:
        return False

    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Error verifying API key for {provider}: {str(e)}")
        return False

# Load the document
def load_document(file):
    """Functions links the PDFs using library called PyPDF into an array of documents where each document contains the page, content and metadata with a page number."""
    # prevent from circular dependencies and benefit from a more reliable refactoring of our code. | if we utilize the function and it will work because it contains everthing it int
    # from langchain.document_loaders import PyPDFLoader 
    from langchain_community import document_loaders
    
    # import json
    # from pathlib import Path
    name, ext = os.path.splitext(file) # file.split('/')[-2], file.split('/')[-1]
    print(name, ext)
    
    loader = {
        '.pdf': document_loaders.PyPDFLoader(file),
        '.docx': document_loaders.Docx2txtLoader(file),
        '.txt': document_loaders.TextLoader(file),
        '.xlsx': document_loaders.UnstructuredExcelLoader(file, mode="elements"),
        '.csv': document_loaders.CSVLoader(file),
        '.html': document_loaders.BSHTMLLoader(file), # UnstructuredHTMLLoader(file)
        # '.py': document_loaders.PythonLoader(file), # security concerns
        # '.json': json.loads(Path(file).read_text())
    } # url of the file or file path in a file system

    if ext not in loader.keys():
        print("Extension Doesn't Exists!")
        return None
    
    if ext == '.json':
        return loader[ext]
    
    print(f"Loading the '{file}'")
    data = loader[ext].load_and_split() if ext == '.pdf' else loader[ext].load() # this will return a list of langchain documents, one document for each page
    return data # data is splitted by pages and we can use indexes to display a specific page

# Define the function to ensure the directory exists
def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create it if it does not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    st.title('Synthetic Dataset Generator')

    # Select API provider
    api_provider = st.selectbox('Select your API provider:', ['Select an API', 'OpenAI', 'Google API'])
    
    # Conditional rendering of API Key input based on provider selection
    if api_provider != 'Select an API':
        api_key = st.text_input('Enter your API Key:', type="password")
        if st.button('Verify API Key'):
            if api_key and verify_api_key(api_key, api_provider):
                st.success('API Key verified!')
            else:
                st.error('Invalid API Key')

    # Choose input method
    input_method = st.radio("Choose your input method:", ['Upload a file', 'Enter data manually'])

    # File uploader widget
    # Errorcode: AxiosError: Request failed with status code 403
    # streamlit run main.py --server.enableXsrfProtection false
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your files:", type=['pdf', 'docx', 'txt', 'html', 'csv', 'xlsx'], accept_multiple_files=False)

    if uploaded_file:
        try:
            # Create a directory for files if it doesn't exist
            ensure_directory_exists('./files')

            # Construct file path
            file_path = os.path.join('./files', uploaded_file.name)
            
            # Write file to disk
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Load and process the document
            columns = load_document(file_path)
            if columns is not None:
                st.write("File processed successfully.")
                # st.write(columns)  # or handle as per your function's return type
            else:
                st.error("Unsupported file type or failed to process file.")
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")

    elif input_method == 'Enter data manually':
        # Manual data input
        columns = st.text_area("Enter column names and types separated by commas (e.g., name\:str, age\:int, salary\:float)")
        if columns:
            col_list = [col.strip().split(':') for col in columns.split(',') if ':' in col]
            if col_list:
                st.write("Columns specified:")
                # Create a dataframe to display the columns in table format
                df = pd.DataFrame(col_list, columns=['Column Name', 'Type'])
                st.table(df)

    num_records = st.number_input('Enter the number of records you want to generate:', min_value=1, value=10)

    # Optional scenario description
    if st.checkbox('I want to provide a scenario description for the dataset'):
        scenario_description = st.text_area("Describe the scenario for which the dataset is being generated:")
        if scenario_description:
            st.write("Scenario Description Provided:")
            st.write(scenario_description)
            
    if st.button('Generate Dataset'):
        with st.spinner('Generating dataset... Please wait'):
            prompt = f"Generate {num_records} records for a dataset with the following columns and types: {columns}. Scenario description: {scenario_description}"
            
            # Initialize OpenAI client
            client = OpenAI()
            client.api_key = api_key or os.getenv("OPENAI_API_KEY")

            # Sending prompt to OpenAI's GPT-3.5-turbo
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are responsible for dataset generation and designed to output in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )

            try:
                # Parsing the JSON data from the response
                response = completion.choices[0].message.content.strip()  # Getting the actual text response
                # st.write("Raw response:", response)  # Display raw response for debugging
                response_data = json.loads(response)  # Parse the complete JSON string
                # st.write("Parsed JSON:", response_data)  # Display parsed JSON for debugging

                # Check if the JSON response is a dictionary containing an array or a direct array
                if isinstance(response_data, dict):
                    # Try to find the first list in the dictionary values and assume it's the dataset
                    for key, value in response_data.items():
                        if isinstance(value, list):
                            data = value
                            break
                    else:
                        st.error("No list found in the JSON object.")
                        # st.stop()
                        return
                elif isinstance(response_data, list):
                    data = response_data  # Directly use the list as data
                else:
                    st.error("JSON does not contain a list or an object containing a list.")
                    # st.stop()
                    return

                df = pd.DataFrame(data)
                st.write("Preview of the first 5 records:")
                st.dataframe(df.head())

                # Convert DataFrame to CSV for download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Dataset", csv, "synthetic_data.csv", "text/csv", key='download-csv')
            except Exception as e:
                st.error(f"Error processing data: {e}")
                # st.stop()
                return


if __name__ == "__main__":
    main()