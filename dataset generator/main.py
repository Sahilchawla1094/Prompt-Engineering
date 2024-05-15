import os
import pandas as pd
import requests
<<<<<<< HEAD
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
=======
import openai
from openai import OpenAI
import json
import time
import requests
from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI

# Api keys 
file = './.env' # new path to store new api_keys
keys = find_dotenv(file,raise_error_if_not_found=True)
load_dotenv(keys, override=True)

# Verification
# def verify_api_key(api_key, provider):
#     try:
#         if provider == "OpenAI":
#             headers = {"Authorization": f"Bearer {api_key}"}
#             url = "https://api.openai.com/v1/models"
#             response = requests.get(url, headers=headers)
#         elif provider == "Azure":
#             headers = {"api-key": api_key}
#             url = "https://ey-sandbox.openai.azure.com"  # Modify with your specific Azure endpoint
#             response = requests.get(url, headers=headers)
#         else:
#             return False
    
        
#         return response.status_code == 200
#     except Exception as e:
#         print(f"Error verifying API key for {provider}: {str(e)}")
#         return False
>>>>>>> 640a480e636308cd4770fc8589913378327393c5

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
<<<<<<< HEAD
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
=======

    # Select API provider
    api_provider = st.selectbox('Select your API provider:', ['OpenAI', 'Azure'])
    api_key = st.text_input('Enter your API Key:', type="password")
    # btn = st.button('Verify API Key')
    # if api_key and btn:
    #     if verify_api_key(api_key, api_provider):
    #         if api_provider == "OpenAI":
    #             os.environ['OPENAI_API_KEY'] = api_key 
    #         else: 
    #             os.environ['AZURE_API_KEY'] = api_key
    #         st.success('API Key verified!')
    #     else:
    #         st.error('Invalid API Key or verification failed.')
    #         return
        

    # Choose input method
    # api_key = "sk-proj-ajwKgSgObQnPITkIdZuyT3BlbkFJNU2KLr6Ozk7ZyVeVxuHn"
    # input_method = st.radio("Choose your input method:", ['Upload a file', 'Enter data manually'])

    # File uploader widget
    # Errorcode: AxiosError: Request failed with status code 403
    # streamlit run main.py --server.enableXsrfProtection false
    


    # File uploader widget
    prompt = ''
    columns = ''
    uploaded_file = ""

    # if input_method == "Upload a file":
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
                st.success("File processed successfully.")
                # st.write(columns)  # or handle as per your function's return type
            else:
                st.error("Unsupported file type or failed to process file.")
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
                
            

    # elif input_method == 'Enter data manually':
    #     # Manual data input
    #     columns = st.text_area("Enter column names and types separated by commas (e.g., name\:str, age\:int, salary\:float)")
    #     if columns:
    #         col_list = [col.strip().split(':') for col in columns.split(',') if ':' in col]
    #         if col_list:
    #             st.write("Columns specified:")
    #             # Create a dataframe to display the columns in table format
    #             df = pd.DataFrame(col_list, columns=['Column Name', 'Type'])
    #             st.table(df)

    num_records = st.number_input('Enter the number of records you want to generate:', min_value=1, value=10)

    # Optional scenario description
    # if st.checkbox('I want to provide a scenario description for the dataset'):
    #     scenario_description = st.text_area("Describe the scenario for which the dataset is being generated:")
    #     prompt += f"Scenario description: {scenario_description}"

        # if scenario_description:
            # st.write("Scenario Description Provided:")
            # st.write(scenario_description)
    # print("Prompt: ",prompt)
    
    if st.button('Generate Dataset'):
        with st.spinner('Generating dataset... Please wait'):
            # client initialize
            if api_provider == "OpenAI":            
                # Initialize OpenAI client
                client = OpenAI()
                client.api_key = api_key or os.getenv("OPENAI_API_KEY")
            elif api_provider == "Azure":
                # Initialize Azure OpenAI client
                client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=os.getenv("AZURE_OPENAI_API_KEY"), api_version="turbo-2024-04-09")
            
            prompt = f"Generate {num_records}: {columns}"
            
            total_data = []
            records_per_batch = 10  # Adjust based on trial and error
            batches = (num_records + records_per_batch - 1) // records_per_batch
            retries = 0

            for i in range(batches):
                while retries < 3:
                    try:
                        completion = client.chat.completions.create(
                            model= "gpt-3.5-turbo" if api_provider == "OpenAI" else "gpt-4",
                            messages=[
                                {"role": "system", "content": "Generate a JSON formatted dataset."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        batch_data = json.loads(completion.choices[0].message.content.strip())
                        total_data.extend(batch_data)
                        if len(total_data) >= num_records:
                            break
                        retries = 0  # Reset retry counter after a successful request
                    except json.JSONDecodeError as e:
                        st.error(f"JSON decoding failed: {e}")
                        st.error("Received data: " + total_data)  # Show data if JSON parsing fails
                        continue  # Skip this batch
                    except openai.RateLimitError:
                        st.warning("Rate limit exceeded, retrying in 30 seconds...")
                        time.sleep(30 * (2 ** retries))  # Exponential backoff
                        retries += 1
                    except Exception as e:
                        st.error(f"Error during data generation: {e}")
                        return
                if len(total_data) >= num_records:
                    break

            if len(total_data) < num_records:
                st.error("Failed to generate the requested number of records after several attempts.")
                return
            try:
                df = pd.DataFrame(total_data[:num_records])
                
                st.write("Preview of the first 5 records:")
                st.dataframe(df.head())
                
                # Convert DataFrame to CSV for download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Dataset", csv, "synthetic_data.csv", "text/csv", key='download-csv')
            except Exception as e:
                st.error(f"Error processing data: {e}")
                # st.stop()
                return
>>>>>>> 640a480e636308cd4770fc8589913378327393c5

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
