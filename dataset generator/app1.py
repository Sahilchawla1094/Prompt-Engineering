import streamlit as st
import os
import pandas as pd
import openai
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# from openai import AzureOpenAI # 
from langchain_openai import AzureOpenAI
from langchain_openai import ChatOpenAI # from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import create_model
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX

# Set API Key securely
api_key = ""
azure_endpoint = ""
azure_api_version = ""
azure_deployment_name = ""
synthetic_results = ""

# Define a generator function for AzureChatOpenAI similar to the OpenAI one
# def create_azure_data_generator(output_schema, client, subject, extra, runs, azure_deployment_name):
#     prompt = f"""
#     Generate {runs} instances of JSON formatted dataset based on these specifications:
#     Subject: {subject}
#     Extra requirement: {extra}
#     Each instance should be unique and follow the schema provided.
#     """
#     completion = client.chat.completions.create(
#         model=azure_deployment_name,
#         temperature=1, 
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=1500,  # Adjust based on the complexity of the output schema
#         n=1,  # Adjust if multiple completions are needed in a single call
#         stop=None  # Define as necessary
#     )
    
#     # Parse the generated JSON content from the completion
#     # Assuming that the content is properly formatted JSON representing a list of entries
#     print(completion)
#     data = json.loads(completion.choices[0].message.content)
#     return [output_schema(**item) for item in data]  # Create a new model instance for each item in the list


st.set_page_config(page_title="Synthetic Data Generator")
with st.sidebar.title("Settings"):
    api_provider = st.selectbox('Select your API provider:', ['OpenAI', 'Azure OpenAI'])

    if api_provider == "Azure OpenAI":
        api_key = st.sidebar.text_input("API Key", type="password")
        azure_endpoint = st.sidebar.text_input("Azure OpenAI Endpoint")
        azure_api_version = st.sidebar.text_input("API Version")
        azure_deployment_name = st.sidebar.text_input("Deployment Name")
        os.environ["OPENAI_API_KEY"] = api_key  # 4b81012d55fb416c9e398f6149c3071e
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint # https://ey-sandbox.openai.azure.com/
        os.environ['AZURE_API_VERSION'] = azure_api_version # 2023-05-15 # turbo-2024-04-09
        os.environ['AZURE_DEPLOYMENT_NAME'] = azure_deployment_name # gpt-4
    elif api_provider == "OpenAI":
        api_key = st.sidebar.text_input("API Key", type="password")
        os.environ["OPENAI_API_KEY"] = api_key  # sk-proj-ajwKgSgObQnPITkIdZuyT3BlbkFJNU2KLr6Ozk7ZyVeVxuHn
    

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Sample:")
    st.write(data.head())
    st.write("Data shape:", data.shape)

    # Dynamically create a data model based on the uploaded CSV
    data_columns = {col: float if data[col].dtype in ['float64', 'int64'] else str for col in data.columns}
    DynamicModel = create_model('DynamicModel', **{col: (float, ...) if typ == float else (str, ...) for col, typ in data_columns.items()})

    # Convert first few rows into examples for the synthetic data generator
    examples = [{"example": ", ".join([f"{col}: {row[col]}" for col in data.columns])} for _, row in data.head().iterrows()]

    prompt_template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject", "extra"],
        example_prompt=PromptTemplate(input_variables=["example"], template="{example}"),
    )
    
    # User input for number of synthetic rows
    num_rows = st.number_input('Enter the number of synthetic rows to generate', min_value=1, value=10)

    # Button to generate data
    if st.button('Generate Synthetic Data'):
        with st.spinner('Generating dataset... Please wait'):
            if api_provider == "OpenAI":
                synthetic_data_generator = create_openai_data_generator(
                    output_schema=DynamicModel,
                    llm= ChatOpenAI(api_key=api_key),
                    prompt=prompt_template,
                )
                
                
            else:
                # Initialize the Azure LLM client with the appropriate parameters
                # client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=azure_api_version)
                
                # synthetic_data_generator = create_openai_data_generator(
                #     output_schema=DynamicModel,
                #     llm=openai.completions.create(model="text-davinci-003",temperature=0.7,max_tokens=512, prompt="generate pandas dataframe", stop=None),
                #     client=client,
                #     prompt=prompt_template,
                # )
                
                # # Generate synthetic data using the adjusted Azure client interaction
                # synthetic_results = create_azure_data_generator(
                #     output_schema=DynamicModel,
                #     prompt=prompt_template,
                #     client=client,
                #     subject='custom_data',
                #     extra="the name must be chosen at random. Make it something you wouldn't normally choose.",
                #     runs=num_rows,
                #     azure_deployment_name=azure_deployment_name
                # )
                synthetic_data_generator = SyntheticDataGenerator(
                    template=prompt_template,
                    llm = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=azure_api_version, model=azure_deployment_name),
                )
                
                synthetic_results = synthetic_data_generator.generate(
                        subject="custom_data",
                        extra="the name must be chosen at random. Make it something you wouldn't normally choose.",
                        runs=num_rows,  # Use user-defined number of rows
                )

            
            # Prepare the data for display
            synthetic_data = [{col: getattr(item, col) for col in data_columns} for item in synthetic_results]
            synthetic_df = pd.DataFrame(synthetic_data)

            
            # Format the DataFrame to display proper data types
            for col in synthetic_df.columns:
                synthetic_df[col] = synthetic_df[col].astype(data.dtypes[col])
            
            st.write(synthetic_df)
            st.write("Data shape:", synthetic_df.shape)
            st.download_button("Download Dataset", synthetic_df.to_csv(index=False).encode('utf-8'), "synthetic_data.csv", "text/csv", key='download-csv')