import streamlit as st
import os
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI #from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import create_model
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX

# Set API Key securely
st.set_page_config(page_title="Synthetic Data Generator")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Sample:")
    st.write(data.head())

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
        st.spinner('Generating dataset... Please wait')
        synthetic_data_generator = create_openai_data_generator(
            output_schema=DynamicModel,
            llm=ChatOpenAI(temperature=1),
            prompt=prompt_template,
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

# Run this with 'streamlit run your_script_name.py'
