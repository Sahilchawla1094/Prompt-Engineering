import streamlit as st
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Set OpenAI API key (ensure you have a valid key set here)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Define the prompt template
demo_template = '''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt = PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
chain1 = LLMChain(llm=llm, prompt=prompt)

# Streamlit user interface
st.title("Financial Concept Explainer")

# User input for the financial concept
financial_concept = st.text_input("Enter a financial concept:", "GDP")

if st.button("Explain"):
    # Format the prompt with the user's input
    formatted_prompt = prompt.format(financial_concept=financial_concept)
    
    # Run the chain to generate the explanation
    result = chain1.run(financial_concept)
    
    # Display the result
    st.text_area("Explanation", value=result, height=300)
