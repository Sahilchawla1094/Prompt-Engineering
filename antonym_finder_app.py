import streamlit as st
import os
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Set OpenAI API key (replace with your actual API key)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Streamlit interface setup
st.title("Antonym Finder")

# User input
user_input = st.text_input("Enter a word to find its antonym:", "")

if user_input:
    # First, create the list of few shot examples.
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    # Next, specify the template to format the examples.
    example_formatter_template = """Word: {word}
Antonym: {antonym}
"""

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_formatter_template,
    )

    # Create the `FewShotPromptTemplate` object.
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input\n",
        suffix=f"Word: {user_input}\nAntonym: ",
        input_variables=["input"],
        example_separator="\n",
    )

    # Initialize LLMChain with the prompt
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)

    # Execute the chain to get the antonym
    result = chain.run({'input': user_input})

    # Display the result
    st.write("Antonym:", result)
