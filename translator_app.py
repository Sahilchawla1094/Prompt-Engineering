import streamlit as st
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Set OpenAI API key (replace with your actual API key)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Define the prompt template
template = '''In an easy way, translate the following sentence '{sentence}' into {target_language}.'''

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
language_prompt = PromptTemplate(
    input_variables=["sentence", 'target_language'],
    template=template,
)

# Streamlit interface
st.title("Sentence Translator")

# User inputs
sentence = st.text_input("Enter a sentence:", "Hello, how are you?")
target_language = st.selectbox("Select target language:", ["Hindi", "Spanish", "French"])

# When the button is clicked
if st.button("Translate"):
    formatted_prompt = language_prompt.format(sentence=sentence, target_language=target_language.lower())
    chain2 = LLMChain(llm=llm, prompt=formatted_prompt)
    
    # Execute the chain to get the translation
    result = chain2.run({'sentence': sentence, 'target_language': target_language.lower()})
    
    # Display the result
    st.write("Translation:", result)

