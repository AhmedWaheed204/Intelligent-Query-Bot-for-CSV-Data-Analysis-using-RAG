import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import HumanMessage
from langchain.schema import (
    SystemMessage,
    AIMessage
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
from streamlit_chat import message
import re

# Set up environment variables
os.environ['API_KEY'] = "AIzaSyAY8h1ZjOjSAXGicPVr9ENV9Q7R31R-14Q"

# Updated prompt template for converting text to SQL
template = """You are an AI assistant knowledgeable about the data from the uploaded CSV file. Your task is to convert user queries into SQL queries that can be executed against the database.

Here are some examples of how you should convert text queries to SQL:
1. "What is the average value of the 'Age' column?" -> "SELECT AVG(Age) FROM uploaded_data;"
2. "How many entries have a value greater than 50 in the 'Score' column?" -> "SELECT COUNT(*) FROM uploaded_data WHERE Score > 50;"

If the query is not related to SQL or the dataset, you should handle it as a general conversation and provide a thoughtful response.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

def clean_sql_query(sql_query):
    # Remove code block formatting and extra spaces
    cleaned_query = re.sub(r'```sql|```', '', sql_query).strip()
    return cleaned_query

def validate_columns(sql_query, valid_columns):
    """Check if all column names in the SQL query are valid."""
    # Extract potential column names from the query using a regex pattern
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    query_columns = re.findall(pattern, sql_query)

    # Find columns that are not in the valid columns list
    invalid_columns = [col for col in query_columns if col not in valid_columns]

    if invalid_columns:
        return False, invalid_columns
    return True, []

def main(api_key, template):
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro", temperature=0.7)

    # Streamlit UI
    st.header("Your AI Home Assistant 🤖")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Dataset Shape: {df.shape}")
            st.write(f"Dataset Columns: {df.columns.to_list()}")

            # Save the dataset to a SQLite database
            engine = create_engine("sqlite:///uploaded_data.db")
            df.to_sql("uploaded_data", engine, index=False, if_exists="replace")

            # Connect to the SQL database
            db = SQLDatabase(engine=engine)

            # Create SQL agent
            agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

        except Exception as e:
            st.write(f"An error occurred while processing the file: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # Update the prompt template to use only history and input variables
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)
    conversation = ConversationChain(
        prompt=prompt_template,
        llm=llm,
        memory=memory,
        verbose=True
    )

    with st.sidebar:
        user_input = st.text_input("Enter your Question:", key="input")
        if st.button("Ask"):
            if uploaded_file is not None and user_input:
                # Update conversation history
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("Generating SQL query..."):
                    # Generate SQL query from text input
                    sql_query = conversation.predict(input=st.session_state.messages)
                    # Clean up the SQL query
                    sql_query = clean_sql_query(sql_query)
                    
                    # Validate the SQL query
                    if sql_query.lower().startswith("select"):
                        # Validate the columns in the SQL query
                        is_valid, invalid_columns = validate_columns(sql_query, df.columns)
                        if is_valid:
                            try:
                                st.write(f"Generated SQL Query: {sql_query}")

                                # Execute the SQL query
                                result = pd.read_sql_query(sql_query, engine)
                                if result.empty:
                                    st.write("No results found.")
                                else:
                                    st.write("Query Result:")
                                    st.dataframe(result)  # Display as a dataframe
                            except Exception as e:
                                st.write(f"An error occurred while running the query: {e}")
                        else:
                            st.write(f"The query contains invalid column names: {', '.join(invalid_columns)}. Please check the column names and try again.")
                    else:
                        st.write("The generated query is not a valid SELECT query. Please try a different question.")

                    # Append the AI's response to the conversation history
                    if isinstance(sql_query, str):
                        st.session_state.messages.append(AIMessage(content=sql_query))
                    else:
                        st.session_state.messages.append(AIMessage(content="Could not generate a valid SQL query."))
                
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main(api_key=os.environ['API_KEY'], template=template)
