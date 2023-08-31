from transformers import TapexTokenizer, BartForConditionalGeneration, pipeline
import pandas as pd
import streamlit as st
from dotenv import load_dotenv #for lang chain
import torch
from langchain import SQLDatabase, SQLDatabaseChain, LlamaCpp
import os
import psycopg2
import streamlit

def create_sql_table(query):

    conn = psycopg2.connect(
        host="xxx",
        database="xx",
        user="xx",
        password="xxx"
    )

    data_from_db = pd.read_sql_query(query, conn)
    conn.close()
    data_df = pd.DataFrame(data_from_db)
    return data_df

def llama(query):
    db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:xxx@xxx/xxx",
    include_tables=['patients'],
    sample_rows_in_table_info=5)
    
    llm = LlamaCpp(
    model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin",
    temperature=0.75,
    max_tokens=5000,
    top_p=1,
    n_ctx=10000,
    #callback_manager=callback_manager,
    verbose=True,)
    
    # Create db chain
    QUERY = """
    Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    {question}
    """
    question = QUERY.format(question=query)
    # Setup the database chain
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    return(db_chain.run(question))



def tapas(query, sql_query):
    tqa = pipeline(task="table-question-answering",
                   model="google/tapas-base-finetuned-wtq")
    result = tqa(create_sql_table(sql_query), query=query)["answer"]
    return result

def tapex(query, sql_query):
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    encoding = tokenizer(table=create_sql_table(sql_query), query=query, return_tensors="pt")
    outputs = model.generate(**encoding)
    # Sonuçları işleme ve yazdırma
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results

def main():
    load_dotenv() 
    st.set_page_config(page_title="TAPAS AND TAPEX", page_icon=":books:")
    
    
    st.subheader("Your Documents")
    option = st.selectbox(
                        'Which model do you want to use?',
                        ('TAPEX', 'TAPAS','LLAMA2'))
    sql_query = st.text_input('SQL Query', 'SELECT  Name,Surname,Gender,FatherName,BirthPlace FROM Patients LIMIT 30')

    question = st.text_input('Question', 'Is there a any patient which name is Umur?')

    if st.button("Ask"):
        with st.spinner("Processing"):
            # get pdf text 
            if option == 'TAPEX':
                st.success(tapex(question, sql_query))
            elif option == 'TAPAS':
                st.success(tapas(question, sql_query))
            else:
                st.success(llama(question))



if __name__ == '__main__':
    main()