import time
import logging
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from get_model import get_bedrock_model

# Database URI
db_uri_postgres = "postgresql+psycopg2://postgres:123@localhost:5432/hiredb"
db = SQLDatabase.from_uri(db_uri_postgres)

# Initialize LLM
llm = get_bedrock_model()

# Prompt Templates
templates = {
    'low': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: Given the SQL schema, return the job_ids from the job table where at least 3 of the provided skills {skills} Use the following format:
    Example Query: SELECT j.job_id FROM job j JOIN application a ON j.job_id = a.job_id JOIN jobseeker js ON a.job_seeker_id = js.job_seeker_id WHERE string_to_array(js.skills, ',') && ARRAY['Python', 'Java', 'C', 'JavaScript', 'Spring Boot', 'Flask', 'TensorFlow', 'Bootstrap', 'ReactJS', 'Scikit-learn', 'Pandas', 'NumPy', 'Git', 'GitHub', 'JIRA', 'Photoshop', 'Cloud Technologies', 'Virtual and Augmented Reality', 'Blockchain', 'Prompt Engineering', 'IoT', 'Sports Analytics'];
    Generate a very similar query with the given {skills} replacing the placeholder values. Note that string_to_array(js.skills, ',') part is Must needed because It need to convert to array (without any explanations or formatting make a cumpulsory single line query):
    ''',
    'medium': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} match with the job's skills.

    Postgres SQL Query (without any explanations or formatting make a single line query):
    ''',
    'intermediate': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: Given the SQL schema, return the job_ids from the job table where at least 3 of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column. Use the following format:
    Example Query: SELECT j.job_id FROM job j JOIN application a ON j.job_id = a.job_id JOIN jobseeker js ON a.job_seeker_id = js.job_seeker_id WHERE string_to_array(js.skills, ',') && ARRAY['Python', 'Java', 'C', 'JavaScript', 'Spring Boot', 'Flask', 'TensorFlow', 'Bootstrap', 'ReactJS', 'Scikit-learn', 'Pandas', 'NumPy', 'Git', 'GitHub', 'JIRA', 'Photoshop', 'Cloud Technologies', 'Virtual and Augmented Reality', 'Blockchain', 'Prompt Engineering', 'IoT', 'Sports Analytics'] AND js.experience LIKE '%2 Year%';
    Generate a very similar query with the given {skills} and {experience} replacing the placeholder values. Note that string_to_array(js.skills, ',') part is Must needed because It need to convert to array (without any explanations or formatting make a cumpulsory single line query):
    ''',
    'high': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column.

    SQL Query (without any explanations or formatting make a single line query):
    '''
}

def get_schema(_):
    """Get the schema of the database."""
    return db.get_table_info()


def invoke_bedrock_with_retry(prompt_chain, inputs, max_retries=5, initial_wait=1):
    """
    Invoke Bedrock model with retry logic.
    """
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return prompt_chain.invoke(inputs)
        except Exception as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"ThrottlingException: Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                logging.error(f"Error invoking Bedrock: {e}")
                raise
    raise RuntimeError("Exceeded maximum retries while invoking Bedrock.")


def generate_sql(template_name, skills, experience=None):
    """
    Generate SQL query using the LLM and prompt templates.
    """
    prompt = ChatPromptTemplate.from_template(templates[template_name])
    sql_chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )

    inputs = {"skills": skills}
    if experience:
        inputs["experience"] = experience

    generated_sql = invoke_bedrock_with_retry(sql_chain, inputs)

    if 'SQL Query:' in generated_sql:
        generated_sql = generated_sql.split('SQL Query:')[1].strip()

    return generated_sql


def run_query(query):
    """
    Run the SQL query against the database.
    """
    return db.run(query)


# Functions for different match types

def return_low_matched_jobs(skills):
    sql_query = generate_sql('low', skills)
    logging.info(f"Low matched query: {sql_query}")
    return run_query(sql_query)


def return_medium_matched_jobs(skills):
    sql_query = generate_sql('medium', skills)
    logging.info(f"Medium matched query: {sql_query}")
    return run_query(sql_query)


def return_intermediate_matched_jobs(skills, experience):
    sql_query = generate_sql('intermediate', skills, experience)
    logging.info(f"Intermediate matched query: {sql_query}")
    return run_query(sql_query)


def return_high_matched_jobs(skills, experience):
    sql_query = generate_sql('high', skills, experience)
    logging.info(f"High matched query: {sql_query}")
    return run_query(sql_query)


# Example usage
if __name__ == "__main__":
    skills = "Python, Java, SQL"
    experience = "2 Years"

    try:
        low_matched_jobs = return_low_matched_jobs(skills)
        print(f"Low Matched Jobs: {low_matched_jobs}")

        intermediate_matched_jobs = return_intermediate_matched_jobs(skills, experience)
        print(f"Intermediate Matched Jobs: {intermediate_matched_jobs}")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
