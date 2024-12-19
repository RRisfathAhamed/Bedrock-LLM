from langchain.prompts import ChatPromptTemplate
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_DB_PATH = "job-db"

PROMPT_TEMPLATE = """
Fetch the necessary details from the following context the context is a Job Poster posted by Job Provider Therefore See the context in the sense of Resume :

{context}

---

{question}
"""


def query_rag_job():
    query_text = """


   [Your Role]

You are an AI assistant specializing in extracting information from job postings. Your task is to analyze the provided job posting text and populate a structured JSON object based on a predefined schema.

[Schema]

The schema you need to adhere to is as follows:

```json
{
    "JobID": "auto-generated",
    "ProviderID": "reference to JobProviders collection",
    "Title": "",
    "Description": "",
    "Location": {
        "City": "",
        "Country": ""
    },
    "Requirements": {
        "Education": [],
        "Skills": [],
        "Experience": null 
    },
    "PostedDate": "timestamp",
    "Applications": []
}

[Instructions]

Extract Information: Carefully read the job posting text and identify the relevant details to fill in the schema fields.
Populate JSON: Fill in the JSON object with the extracted information. If a field is not mentioned in the job posting, leave it as null.
Job Title: Extract the job title from the posting.
Job Description: Extract the full job description, including responsibilities, qualifications, and any other relevant details.
Location: Extract the city and country information. If only the city is mentioned, leave the country as null.
Requirements:
Education: Identify any educational qualifications or degrees mentioned.
Skills: Create a list of the required skills. Include both hard and soft skills if mentioned.
Experience: Extract the number of years of experience required. If not explicitly stated, leave it as null.
Output: Return the completed JSON object as your final output.


        """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text
