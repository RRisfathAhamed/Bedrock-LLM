from langchain.prompts import ChatPromptTemplate
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
import json

CHROMA_DB_PATH = "database"



PROMPT_TEMPLATE = """
Fetch the necessary details from the following context the context is a Resume of a Job Seeker Therefore See the context in the sense of Resume :

{context}

Output as a JSON
---

{question}
"""

CHAT_PROMPT_TEMPLETE='''
[Your Name]
Your Name is JOB_BOT , You are Responsible for Providing Insights about Jobs and Job Market
[Your Role]
You are an expert Job Advisor specializing in the current job market. Your goal is to provide clear, concise, and actionable advice to job seekers' questions.

[Job Seeker Details]
The Below are the Details of the Job Seeker
{details}

[Your Task]
1. Carefully review the job seeker's details to understand their situation.
2. Based on their details and the current job market trends, answer their question directly.
3. Keep your responses focused on job-related topics (e.g., job search strategies, industry trends, skill development).
4. Tailor your advice to the job seeker's experience level and career goals.
5. Offer specific and actionable suggestions whenever possible.
6. Be professional, supportive, and encouraging in your tone.

[Important Note]
Avoid providing information outside the scope of jobs and the job market.

[Job Seeker's Question]
{question} 

'''

FEEDBACK_PROMPT='''

[Your Responsibility]
You are Responsible for Providing a Detailed Report and Feedback on Job Skills

[Your Role]
You are an expert in skill assessment and career development. Your goal is to provide a thorough analysis and actionable feedback on the provided skills, identify areas for improvement, and offer insights on the job market relevant to the skillset.

[Skills Details]
The following are the existing skills:

{details}

[Your Task]

	1.	Carefully review the provided skills to understand the current abilities and expertise.
	2.	Provide a comprehensive analysis of strengths and how these align with current job market trends.
	3.	Identify specific areas where the skills can be improved to enhance employability.
	4.	Offer detailed suggestions on how to improve these skills, including recommended courses, certifications, or practical experiences.
	5.	Provide insights into the current job market situation, highlighting relevant industries and job roles that match these skills.
	6.	Tailor the feedback and suggestions to be actionable and relevant to the context of the provided skills.
	7.	Maintain a professional, supportive, and neutral tone throughout the feedback.

[Important Note]
Focus on providing constructive feedback related to skills and the job market. Avoid providing information outside this scope.

'''


def query_ragcv():
    query_text = """

[Your Role]

You are an AI assistant specializing in extracting information from resumes/CVs. Your task is to analyze the provided CV Data which is in the context text and populate a structured JSON object based on a predefined schema.

[Schema]

The schema you need to adhere to is as follows:

```json
{    
    "skills": [],
    "experience": {
        "Years": [], 
        "Details": [
            {
                "CompanyName": "",
                "Role": "",
                "Duration": "" 
            }
        ]
    },
    "education": [
        {
            "Degree": "",
            "Institution": "",
            "GraduationYear": null 
        }
    ],

}
[Instructions]
Extract Information: Carefully read the CV text and identify the relevant details to fill in the schema fields.
Populate JSON: Fill in the JSON object with the extracted information. If a field is not mentioned in the CV, leave it as null.
Skills List: Create a list of skills from the "Technical Skills" section of the CV Skills need to be seperated by comma.
Experience Details: Number of Years "example 2 Years" Extract company name, role, and duration (if available) from the "Working Experiences" section. If the duration is not explicitly stated, make a reasonable estimate based on the context it should be in number of years if No of Years Less than 1 then pointout Fresher.
Education Details: Extract degree, institution, and graduation year (if available) from the "Education" section.
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

    # Extract the JSON part from the response
    start_index = response_text.find("{")
    end_index = response_text.rfind("}") + 1
    json_string = response_text[start_index:end_index]

    # Convert the JSON string into an actual JSON object
    json_response = json.loads(json_string)

    return json_response


def chat(query_text,details):

    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    # results = db.similarity_search_with_score(query_text, k=5)
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLETE)
    prompt = prompt_template.format(question=query_text,details=details)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)

    # Extract the JSON part from the response


    return response_text

def gen_feedback(details):
    prompt_template = ChatPromptTemplate.from_template(FEEDBACK_PROMPT)
    prompt = prompt_template.format(details=details)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    return response_text


