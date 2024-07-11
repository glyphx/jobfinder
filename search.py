import requests
from bs4 import BeautifulSoup
import openai
import pinecone
import time

openai.api_key = "YOUR_OPENAI_API_KEY"
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")  # Use your Pinecone environment

index_name = 'job-search'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # dimension should match OpenAI's embedding dimension
index = pinecone.Index(index_name)

def load_resume(file_path):
    with open(file_path, 'r') as file:
        resume_content = file.read()
    return resume_content

def generate_job_search_query(job_title, platform):
    prompt = f"Generate a search query to find remote '{job_title}' job listings on {platform}."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def search_platform(platform, query):
    if platform == "Indeed":
        url = f"https://www.indeed.com/jobs?q={query}&l=Remote"
    elif platform == "ZipRecruiter":
        url = f"https://www.ziprecruiter.com/candidate/search?search={query}&location=Remote"
    elif platform == "SimplyHired":
        url = f"https://www.simplyhired.com/search?q={query}&l=Remote"
    elif platform == "AngelList":
        url = f"https://angel.co/jobs?remote=true&query={query}"
    elif platform == "FlexJobs":
        url = f"https://www.flexjobs.com/search?location=Remote&search={query}"
    else:
        return None
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None

def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def parse_job_listings_with_openai(html_content, resume_content):
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(separator='\n')

    prompt = f"""
    Extract job listings from the following text. Each listing should include title, company, location, salary, link, summary, match, and justification. Format the output in YAML.

    Resume:
    {resume_content}

    Example input:
    Software Engineer | Google | Remote | $120,000 - $150,000 | https://jobs.google.com/software-engineer | This is a summary of the job listing. | 90/100 | Matches key skills and salary range.

    Output format:
    - title: Software Engineer
      company: Google
      location: Remote
      salary: $120,000 - $150,000
      link: https://jobs.google.com/software-engineer
      summary: This is a summary of the job listing.
      match: 90/100
      justification: Matches key skills and salary range.

    Text to extract from:
    {text_content}
    """
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=2048
    )

    jobs = []
    listings = response.choices[0].text.strip().split('\n- ')
    for listing in listings:
        if listing.startswith('- '):
            listing = listing[2:]
        parts = listing.split('\n')
        job_info = {}
        for part in parts:
            if ': ' in part:
                key, value = part.split(': ', 1)
                job_info[key.strip()] = value.strip()
        if len(job_info) >= 8:
            jobs.append(job_info)
    return jobs

def find_job_description_link(html_content, job_title, company):
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a')
    
    for link in links:
        href = link.get('href')
        if href and company.lower() in href and 'careers' in href:
            prompt = f"Is this link a job description for the position '{job_title}' at '{company}'?\nLink: {href}\nPlease answer yes or no."
            try:
                response = openai.Completion.create(
                    engine="davinci",
                    prompt=prompt,
                    max_tokens=50
                )
                answer = response.choices[0].text.strip().lower()
                if 'yes' in answer:
                    return href
            except openai.error.OpenAIError as e:
                print(f"Error during OpenAI API call: {e}")
                time.sleep(1)  # Wait a bit before retrying in case of rate limit
    return None
