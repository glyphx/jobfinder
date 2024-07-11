import os
from search import generate_job_search_query, search_platform, parse_job_listings_with_openai, find_job_description_link, load_resume, generate_embedding
from output import output_job_listings, filter_existing_jobs
import pinecone

pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")  # Use your Pinecone environment

index_name = 'job-search'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # dimension should match OpenAI's embedding dimension
index = pinecone.Index(index_name)

def process_jobs(job_listings, existing_jobs):
    new_jobs = []

    for job in job_listings:
        is_duplicate = any(
            existing_job['title'] == job['title'] and
            existing_job['company'] == job['company'] and
            existing_job['location'] == job['location'] and
            existing_job['salary'] == job['salary']
            for existing_job in existing_jobs
        )
        if is_duplicate:
            continue

        html_content = search_platform("Google", job['title'] + ' ' + job['company'])
        job_description_link = find_job_description_link(html_content, job['title'], job['company'])
        
        job_info = {
            'title': job['title'],
            'company': job['company'],
            'location': job.get('location', 'Remote'),
            'salary': job.get('salary', 'Not specified'),
            'link': job['link'],
            'secondary_url': job_description_link,
            'summary': job.get('summary', 'No summary available'),
            'match': job.get('match', 'Not specified'),
            'justification': job.get('justification', 'Not specified')
        }
        
        new_jobs.append(job_info)

        # Generate and store vector in Pinecone
        job_vector = generate_embedding(job_info['summary'])
        index.upsert([(str(job_info['link']), job_vector)])

    return new_jobs

def find_new_job_listings(job_title, platforms, resume_content):
    job_listings = []
    for platform in platforms:
        query = generate_job_search_query(job_title, platform)
        html_content = search_platform(platform, query)
        if html_content:
            platform_jobs = parse_job_listings_with_openai(html_content, resume_content)
            job_listings.extend(platform_jobs)
    return job_listings

def query_pinecone_with_resume(resume_content):
    resume_vector = generate_embedding(resume_content)
    result = index.query([resume_vector], top_k=5, include_values=True)
    
    return result

if __name__ == '__main__':
    resume_file = "Shawn_Ott_Resume.txt"
    resume_content = load_resume(resume_file)
    
    existing_jobs = []

    platforms = ["Indeed", "ZipRecruiter", "SimplyHired", "AngelList", "FlexJobs"]
    new_job_listings = find_new_job_listings("security engineer", platforms, resume_content)

    filtered_new_jobs = filter_existing_jobs(new_job_listings, existing_jobs)

    new_jobs = process_jobs(filtered_new_jobs, existing_jobs)

    output_job_listings(new_jobs, 'new_job_listings.yaml')

    # Query Pinecone with resume
    pinecone_results = query_pinecone_with_resume(resume_content)
    for result in pinecone_results['matches']:
        print(f"Job Link: {result['id']}, Distance: {result['score']}")
