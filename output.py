import yaml

def output_job_listings(job_listings, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(job_listings, file)

def filter_existing_jobs(new_jobs, existing_jobs):
    filtered_jobs = []
    for job in new_jobs:
        is_duplicate = any(
            existing_job['title'] == job['title'] and
            existing_job['company'] == job['company'] and
            existing_job['location'] == job['location'] and
            existing_job['salary'] == job['salary']
            for existing_job in existing_jobs
        )
        if not is_duplicate:
            filtered_jobs.append(job)
    return filtered_jobs
