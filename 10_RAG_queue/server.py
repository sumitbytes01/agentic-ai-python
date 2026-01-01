from fastapi import FastAPI, Query
from client.rq_client import queue


app = FastAPI()

@app.get("/")
def root():
    return {"Status": "Server is up and running !!!"}

@app.post("/chat_query/")
def process_query(
    query: str = Query(..., description="User query string")):
    job = queue.enqueue(process_query, query)
    return {"job_id": job.id, "status": "queued"}

@app.get("/job_result/}")
def get_job_result(job_id: str = Query(..., description="Job ID to fetch result for")):
    job = queue.fetch_job(job_id)
    if job is None:
        return {"error": "Job not found"}
    if job.is_finished:
        return {"result": job.result}
    else:
        return {"status": job.get_status()}
