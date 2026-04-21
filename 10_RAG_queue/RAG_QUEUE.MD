Async coding with RAG
    It takes a lot of time to get embeddings done, etc and finally have the data in the vector database. Async would be better  

Queue in System Design
    Push the requests into the queue
    valkey (Alternative for Redis) - OPEN SOURCE - Valkey is an open-source, in-memory key–value data store—very similar in purpose and design to Redis.RAM (memory) instead of on disk,
        commonly used for:
            Caching (speeding up apps)
            Session storage
            Real-time analytics
            Message queues

RQ - REDIS QUEUE - python library for queuing jobs
    RQ has 3 main components:
        Queue
            Holds jobs waiting to be processed
        Worker
            A separate process that pulls jobs and executes them
        Redis / Valkey
            Stores the queue and job data
    When to use RQ
    Use RQ if:
        You're using Python
        You want a simple background job system
        You already use Redis/Valkey
        You don’t need complex distributed workflows
    Limitations
        Not as scalable as Celery
        Fewer advanced features (workflows, chaining, etc.)
        Single-language focus (Python)
pip install rq
    rq worker

    
    
    create a docker compose file including valkey and vectorDB - qdrant


    Symantic Cache

You can run multiple workers parallely
    and the queue is just one. 
    each worker takes a job from the queue and processes it asynchronously.