# fastapi dev example.py
# pip install "fastapi[standard]"
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "world!!"}
