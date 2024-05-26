import uvicorn
import logging

logging.basicConfig(level=logging.ERROR)
logging.captureWarnings(True)

if __name__ == '__main__':
    uvicorn.run("api.knowledge_store_api:app", host="0.0.0.0", port=8000, reload=True)
