import logging
import ast
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Gradio Client
client = Client("yuntian-deng/o1mini")


class Message(BaseModel):
    text: str

def extract_latest_response(raw_response):
    try:
        # Remove the log tags and extract the content
        content = raw_response.replace('<log>\nINFO:root:Raw response from Gradio Client: ', '').replace('\n</log>', '')
        
        # Parse the content as a Python tuple
        parsed_content = ast.literal_eval(content)
        
        # Extract the first element of the tuple (the list of messages)
        messages = parsed_content[0]
        
        # Get the last message in the list
        last_message = messages[-1]
        
        # The response is the second element of the last message tuple
        response = last_message[1]
        
        return response
    except Exception as e:
        print(f"Error extracting response: {str(e)}")
        return None

@app.post("/api/chat")
async def chat(message: Message):
    try:
        logger.info(f"Received message: {message.text}")
        
        # Send the entire conversation history to the Hugging Face Space
        response = client.predict(str(message.text), api_name="/predict")
        logger.info(f"Raw response from Gradio Client: {response}")
        
        # Extract the latest response
        extracted_response = extract_latest_response(str(response))
        logger.info(f"Extracted response: {extracted_response}")
        
        return {"response": extracted_response}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_chat():
    try:
        logger.info("Resetting chat...")
        # Call the reset endpoint of the Gradio client
        reset_response = client.predict(api_name="/reset_textbox")
        logger.info(f"Raw reset response from Gradio Client: {reset_response}")

        logger.info(f"Reset response: {reset_response}")
        return {"status": "Chat has been reset.", "response": reset_response}
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset chat.")