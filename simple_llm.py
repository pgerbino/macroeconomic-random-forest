import os
from openai import OpenAI

# Get the API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Create a chat completion
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# Print the response
print(response.choices[0].message.content)
