# from groq import Groq 
# import os
# from dotenv import load_dotenv
# # Load environment variables from .env
# load_dotenv()

# # Pass the API key to Groq
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# completion = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain why fast inference is critical for reasoning models"
#         }
#     ]
# )
# print(completion.choices[0].message.content)




from openai import OpenAI
import os
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    input="Explain the importance of fast language models",
    model="openai/gpt-oss-20b",
)
print(response.output_text)
