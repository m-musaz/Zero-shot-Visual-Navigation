# from ollama import chat
# from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

from ollama import chat

class ResponseWrapper:
    """A simple wrapper to mimic a response object with a .content attribute."""
    def __init__(self, content):
        self.content = content

class LLM:
    def invoke(self, prompt: str, model: str = 'llama3.2') -> ResponseWrapper:
        # Send the prompt to the model and retrieve the response
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        # Return a wrapped response with a .content attribute
        return ResponseWrapper(response.message.content)

# Example usage
llm = LLM()
prompt = "What happened in world war 2?"
response = llm.invoke(prompt)
print("Model response:", response.content)

