with open("token.txt", "r") as token_file:
    tokens = token_file.read().strip()
HUGGINGFACEHUB_API_TOKEN = tokens.split()[0]
LANGSMITH_API_KEY = tokens.split()[1]
print(LANGSMITH_API_KEY)