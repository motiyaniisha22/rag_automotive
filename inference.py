import api

# while True:
#     user_input = input("Paste Json here: ")
#     if user_input == "exit":
#         break
user_input = """{ ‘make’: ‘ford’, ‘model’: ‘escape, ‘year’: ‘2001’, ‘issue’: ‘stuck throttle risk’}"""
response = api.generate_response(user_input)
print(response)

