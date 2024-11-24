import api
import time

# while True:
#     user_input = input("Paste Json here: ")
#     if user_input == "exit":
#         break
user_input = "{ 'make': 'ford', 'model': 'escape', 'year': '2001', 'issue': 'stuck throttle risk'}"

start_time = time.time()
response = api.generate_response(user_input)
end_time = time.time()
print(response)
print("Time taken = ", end_time-start_time)

