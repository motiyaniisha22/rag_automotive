import time

# user_input = "{'make': 'ford', 'model' : 'escape', 'year': '2001', 'issue': 'stuck throttle risk'}"

while True:
    user_input = input("Paste Json here: ")
    start_time = time.time()
    response, retrieved_docs = generate_response(user_input)
    end_time = time.time()
    print("Summary :\n",response)
    print("\n\nRetrieved Documents :\n",retrieved_docs)
    print("\n\nTime taken = ", (end_time - start_time) // 60, "minutes")
    if user_input == "exit":
        break
