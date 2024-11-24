import time
import api
import argparse

parser = argparse.ArgumentParser(description="Summarization")
parser.add_argument('--json_query', type=str, help="JSON string with automobile issue and car details", required=True)

args = parser.parse_args()
user_input = args.json_query

# start_time = time.time()
response, retrieved_docs = api.generate_response(user_input)
# end_time = time.time()
print("Summary :\n",response)
print("\n\nRetrieved Documents :\n",retrieved_docs)
# print("\n\nTime taken = ", (end_time - start_time) // 60, "minutes")
