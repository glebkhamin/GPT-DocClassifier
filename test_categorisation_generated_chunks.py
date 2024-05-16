"""
So I think the best approach is to use the CSVs as they are a standard format.
I will load in a CSV and go through each chunk and use fine-tuned model to
assign it a topic and then write to the CSV.
"""
import csv
import glob
import json
import os, subprocess
import string
import sys
from pprint import pprint

import unicodedata
from keys import KEY
NON_VOCAB_PROMPT = True

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=KEY,
)
SEED = 42
FT_MODEL = "replace_the_value_of_FT_MODEL_with_the_model_name " 

# note, this uses the post v1 openai API
def send_prompt(prompt, model="gpt-3.5-turbo", temperature=1, seed=SEED):
    completions = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        response_format={"type": "json_object"},
        temperature=temperature,
        seed=seed,
    )
    return completions.choices[0].message.content

def build_prompt(chunk):
    max_category_count = 3
    categories_for_prompt = ""
    if NON_VOCAB_PROMPT:
        prompt = open("topic_assignment_no_vocab_prompt.txt", "r").read()
    else:
        prompt = open("topic_assignment_prompt.txt", "r").read()
    # need to replace {{max_category_count}}, {{text_chunk}} and
    # {{categories_list_of_strings}} with the correct values
    prompt = prompt.replace("{{max_topic_count}}", str(max_category_count))
    prompt = prompt.replace("{{text_chunk}}", "'" + chunk + "'")
    # this will be non-existent in no-vocab prompt
    prompt = prompt.replace("{{topics_list_of_strings}}", categories_for_prompt)
    prompt = prompt.replace("\n", " ")
    return prompt

# read in chunks from a csv
# for each chunk, use the fine-tuned model to assign a topic
# write the topic to a new csv
def assign_topics_csv(filename):
    print("Assigning topics to chunks in csv: ", filename)
    # read in the csv using csv package
    with open(filename, "r") as file:
        reader = csv.reader(file)
        # Read the header
        header = next(reader)
        # Read the remaining data
        data = [row for row in reader]
    # create data_dict from header
    """data_dict = {}
    for col in header.split(","):  
        data_dict[col] = [] """
    # create new csv
    new_filename = filename.replace(".csv", "_topics.csv")
    with open(new_filename, "w") as f:
        header += ["assigned","topics"]
        f.write(",".join(header) + "\n")
        for row in data:
            # get the chunk
            chunk = row[3]
            # get the topic
            prompt = build_prompt(chunk)
            prompt = prompt.replace("\n", " ")
            print(".", end="")
            predicted_topics = send_prompt(prompt, model=FT_MODEL)
            # pull out of json
            try:
                predicted_topics = eval(predicted_topics.strip().replace("\t", ""))
                predicted_topics = predicted_topics.get('topics', [str(predicted_topics)])
            except Exception as e:
                print("\nError with predicted_topics: ", predicted_topics)
                print("Error: ", e)
                predicted_topics = [""]
            predicted_topics = [t for t in predicted_topics]

            # write to new csv
            row_string = ""
            for c, col in enumerate(row):
                if c == 3:
                    col = '"' + col + '"'
                #if col!="":
                row_string += f"{col},"

            for topic in predicted_topics:
                topic = topic.replace("{}","")
                row_string += f"{topic},"
            row_string = row_string[:-1] + "\n"
            f.write(row_string)
    print("\nDone. Wrote to: ", new_filename)


#filename = "/doc_analysis2023/scoresheet_gpt4_gleb.csv"
filename = "scoresheet.csv"
#ilename = "/doc_analysis2023/scoresheet_recursive_325_gleb.csv"
assign_topics_csv(filename)
