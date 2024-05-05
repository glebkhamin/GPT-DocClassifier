#
"""
Fine tuning inputs are a list like this:
{"prompt": "This is a test", "completion": "This is a test completion"}
{"prompt": "This is a test2", "completion": "This is a test completion2"}
etc.
So for categorisation it would be:
{"prompt": <chunk as a string>, "completion": <list of categories as a string>}
{"prompt": <chunk as a string>, "completion": <list of categories as a string>}
etc.
concrete EXample:
{"prompt": "By digitizing how we engage with our employees through Amber,
    we have increased the scope and frequency of employee feedback and",
    "completion": "E-sat,Employment Engagement"}
{"prompt": "career philosophy by encouraging employees to reflect on their performance, set challenging goals, receive feedback, identify their
        development needs and find relevant learning and training opportunities.",
        "completion": "Employment Engagement"}
etc
"""
import csv
import random
import shutil
import sys

from keys import KEY
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

# import combinations from itertools
from itertools import combinations
from openai import OpenAI
import pandas as pd
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=KEY,
)

from inputdata_analysis import analyse_chunk_df, topic_count,get_chunks_by_topic
PARAGRAPHISE = False
NON_VOCAB_PROMPT = True
AUGMENT_DATA = True
USE_OVER_ONES = False
SEED = 42
GREATER_THAN_9_CATEGORIES = ['employee engagement', 'risk', 'about', 'digital', 'purpose', 'employee satisfaction', 'capital', 'commercial', 'cost', 'customer centric', 'environmental and social and corporate governance', 'growth']
USE_NOT_CATEGORIES = False
USE_GREATER_THAN_9_CATEGORIES = False
ALL_CATEGORIES = ['About', 'C-Sat', 'Capital', 'Cash', 'Commercial', 'Corporate actions', 'Cost', 'Customer centric', 'Digital', 'E-Sat', 'ESG', 'Employee engagement', 'Financial management', 'Growth', 'Not Financial management', 'Not employee-engagement', 'Not growth', 'Not operational excellence', 'Not purpose', 'Not responsible business', 'Not revenue', 'Not risk', 'Not risk management', 'Not-E-Sat', 'Not_Commercial', 'Not_Corporate actions', 'Not_about', 'Not_cost', 'Not_customer centric', 'Not_digital', 'Operational excellence', 'Profit', 'Purpose', 'Responsible business', 'Revenue', 'Risk', 'Risk management', 'Sustainability']
if USE_OVER_ONES:
    ALL_CATEGORIES = ['About', 'Capital', 'Commercial', 'Cost',
                      'Customer centric', 'Digital', 'E-Sat', 'ESG', 'Employee engagement', 'Financial management',
                      'Growth', 'Not Financial management', 'Not employee-engagement', 'Not growth',
                      'Not operational excellence', 'Not purpose', 'Not responsible business', 'Not revenue',
                      'Not risk', 'Not risk management', 'Not-E-Sat', 'Not_Commercial', 'Not_Corporate actions',
                      'Not_about', 'Not_cost', 'Not_customer centric', 'Not_digital', 'Operational excellence',
                      'Profit', 'Purpose', 'Responsible business', 'Risk', 'Risk management']

if USE_GREATER_THAN_9_CATEGORIES:
    ALL_CATEGORIES = GREATER_THAN_9_CATEGORIES
ALL_CATEGORIES = [c.lower() for c in ALL_CATEGORIES]
if not USE_NOT_CATEGORIES:
    ALL_CATEGORIES = [c for c in ALL_CATEGORIES if not c.startswith('not ') and not c.startswith('not_') and not c.startswith('not-')]

category_expansions = {'ESG': 'Environmental and social and corporate governance',
                       'E-Sat': 'Employee satisfaction',
                       'C-Sat': 'Customer satisfaction'}
category_expansions = {k.lower(): v.lower() for k, v in category_expansions.items()}

#CATEGORY_COUNTS = {'employee engagement': 19, 'risk': 16, 'about': 15, 'digital': 14, 'purpose': 13, 'employee satisfaction': 12, 'capital': 10, 'commercial': 10, 'cost': 10, 'customer centric': 10, 'environmental and social and corporate governance': 10, 'growth': 10, 'responsible business': 10, 'operational excellence': 3, 'profit': 3, 'risk management': 3, 'financial management': 2, 'cash': 1, 'corporate actions': 1, 'customer satisfaction': 1, 'revenue': 1, 'sustainability': 1}
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

def convert_discy_csvs(file, append=False):
    # this is a complex process because the files provided by the
    # business partner Discy are in a multi-line CSV format.
    # So we need to read the file line by line and convert it to a proper CSV
    # Types of lines:
    # Type A:
    # Genpact is a global professional services firm that makes business transformation real.,About
    # Type B:
    # "We believe our approach to business transformation, enabled through
    # combining our deep industry and process
    # expertise with our advanced skills in digital and analytics,
    # differentiates us from our competitors.",About
    # (i.e. has multi-line chunk with \n's in it)
    # Type C:
    # "We are also subject to regulation by regional bodies
    # such as the European Union (""EU"").",Not purpose
    # (i.e. Type B with quotes within the quotes)
    # Type D:
    # Is a Type A, B, or C but has but  contains one or more
    # non alphanumeric character 8226 (a bullet point)
    #
    # How identify the different types?
    # Type A: only a single ',' and no speech marks around the chunk
    # Type B:
    #       First line: starts with a speech mark
    #       Middle lines: have no speech marks
    #       Last line: ends with a speech mark and a comma and a category/completion
    # Type C:
    #       First line: starts with a speech mark
    #       One or more lines contain double speech marks
    #       Last line: ends with a speech mark and a comma and a category/completion
    # Type D:
    #       Is a Type A, B, or C but has but an ord 8226 (a bullet point)
    #
    # General algorithm for these sorts of things is:
    #   Have a series of flags which indicate that you have encountered
    #   a particular type (i.e. set to True when ecountered).
    #   Keep that flag true until you have processed
    #    the last line of that type.
    #   Convert that all to a normal CSV row, and then set flags to False.
    #

    multi_lines = []
    standard_csv = []
    in_type_B = False

    with open(f"fine_tuning_data/{file}", "r") as f:
        g = open(f"fine_tuning_data/fine_tuning_standardised.csv",
                 "a" if append else "w")
        if not append:
            g.write("prompt,completion\n")
        for row in f.readlines():
            try:
                row = row.replace('""', "'") # deal with Type C
                # NOTE - THAT THIS WON'T OPEN IN EXCEL
                # NEED TO USE SOMETHING LIKE ">>" INSTEAD OF "-" FOR THAT
                row = row.replace(chr(8226), "-")  # deal with Type D
                # so now we've removed all Type C and D
                if not in_type_B:
                    # Type A
                    if row.count(",") == 1 and row.count('"') == 0:
                        # Type A
                        # minimum of 2 words for a prompt
                        if len(row.split(",")[0].split()) > 2:
                            standard_csv.append((row.split(",")[0], row.split(",")[1].strip()))
                            g.write(f"{standard_csv[-1][0]},{standard_csv[-1][1]}\n")
                        #print(f"{standard_csv[-1][0]},{standard_csv[-1][1]}\n")
                    # Type B
                    elif row[0] == '"' and row.count('"') == 1:
                        in_type_B = True
                        first_line = row.replace("\n"," ") # remove \n
                        multi_lines.append(first_line)
                elif in_type_B:  # so we're in a type
                    # is it the last line of a type B?
                    c_split = row.split(",")
                    if len(c_split) > 1:
                        last_but_one_split = c_split[-2][-1]
                        last_split = c_split[-1]
                        if last_but_one_split.endswith('"') and last_split.endswith("\n"):
                            # last line of a type B
                            multi_lines.append(row.strip())
                            # now convert multi_lines to a standard CSV row
                            standard_row = multi_lines[0] #[1:-1] # remove first "
                            for line in multi_lines[1:]:
                                standard_row += line
                            #last_line = multi_lines[-1].split('"')
                            #standard_row += last_line[0]+'"'+last_line[1].strip()
                            # find all indices of commas in standard_row
                            comma_indices = [i for i, ltr in enumerate(standard_row) if ltr == ',']
                            # minimum of 2 words for a prompt
                            if len(standard_row[:comma_indices[-1]].split()) > 2:
                                standard_csv.append((standard_row[:comma_indices[-1]],
                                                     standard_row[comma_indices[-1]+1:].strip()))
                                g.write(f"{standard_csv[-1][0]},{standard_csv[-1][1]}\n")
                            in_type_B = False
                            multi_lines = []
                            continue
                    # not an end or start line but in a type B
                    multi_lines.append(row.replace("\n", " "))
            except:
                #g.write(f"Error with row: ,{row.replace(',',' ').strip()}\n")
                print(f"Error with row: ,{row.replace(',',' ').strip()}\n")
                multi_lines = []
                in_type_B = False
                continue
        g.close()



# goes through the csvs provided by discy
# gets rid of their multi-line format
# and combines them into a single csv
# with one keyword per row
def create_combined_discy_csv():
    convert_discy_csvs("accepted_suggestions - About.csv")
    convert_discy_csvs("accepted_suggestions - KPIs.csv", append=True)
    convert_discy_csvs("accepted_suggestions - Themes.csv", append=True)


def expand_category_and_lower(category):
    # if category has an expansion, return it
    return category_expansions.get(category, category).lower()

# now go through combined csv, and if there are any chunks which
# are repeated but with different categories, then combine them
# into a single row with multiple categories separated by a comma
def combine_categories():
    with open("fine_tuning_data/fine_tuning_standardised.csv", "r") as f:
        g = open("fine_tuning_data/fine_tuning_standardised_combined.csv", "w")
        csv_reader = csv.reader(f)
        next(csv_reader) # skip header
        prev_chunks = {}  # dictionary
        for row in csv_reader:
            chunk = row[0]
            category = row[1].lower()
            if category in ALL_CATEGORIES:  # mainly to exclude 'Not...'
                category = expand_category_and_lower(category)  # expand ESG, E-sat etc
                if chunk not in prev_chunks:  # i.e. not in the keys
                    prev_chunks[chunk] = [category]
                # avoid repeats of the same chunk, category pairs (which do exists in the discy files)
                elif category not in prev_chunks[chunk]:
                    prev_chunks[chunk].append(category)
        # now go through prompts and combine them
        print(prev_chunks)
        for chunk in prev_chunks: # keys
            if chunk.strip() != "": # and len(prev_chunks[chunk]) > 0: # avoid empty category lists
                g.write('"'+chunk+'","')
                category_string = ""
                for i, category in enumerate(prev_chunks[chunk]):
                    category_string += category
                    if i != len(prev_chunks[chunk]) - 1:
                        category_string += ","
                    else:
                        category_string += '"\n'
                        g.write(category_string)
                        category_string = ""
        g.close()

# now need to convert into this format
"""
{"messages": 
[
{"role": "user", "content": "What's the capital of France?"}, 
{"role": "system", "content": "Paris, as if everyone doesn't know that already."}
]
}
"""
def convert_chunk_csv_to_prompt_completion_jsonl(csv_filename, jsonl_filename):
    # load in the combined csv
    #with open("fine_tuning_data/fine_tuning_standardised_combined.csv", "r") as f:
    with open(csv_filename, "r") as f:
        #g = open("fine_tuning_data/fine_tuning_standardised.jsonl", "w")
        g = open(jsonl_filename, "w")
        csv_reader = csv.reader(f)
        next(csv_reader) # skip header
        for row in csv_reader:
            g.write('{"messages": [{"role": "user", "content": "The category or categories relevant to the following chunk <chunk>'+row[0]+'</chunk> are: "},')
            g.write('{"role": "assistant", "content": "' + ','.join(row[1:]) + '"}]}\n')
        g.close()


def convert_chunk_csv_to_large_prompt_completion_jsonl(csv_filename, jsonl_filename):
    max_category_count = 3
    # load in the combined csv
    #with open("fine_tuning_data/fine_tuning_standardised_combined.csv", "r") as f:
    with open(csv_filename, "r") as f:
        #g = open("fine_tuning_data/fine_tuning_standardised_large.jsonl", "w")
        g = open(jsonl_filename, "w")
        csv_reader = csv.reader(f)
        categories_for_prompt = ','.join([expand_category_and_lower(c) for c in ALL_CATEGORIES])
        next(csv_reader) # skip header
        for row in csv_reader:
            if NON_VOCAB_PROMPT:
                prompt = open("topic_assignment_no_vocab_prompt.txt", "r").read()
            else:
                prompt = open("topic_assignment_prompt.txt", "r").read()
            # need to replace {{max_category_count}}, {{text_chunk}} and
            # {{categories_list_of_strings}} with the correct values
            prompt_train = prompt.replace("{{max_topic_count}}", str(max_category_count))
            prompt_train = prompt_train.replace("{{text_chunk}}", "'"+row[0]+"'")
            #prompt_train = prompt_train.replace("{{categories_list_of_strings}}", ','.join(row[1:]))
            # this will be non-existent in no-vocab prompt
            prompt_train = prompt_train.replace("{{topics_list_of_strings}}", categories_for_prompt)
            #to_exclude = "Do not include any markdown like \"\'\'\'\" or \"\'\'\'json\"!"
            #prompt_train = prompt_train.replace(to_exclude, "")
            prompt_train = prompt_train.replace("\n", " ")
            # need completion to look like this {"categories": [category0, category1, ...]}
            categories = []
            cats = row[1].split(",")
            for c in cats:
                categories.append("'"+c+"'")
            #print(categories)
            completion_train = "{'topics': " + str(categories).replace('"', '') + '}'
            g.write('{"messages": [{"role": "user", "content": "' + prompt_train + '"},')
            g.write('{"role": "assistant", "content": "' + completion_train + '"}]}\n')
        g.close()

# now need to split into folds
def split_into_folds(n_splits=3):
    # load in the combined csv
    with open("fine_tuning_data/fine_tuning_standardised_combined.csv", "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        data = []
        for row in csv_reader:
            categories = row[1].split(",")
            data.append({"chunk":row[0],"topics": [c.replace("'", "") for c in categories]})
    # Extract features and labels
    X = [item["chunk"] for item in data]
    y = [item["topics"] for item in data]
    if n_splits > 1:
        # Convert labels to a binary matrix (for multi-label stratification)
        mlb = MultiLabelBinarizer()
        y_mlb = mlb.fit_transform(y)
        # Split the data
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        X_y_folds = []

        # Iterate over each fold
        for train_index, test_index in mskf.split(X, y_mlb):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            # Store the train and test sets for this fold
            X_y_folds.append(((X_train, y_train), (X_test, y_test)))
    else:
        X_y_folds = []
        X_y_folds.append((X,y))

    return X_y_folds

# now need to split into train and validate
def split_into_train_and_validate(test_size=0.2):
    # load in the combined csv
    with open("fine_tuning_data/fine_tuning_standardised_combined.csv", "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        data = []
        for row in csv_reader:
            categories = row[1].split(",")
            data.append({"chunk":row[0],"topics": [c.replace("'", "") for c in categories]})
    # Extract features and labels
    X = [item["chunk"] for item in data]
    y = [item["topics"] for item in data]
    # Convert labels to a binary matrix (for multi-label stratification)
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y)
    # Split the data
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in msss.split(X, y_mlb):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    return X_train, X_test, y_train, y_test


def write_X_y_to_csv(X,y, csv_filename):
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "topics"])
        for i in range(len(X)):
            y_s = ','.join(y[i])
            writer.writerow([X[i], y_s])

def gen_folds_and_write_to_csv_json(filename_root, n_splits=3, aug_threshold=10):
    X_y_folds = split_into_folds(n_splits)
    for i in range(len(X_y_folds)):
        if n_splits > 1:
            print("Writing fold " + str(i) + " to csv and jsonl")
            (X_train_fold, y_train_fold), (X_test_fold, y_test_fold) = X_y_folds[i]
            train_root = filename_root + "_train_fold_" + str(i)
            test_root = filename_root + "_test_fold_" + str(i)
            write_X_y_to_csv(X_train_fold, y_train_fold, train_root + ".csv")
            write_X_y_to_csv(X_test_fold, y_test_fold, test_root + ".csv")
        else:
            X,y = X_y_folds[i]
            train_root = filename_root + "_train"
            write_X_y_to_csv(X, y, train_root + ".csv")
        if AUGMENT_DATA:
            augment_chunk_csv(train_root + ".csv", aug_threshold)
        elif PARAGRAPHISE:
            convert_chunk_to_para_csv(train_root + ".csv")


        # use convert_chunk_csv_to_large_prompt_completion_jsonl() to convert train and test csvs to jsonl
        convert_chunk_csv_to_large_prompt_completion_jsonl(train_root +
                            ".csv", train_root + ".jsonl")
        if AUGMENT_DATA:
            convert_chunk_csv_to_large_prompt_completion_jsonl(train_root +
                            "_augmented.csv", train_root + "_augmented.jsonl")
        elif PARAGRAPHISE:
            convert_chunk_csv_to_large_prompt_completion_jsonl(train_root +
                            "_para.csv", train_root + "_para.jsonl")
        else:
            convert_chunk_csv_to_large_prompt_completion_jsonl(test_root +
                    ".csv", test_root + ".jsonl")



"""def openai_fine_tune_train_test():
    client.files.create(
        file=open("fine_tuning_data/fine_tuning_standardised.jsonl", 'rb'),
        purpose='fine-tune'
    )

    fine_tune_response = client.fine_tuning.create(
        training_file="fine_tuning_data/fine_tuning_standardised_train.jsonl",
        validation_file="fine_tuning_data/fine_tuning_standardised_validate.jsonl",
        model="gpt-3.5-turbo-1106",
        name="fine_tuning_standardised"
    )
    print(fine_tune_response)"""

def openai_fine_tune_train(filename):
    file_resp = client.files.create(
        file=open(filename, 'rb'),
        purpose='fine-tune'
    )
    #print(file_resp)
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=file_resp.id,
        model="gpt-3.5-turbo-1106"
    )
    #print(fine_tune_response)
    return fine_tune_response

def openai_fine_tune_train_test(train_filename, test_filename):
    train_file_resp = client.files.create(
        file=open(train_filename, 'rb'),
        purpose='fine-tune'
    )
    if test_filename != "":  # if there is a test file string
        test_file_resp = client.files.create(
            file=open(test_filename, 'rb'),
            purpose='fine-tune'
        )
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=train_file_resp.id,
            validation_file=test_file_resp.id,
            model="gpt-3.5-turbo-1106"
        )
    else:
        print("USING NO TEST FILE")
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=train_file_resp.id,
            model="gpt-3.5-turbo-1106"
        )
    return fine_tune_response


def run_fine_tune_folds(filename_root, n_splits=3):
    responses = []
    augmented = ""
    if AUGMENT_DATA:
        augmented = "_augmented"
    elif PARAGRAPHISE:
        augmented = "_para"
    for i in range(n_splits):
        train_filename = filename_root + "_train_fold_" + str(i) + augmented + ".jsonl"
        test_filename = filename_root + "_test_fold_" + str(i) + ".jsonl"
        fine_tune_response = openai_fine_tune_train_test(train_filename, test_filename)
        responses.append(fine_tune_response)
    return responses

# used the combined train file to fine tune
def run_fine_tune_all(filename_root):
    augmented = ""
    if AUGMENT_DATA:
        augmented = "_augmented"
    elif PARAGRAPHISE:
        augmented = "_para"
    # create a new file which is a combination of one train fold with its test fold
    all_train_filename = filename_root + "_train" + augmented + ".jsonl"
    """with open(all_train_filename, "w") as f:
        i = 0 # just use the first fold
        train_filename = filename_root + "_train_fold_" + str(i) + augmented + ".jsonl"
        test_filename = filename_root + "_test_fold_" + str(i) + ".jsonl"
        # read in train file
        with open(train_filename, "r") as f_train:
            for line in f_train:
                f.write(line)
        # read in test file
        with open(test_filename, "r") as f_test:
            for line in f_test:
                f.write(line)"""
    fine_tune_response = openai_fine_tune_train_test(all_train_filename, "")
    return fine_tune_response

def check_fine_tune_status(id):
    fine_tune_response = client.fine_tuning.jobs.retrieve(id)
    print(fine_tune_response)

def convert_sentence_to_para(sentence, categories, relevant_to_categories=False):
    # use gpt-4 to convert a sentence into a paragraph of approximately 300 letters
    # and write it to a new csv file
    prompt = "The below sentence X is about these topics: " + categories + "\n "
    prompt += "SENTENCE X: "+sentence + "\n"
    prompt += "Create a meaningful paragraph which contains sentence X.\n"
    if relevant_to_categories==True:
        prompt += "The paragraph as a whole should be about the following topics but not mention them explicitly: " + categories + "\n"
    else:
        prompt += "The elements added to X to create the paragraph do not have to be about the topics : " + categories + "\n"
    result = send_prompt(prompt,model="gpt-4-1106-preview")

def convert_sentence_csv_to_para_json(filename):
    # for each chunk in the csv use gpt-4 to convert it into a paragraph of approximately 300 letters
    # and write it to a new csv file
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        new_filename = filename[:-4] + "_para.csv"
        with open(new_filename, "w") as f2:
            writer = csv.writer(f2)
            writer.writerow(["chunk", "topics"])
            for row in reader:
                chunk = row[0]
                categories = row[1]
                para = convert_sentence_to_para(chunk)
                writer.writerow([para, categories])
    return new_filename

def order_by_minimal_intersection(synthetics):
    ss = []
    synthetics_set = [set(s) for s in synthetics]
    for j in range(len(synthetics_set)):
        to_add = [(i, len(synthetics_set[j].intersection(ssi))) for i, ssi in enumerate(synthetics_set[j:])]
        #print(f"to_add: {to_add}")
        #ss0 = [(i,len(ss[0].intersection(ssi))) for i,ssi in enumerate(ss)]
        ss.append(to_add)

    #print(ss)
    synthetics_new = [synthetics[0]]
    for i in range(1, len(synthetics)):
        # for synthetics element i, use ss to find the element with the least intersection with element i
        min_intersection = min(ss[i], key=lambda x: x[1])
        synthetics_new.append(synthetics[min_intersection[0]])
    return synthetics_new

def get_chunks_to_combine(chunks, threshold):
    missing = threshold - len(chunks)
    groups = []
    synthetics = []
    # see how many groups of as large as possible we can make
    # to use as neighbours to synthesize
    for nearest_neighbours in range(threshold // 2, 1, -1):
        groups = list(combinations(range(len(chunks)),
                                   nearest_neighbours))
        # shuffle groups to avoid minimal diversity (e.g. (0,1), (0,2), etc
        if len(groups) >= missing:
            # print(f"Found {len(groups)} groups of {nn} samples for {k}")
            break

    chunks_to_combine = []
    if len(groups) >= missing:
        random.seed(SEED)
        random.shuffle(groups)
        synthetics = list(groups)
        #print(synthetics)
        chunks_to_combine = []

        for i in range(missing):
            sub_chunks = []
            for j in range(len(synthetics[i])):
                sub_chunks.append(chunks[synthetics[i][j]])
            chunks_to_combine.append(sub_chunks)
    return chunks_to_combine


def create_paragraph(chunk, category):
    base_prompt = "The below chunk of text is about this topic: " +\
                  category[0] + "\n"
    prompt = base_prompt + chunk + "\n"
    prompt += f"\nGenerate a new chunk of at least 300 characters long that includes this chunk, and is on the same topic: {category[0]}.\n"
    prompt += "Your response should be in JSON format: {'new chunk': <insert new chunk here>}\n"
    prompt += f"IMPORTANT: the new chunk should be about the same topic as the chunks above ({category[0]}), but be approximately 300 characters long.\n"
    result = send_prompt(prompt, model="gpt-4-1106-preview")
    try:
        result = eval(result)
        result = result.get("new chunk", result[list(result.keys())[0]])
    except:
        print(f"error in generating chunk 300 from {result}")
    return result


def augment_category(chunks, category, threshold):
    # get chunks that will be combined using GPT 4
    chunks_to_combine = get_chunks_to_combine(chunks, threshold)
    base_prompt = "The below chunks of text are about the topic '" +\
                  category[0] + "':\n"
    prompts = []
    for i, chunks in enumerate(chunks_to_combine):
        max_length = max([len(chunk) for chunk in chunks])
        prompt = base_prompt
        for j,chunk in enumerate(chunks):
            prompt += f"Chunk {j+1}: {chunk} \n"

        prompt += f"\nGenerate a new chunk which is a textual interpolation of the above chunks. The new chunk should be a maximum of {max_length} characters long, and should be about the same topic.\n"
        prompt += f"It should not contain the word '{category[0]}'.\n"
        #prompt += "But make the chunk very different from the chunks above, not just a rewording of them.\n"
        prompt += "Your response should be in JSON format: {'new chunk': <insert new chunk here>}\n"
        prompt += f"IMPORTANT: the new chunk should be different to all of the chunks above, but be about the same topic ('{category[0]}').\n"
        prompts.append(prompt)
    results = []
    for seed_i, prompt in enumerate(prompts):
        # need to change seed to ensure different results
        #print(prompt)
        result = send_prompt(prompt, model="gpt-4-1106-preview",
                             seed=seed_i+1)
        #print(result)
        #input("Press enter to continue")

        """result = send_prompt(prompt, model="gpt-3.5-turbo-1106",
                             temperature=1.5, seed=seed_i + 1)"""
        try:
            result = eval(result)
            results.append(result.get("new chunk", result[list(result.keys())[0]]))
        except:
            print(f"error in augmentation result {result}")
    return results


def augment_chunk_csv(filename, threshold=10):
    dataset = pd.read_csv(filename)
    dataset.columns = ["chunk", "topics"]
    dataset, categories = analyse_chunk_df(dataset)
    category_chunks = get_chunks_by_topic(dataset, categories)
    # copy csv to a csv with augmented in its name
    new_filename = filename[:-4] + "_augmented.csv"
    shutil.copyfile(filename, new_filename)
    print("Augmenting chunks...")
    lower_thresh = 4
    print(f"AUG thresholds {lower_thresh} to {threshold}")
    with open(new_filename, "a") as f:
        for c in category_chunks:
            print(".", end="")
            # c[0] is the category name
            # c[1] is the list of chunks
            # 5 examples needed to generate up to 10
            #if len(c[1]) >= 4 and len(c[1]) < threshold: # and c[0] == "responsible business":

            if len(c[1]) >= lower_thresh and len(c[1]) < threshold:
                #print("augmenting category: ", c)
                new_chunks = augment_category(c[1], c, threshold)
                print(f"Augmenting {c[0]} with {len(new_chunks)} new chunks")
                #new_chunks = [str(i) + c[1][0] for i in range(threshold-len(c[1]))]
                for chunk in new_chunks:
                    f.write(f'"{chunk}","{c[0]}"\n')
                pass
    print("#")


def convert_chunk_to_para_csv(filename):
    dataset = pd.read_csv(filename)
    dataset.columns = ["chunk", "topics"]
    # copy csv to a csv with augmented in its name
    new_filename = filename[:-4] + "_para.csv"
    shutil.copyfile(filename, new_filename)
    print("Paragraphing chunks...")
    for chunk, topics in zip(dataset["chunk"], dataset["topics"]):
        with open(new_filename, "a") as f:
            new_chunk = create_paragraph(chunk, topics)
            #print(chunk, topics)
            #print(new_chunk)
            #input("Press enter to continue...")
            f.write(f'"{new_chunk}","{topics}"\n')

        print(".", end="")
    print("#")

def filter_out_less_than_9_categories(input_filename, output_filename):
    dataset = pd.read_csv(input_filename)
    dataset.columns = ["chunk", "topics"]
    dataset, categories = analyse_chunk_df(dataset)
    category_chunks = get_chunks_by_topic(dataset, categories)
    new_filename = output_filename
    shutil.copyfile(input_filename, new_filename)
    with open(new_filename, "a") as f:
        for c in category_chunks:
            if len(c[1]) > 9:
                for chunk in c[1]:
                    f.write(f'"{chunk}","{c[0]}"\n')


# run pipeline to convert the 3 discy files into a jsonl file
# which can be used to fine tune the model
def run_pipeline():
    create_combined_discy_csv()
    combine_categories()
    gen_folds_and_write_to_csv_json(
        "fine_tuning_data/fine_tuning_standardised_combined",
        n_splits=1, aug_threshold=10)
    # aug_threshold is the intensity of the textSMOTE augmentation
    # 10 was used in dissertation
    # e.g. threshold of 10 gives 30 extra training examples, aug_threshold of 15 gives 49 extra, etc

    #X_train, X_validate, y_train, y_validate = split_into_train_and_validate()
    #write_X_y_to_csv(X_train, y_train, "fine_tuning_data/fine_tuning_standardised_combined.csv")
    #write_X_y_to_csv(X_validate, y_validate, "fine_tuning_data/fine_tuning_standardised_combined_validation_only.csv")
    #return


    #

    #write_X_y_to_csv(X_train, y_train, "fine_tuning_data/fine_tuning_standardised_train.csv")
    """convert_chunk_csv_to_large_prompt_completion_jsonl("fine_tuning_data/fine_tuning_standardised_train.csv",
                                                       "fine_tuning_data/fine_tuning_standardised_large_train.jsonl")

    write_X_y_to_csv(X_test, y_test, "fine_tuning_data/fine_tuning_standardised_test.csv")
    convert_chunk_csv_to_large_prompt_completion_jsonl("fine_tuning_data/fine_tuning_standardised_test.csv",
                                                       "fine_tuning_data/fine_tuning_standardised_large_test.jsonl")"""
    #openai_fine_tune()
    #return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    run_pipeline()  # prepare the data
    # train on the prepared data
    resps = run_fine_tune_all("fine_tuning_data/fine_tuning_standardised_combined")
    print(resps)

    sys.exit()

    l = client.fine_tuning.jobs.list()
    for c in l:
        print(c.fine_tuned_model, c.created_at, c.status, c.trained_tokens, c.result_files)
    # get file 'file-wcgpLIcveOf85omhxpEjmUC1' from openai fine-tuning api
    """f = client.files.retrieve_content('file-wcgpLIcveOf85omhxpEjmUC1')
    with open("file-wcgpLIcveOf85omhxpEjmUC1.csv","w") as g:
        g.write(f)"""
    #print(f)

    sys.exit()

    run_pipeline()
    #augment_chunk_csv("fine_tuning_data/fine_tuning_standardised_combined.csv")
    sys.exit()

    l = client.fine_tuning.jobs.list()
    for c in l:
        print(c.fine_tuned_model, c.created_at, c.status, c.trained_tokens)
    sys.exit()

    resps = run_fine_tune_folds("fine_tuning_data/fine_tuning_standardised_combined")
    print(resps)
    sys.exit()





    resps = run_fine_tune_all("fine_tuning_data/fine_tuning_standardised_combined")
    print(resps)
    sys.exit()




    l = client.fine_tuning.jobs.list()
    for c in l:
        print(c.fine_tuned_model, c.created_at, c.status, c.trained_tokens)

    sys.exit()




    #convert_chunk_csv_to_prompt_completion_jsonl()
    #openai_fine_tune_train()
    #convert_chunk_csv_to_large_prompt_completion_jsonl()


    print(len(X_test))
    print(len(X_train))

    # Create the DataFrame
    df_test = pd.DataFrame({
        'chunk': X_test,
        'topics': [",".join(yt) for yt in y_test]
    })
    df_train = pd.DataFrame({
        'chunk': X_train,
        'topics': [",".join(yt) for yt in y_train]
    })

    df_test2 = analyse_chunk_df(df_train)

    single_samples = ['C-Sat', 'Cash', 'Corporate actions', 'Revenue', 'Sustainability']
    # check if they are all in df_test
    for s in single_samples:
        print(s in df_test['topics'].values)

    # get all categories in df_test
    all_categories_test = []
    for c in df_test['topics'].values:
        all_categories_test.extend(c.split(","))
    all_categories_test = list(set(all_categories_test))

    all_categories_train = []
    for c in df_train['topics'].values:
        all_categories_train.extend(c.split(","))
    all_categories_train = list(set(all_categories_train))

    # print all categories in train but not in test
    for c in all_categories_train:
        if c not in all_categories_test:
            print(c)
    """
    Corporate actions 1
    Revenue 1
    Not Financial management 1
    Sustainability 1
    C-Sat 1
    Cash 1
    Not operational excellence 1"""
    print("----")
    # print all categories in test but not in train
    for c in all_categories_test:
        if c not in all_categories_train:
            print(c)

    print(set(all_categories_train)-set(all_categories_test))