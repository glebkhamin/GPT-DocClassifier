# analyse the combined csv file with the training data
import sys

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # load in the csv
    #with open("fine_tuning_data/fine_tuning_standardised_combined.csv", 'r') as f:
    with open("fine_tuning_data/fine_tuning_standardised_combined_train_fold_0.csv", 'r') as f:
    #with open("fine_tuning_data/fine_tuning_standardised_combined_train_fold_2_augmented.csv", 'r') as f:
    #with open("fine_tuning_data/fine_tuning_standardised_combined_train_augmented.csv", 'r') as f:
        dataset = pd.read_csv(f)

    # create column headings 'chunk' and 'topics'
    dataset.columns = ['chunk', 'topics']


def analyse_chunk_df(dataset):
    # create a new column 'chunk_length' with the length of the chunk
    dataset['chunk_length'] = dataset['chunk'].str.len()
    # create a column which counts the number of comma-seperated topics in the topics column
    dataset['num_topics'] = dataset['topics'].str.count(',') + 1

    # fine the maximum number of topics
    max_num_topics = dataset['num_topics'].max()
    print("Max number of topics: ", max_num_topics)

    # because it's 3, we add 3 new columns to the dataframe
    dataset['topic_0'] = ""
    dataset['topic_1'] = ""
    dataset['topic_2'] = ""

    # now go through each row and split the topics into the new columns
    topic_0_col = 4
    for index, row in dataset.iterrows():
        topics = row['topics'].split(',')
        #print(topics)
        for i, c in enumerate(topics):
            c = c.strip()
            dataset.iloc[index, topic_0_col + i] = c

    # now get a list of all possible topics and print them
    topics = list(dataset['topic_0'].unique())
    topics += list(dataset['topic_1'].unique())
    topics += list(dataset['topic_2'].unique())
    # get rid of repeats with set()
    topics = list(set(topics))
    # there will be an empty topic because there will be empty topic_1 and _2 columns for some rows
    topics = [c for c in topics if c != '']

    # sort alphabetically
    topics.sort()
    print("topics: ", topics)
    print("Number of topics: ", len(topics))

    print(dataset.info())
    print(dataset.describe())
    return dataset, topics


# now we can plot the number of chunks in each topic
# first we need to create a new dataframe with the counts
def topic_count(dataset, topics):
    counts = {}
    for c in topics:
        count = len(dataset[dataset['topic_0'] == c]) + \
                len(dataset[dataset['topic_1'] == c]) + \
                len(dataset[dataset['topic_2'] == c])
        counts[c] = count
    return counts

def calculate_synthetics_allowed(counts, threshold=10):
    # for every 2 real samples, we can have 1 synthetic sample
    counts_augmented = {}
    # get max count value
    max_count = max(counts.values())
    for k,v in counts.items():
        if v < threshold:
            # the allowed number of synthetics is the number of ordered pairs of real samples
            # e.g. 3 real samples = 3*2 = 6 synthetics
            # use python library for combinations
            missing = threshold - v
            groups = []
            # see how many groups of as large as possible we can make
            # to use as neighbours to synthesize
            for nn in range(threshold//2,1,-1):
                groups = list(combinations(range(v), nn))
                if len(groups) >= missing:
                    print(f"Found {len(groups)} groups of {nn} samples for {k}")
                    break
            if len(groups) >= missing:
                synthetics = len(groups)
            else:
                synthetics = 0
            #synthetics = v//2
            counts_augmented[k] = min(synthetics+counts[k], max_count)
        else:
            counts_augmented[k] = v
    return counts_augmented


def list_topics_with_more_than_n_samples(counts, n):
    topics = []
    for c in counts:
        if counts[c] >= n:
            topics.append(c)
    return topics

def get_chunks_by_topic(dataset, topics):
    # create a list of tuples - (topic, [list of chunks in that topic])
    topic_chunks = []
    for c in topics:
        chunks = list(dataset[dataset['topic_0'] == c]['chunk']) + \
                 list(dataset[dataset['topic_1'] == c]['chunk']) + \
                 list(dataset[dataset['topic_2'] == c]['chunk'])
        topic_chunks.append((c, list(set(chunks))))
    return topic_chunks

def plot_topic_counts(counts):
    data_renamed = counts.copy()
    data_renamed['ESG'] = data_renamed.pop('environmental and social and corporate governance')

    # Sorting the data by value in descending order
    sorted_data = dict(sorted(data_renamed.items(), key=lambda item: item[1], reverse=True))

    # Generating a list of non-zero odd numbers up to the maximum value
    max_value = max(sorted_data.values())
    odd_numbers = [i for i in range(1, max_value + 2, 2) if i != 0]

    # Creating the bar chart with the updated y-axis
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data.keys(), sorted_data.values())

    # Rotating the x-labels for better visibility
    plt.xticks(rotation=90)

    # Setting the y-axis to only include non-zero odd numbers
    plt.yticks(odd_numbers)

    # Adding labels and title
    plt.xlabel('Topics')
    plt.ylabel('Count')
    plt.title('Count of Topics in Discy Training Data')

    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset, topics = analyse_chunk_df(dataset)
    counts = topic_count(dataset, topics)
    #print(counts)
    # display the counts in reverse order
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    print(counts)
    # now get average length of chunks
    print("Average length of chunks: ", dataset['chunk'].str.len().mean())
    # get min max length of chunks
    print("Min length of chunks: ", dataset['chunk'].str.len().min())
    print("Max length of chunks: ", dataset['chunk'].str.len().max())
    # std of thunk lengths
    print("Std of chunk lengths: ", dataset['chunk'].str.len().std())

    # now get mean number of topics per chunk
    print("Average number of topics per chunk: ", dataset['num_topics'].mean())




# list all topics with more that 30 samples

"""
if __name__ == "__main__":
    print(list_topics_with_more_than_n_samples(counts, 30))
    print(len(list_topics_with_more_than_n_samples(counts, 13)))
    
    
    # now we can plot the counts
    plt.bar(topics, counts)
    plt.xticks(rotation=90)
    plt.show()
    
    # there are 170 Not purpose topics
    
    
    # sort dataset by topic_0
    dataset = dataset.sort_values(by=['topic_0'])
    print(dataset)
"""
