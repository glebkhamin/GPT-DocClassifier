# GPT-DocClassifier

## Project Overview

This repository contains the GPT-DocClassifier, a machine learning solution designed to automate the chunking and tagging of various document types, including PDF, DOCX, XLSX, and PPTX. Built with the cutting-edge capabilities of GPT APIs, this system streamlines the organisation and categorisation of large volumes of documents, making it a valuable tool for content management and data analysis tasks.

## Features
- **Document Chunking:** Breaks down / 'chunks' various document types into meaningful and manageable text chunks for easier processing.
- **Intelligent Tagging:** Utilises a fine-tuned GPT model to intelligently tag text chunks based on content relevance.
- **Multi-format Compatibility:** Processes a range of document formats including PDF, DOCX, XLSX, and PPTX.
- **Content Interpretation:** Capable of interpreting and tagging not only text but also the contents of images and tables within documents.
- **GPT-4 Vision Integration:** Leverages GPT-4-Vision for advanced image and table understanding within documents.
- **CSV Compilation:** Aggregates all chunks and tags into a comprehensive CSV file for subsequent use.

## Getting Started / Process for Executing Code
To get started with the GPT-DocClassifier:

### Relevant Code Files
To execute the code for this project, you'll need the following Python scripts located in the main directory:

- *test_categorisation_generated_chunks.py:* Runs the categorization process on generated chunks.
- *finetune.py:* Handles the fine-tuning of the GPT model.
- *chunker.py:* Splits documents into chunks using GPT-4 Vision.
- *keys.py:* Stores API keys and other sensitive information.
- *inputdata_analysis.py:* Analyses the input data for preprocessing needs.
- *requirements.txt:* Lists all necessary Python libraries.

### Prompt Files
Ensure the following prompt files are in the same directory as the code files for proper execution:

- *section_prompt_chunking.txt* (used for CHUNKING)
- *topic_assignment_no_vocab_prompt.txt* (used for TAGGING)

### Necessary Libraries
Install the required libraries by running the following command in your terminal:

**pip install -r requirements.txt**

The *requirements.txt* file includes the following libraries:

- *openai*
- *pypdf2*
- *langchain*
- *pdf2image*
- *pandas*
- *scikit-learn*
- *iterative-stratification*
- *matplotlib*

### Additional Software
For PDF conversions, LibreOffice is required. You can install it from the following link: https://www.libreoffice.org/get-help/install-howto/

If you are using a Mac, you will need to add soffice to your PATH with the following terminal command:

**export PATH=$PATH:/Applications/LibreOffice.app/Contents/MacOS**

Please note that slight code changes may be needed for non-Mac systems.

### Data Set-Up
Organise your documents for chunking/tagging into a subdirectory named *chunking_tests*. Training CSVs should be placed in a subdirectory called fine_tuning_data.

### Fine-Tuning
- In *keys.py*, replace the placeholder with your OpenAI API key.
- Execute *finetune.py* to combine, augment data, and initiate fine-tuning with OpenAI.
- Once fine-tuning is completed, OpenAI will provide the name of the fine-tuned model. This will be needed for tagging chunks in the next steps.

### Generate Chunks
- Create a directory called chunking_tests and place your test documents there.
- Modify the chunker.py as needed to adjust the number of pages processed.
- Execute chunker.py to generate chunks and output them into a CSV called scoresheet.csv.

### Tag Chunks in CSV
- Update the FT_MODEL variable in test_categorisation_generated_chunks.py with the model name received from OpenAI.
- Run the script to tag each chunk and compile the results into scoresheet_topics.csv.
