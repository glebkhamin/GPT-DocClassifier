# chunking process. One approach
# Convert each document to a pdf
# convert the pdf pages to images
# send each image to the GPT-4 vision model
# with a query for a series of chunks to be returned in JSON format
import glob
import json
import os, subprocess
import string
import sys
from pprint import pprint

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import base64
import unicodedata
from keys import KEY

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=KEY,
)

class DocVision:
    def __init__(self, doc_path):
        #self.categorisation_prompt_filename = "category_assignment_prompt.txt"
        self.chunking_prompt_filename = "section_prompt_chunking.txt"
        self.min_chunk_characters = 80 # this is the average length of a sentence/ 20 is too short
        self.max_chunk_characters = 450
        self.average_chunk_characters = 150
        self.doc_path = doc_path
        print("EXCLUDING XLSX")
        self.allowed_extensions = ['pdf', 'pptx', 'docx']
        self.get_docs_from_path()
        #self.categories = ['Customer Experience','Financial Planning','Marketing','Operations','Technology']

    def get_docs_from_path(self):
        # get all docs filenames and paths within the tree under and in doc_path
        self.docs = []
        for root, dirs, files in os.walk(self.doc_path):
            for file in files:
                if file.split('.')[-1] in self.allowed_extensions:
                    self.docs.append(os.path.join(root, file))
        return self.docs

    # need to install libreoffice at command line (using for example homebrew)
    def convert_msoffice_to_pdf(self, file):
        # convert a xlsx, docx or pptx to pdf
        # libreoffice is called here using soffice
        # need to add soffice to path
        # export PATH=$PATH:/Applications/LibreOffice.app/Contents/MacOS
        command = f'soffice --headless --convert-to pdf "{file}" --outdir "{os.path.dirname(file)}"'
        subprocess.run(command, shell=True)
        file_extension = file.split('.')[-1]
        pdf_path = file.replace(file_extension, 'pdf')
        return pdf_path

    def convert_docs_to_pdf(self, up_to: int = None):
        """convert all docs to pdf using self.convert_msoffice_to_pdf
        write the pdfs to a subfolder of doc_path called pdfs"""
        pdfs_path = os.path.join(self.doc_path, 'pdfs')
        if not os.path.exists(pdfs_path):
            os.mkdir(pdfs_path)
        if not up_to:
            up_to = len(self.docs)
        for doc in self.docs[:up_to]:
            print(f"*Converting {os.path.basename(doc)} to pdf")
            if not doc.endswith('.pdf'):
                pdf_path = self.convert_msoffice_to_pdf(doc)
                # now move it to the pdfs folder
                os.rename(pdf_path, os.path.join(pdfs_path, os.path.basename(pdf_path)))
            else:
                # just copy pdf to pdfs folder, but don't use os.rename
                # because that will move the file, not copy it
                os.system(f'cp "{doc}" "{os.path.join(pdfs_path, os.path.basename(doc))}"')
        # need to install pdf2image via pip
        # need to install poppler-utils at command line (using for example homebrew)
        return pdfs_path

    def convert_pdf_to_images(self, pdf_path, images_path, pages=None):
        # convert a pdf to images
        # create a subfolder of doc_path called images if it doesn't exist
        if not os.path.exists(images_path):
            os.mkdir(images_path)
        # create a subfolder of images names after the pdf if it doesn't exist
        pdf_name = os.path.basename(pdf_path).split('.')[0]
        images_path = os.path.join(images_path, pdf_name)
        if not os.path.exists(images_path):
            os.mkdir(images_path)
        images = convert_from_path(pdf_path, grayscale=True,
                                   first_page=1, last_page=pages)
        image_paths = []
        for i, image in enumerate(images):
            fname = f'{pdf_path}_{i}.png'
            image_paths.append(fname)
            # save to images_path subfolder and then within that, the pdf name subfolder
            image.save(fname, "PNG")
            # move image to images_path subfolder and then within that, the pdf name subfolder
            #os.rename(os.path.join(pdf_path,fname), os.path.join(images_path,
            #                                            os.path.basename(fname)))
            os.rename(fname, os.path.join(images_path,
                                                os.path.basename(fname)))
        return image_paths

    def convert_pdfs_path_to_images(self, pdfs_path: str, images_path: str = "images",
                                    pages: int = 15, up_to: int = None):
        # convert all pdfs in a folder to images
        if not up_to:
            up_to = len(os.listdir(pdfs_path))
        for pdf in os.listdir(pdfs_path)[:up_to]:
            if pdf.endswith('.pdf'):
                print(f"*Converting {pdf} to images")
                self.convert_pdf_to_images(os.path.join(pdfs_path, pdf),
                                           os.path.join(pdfs_path, images_path),
                                           pages=pages)

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # note, this uses the post v1 openai API
    def send_prompt_vision(self, image, prompt, model="gpt-4-vision-preview"):
        base64_image = self.encode_image(image)  # required for gpt-4-vision-preview
        completions = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                          "type": "text",
                          "text": prompt
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                          }
                        }
                      ],
                }
            ],
            model=model,
            seed=42,
            max_tokens=1024
        )
        return completions.choices[0].message.content

    def categorise_chunk(self, chunk):
        print("NO!")
        return
        with open(self.categorisation_prompt_filename, "r") as f:
            prompt = f.read()
        prompt = prompt.replace("{{max_num_classifications}}", str(len(self.categories)))
        prompt = prompt.replace("{{categories_list_of_strings}}", ", ".join(self.categories))
        prompt = prompt.replace("{{text_chunk}}", chunk)
        #max_characters
        result = self.send_prompt(prompt)
        return result

    def chunk_page_image(self, image_path):
        with open(self.chunking_prompt_filename, "r") as f:
            prompt = f.read()
        prompt = prompt.replace("{{min_characters}}", str(self.min_chunk_characters))
        prompt = prompt.replace("{{max_characters}}", str(self.max_chunk_characters))
        prompt = prompt.replace("{{average_characters}}", str(self.average_chunk_characters))
        result = self.send_prompt_vision(image_path, prompt)
        return result

    def get_chunks_for_all_docs(self, images_path, up_to_docs=None, up_to_images=None, use_recursive_splitter=False, recursive_pages=15):
        if not use_recursive_splitter:
            # go through all subdirs of images_path and chunk each image
            # then save the chunks to a json file in the same subfolder
            if not up_to_docs:
                up_to = len(os.listdir(images_path))
            # one directory per doc, each containing images
            for pdf in os.listdir(images_path)[:up_to_docs]:
                # pdf is a directory name
                if not pdf.startswith('.'):
                    print(f"*Chunking {pdf}: ", end="")
                    pdf_path = os.path.join(images_path, pdf)
                    # re-create the chunks.json file in this dir
                    with open(os.path.join(pdf_path, 'chunks.jsonl'), "w") as f:
                        f.write("")
                    images_list = os.listdir(pdf_path)
                    images_list = [i for i in images_list if i.endswith('.png')]    # only pngs
                    if not up_to_images:
                        up_to_images = len(images_list)
                    # sort based on the page number
                    #print(images_list)
                    images_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

                    for p, image in enumerate(images_list[:up_to_images]):
                        if image.endswith('.png'):
                            image_path = os.path.join(pdf_path, image)
                            chunks = self.chunk_page_image(image_path)
                            print(".", end="")
                            try:
                                # add a page number at the top of the json
                                chunks = {"page": p+1, "content": eval(chunks)}
                                with open(os.path.join(pdf_path, 'chunks.jsonl'), "a") as f:
                                    json.dump(chunks, f)
                                    f.write("\n")
                            except Exception as e:
                                print(e)
                                print(f"Failed to save chunks for {image_path}, they were: {chunks}")
                    print("")
        else:
            # note images_path in this case points to the pdf directory not images
            # go through all pdfs in images_path
            # and open each, then convert page to text
            # then split the text into chunks using recursive splitter
            # then save the chunks to a json file in the same subfolder
            pdf_files = os.path.join(images_path, "*.pdf")
            pdf_files = glob.glob(pdf_files)
            pdf_files = [p for p in pdf_files if p.endswith('.pdf')]
            # sort them alphabetically
            pdf_files.sort()
            if not up_to_docs:
                up_to = len(pdf_files)
            with open(os.path.join(images_path, 'chunks_recursive.jsonl'), "w") as f:
                f.write("")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_characters,
                                                           chunk_overlap=0)
            # one directory per doc, each containing images
            for pdf in pdf_files[:up_to_docs]:
                # pdf is a filename
                print(f"*Chunking {pdf}: ", end="")
                # load in pdf and loop through pages
                # converting to text
                # then chunking
                # then saving to json
                reader = PdfReader(pdf)
                for p, page in enumerate(reader.pages[:recursive_pages]):
                    page_text = page.extract_text()
                    # Split the document into chunks using RecursiveCharacterTextSplitter
                    chunks = text_splitter.split_text(page_text)
                    chunk_json = {"chunks": []}
                    # want format {"chunks": [{"chunk_index": 0, "chunk_text": "01 - Doc
                    # to be compatible with gpt4 return
                    for i, chunk in enumerate(chunks):
                        chunk_json["chunks"].append({"chunk_index": i, "chunk_text": chunk})
                    chunks = {"doc": os.path.basename(pdf), "page": p + 1, "content": chunk_json}
                    with open(os.path.join(images_path, 'chunks_recursive.jsonl'), "a") as f:
                        json.dump(chunks, f)
                        f.write("\n")

    def get_chunks_from_jsonl(self, page):
        try:
            page = page.strip()
            # the prompt doesn't always deal with leaving out the below mark up
            page = page.replace("```json", "").replace("```", "")
            # deal with things such as /u2019
            page = unicodedata.normalize('NFKD', page).encode('ascii', 'ignore').decode('ascii')
            page = eval(page.strip())
            page_num = page['page']
            chunks = page['content']['chunks']
        except:
            print(f"Failed to eval chunk: {page}")
            chunks = None
            page_num = -1
        if page.get('doc', None): # recursive chunk includes filename
            return chunks, page_num, page['doc']
        return chunks, page_num

    def convert_all_jsonls_to_csv_scoresheet(self,image_path):
        # go through all subdirs of images_path and read in the
        # chunks.jsonl file in each, then write it to csv
        # one csv for all jsonls
        with open("scoresheet.csv", "w") as f:
            f.write('Doc,Page,"Chunk Index",Chunk Text,"Completeness Score (1-10)","Sense Score (1-10)","Page Coverage Score (1-10) (one / page only)"\n')
            list_of_dirs = os.listdir(image_path)
            list_of_dirs.sort()
            for pdf in list_of_dirs:
                if not pdf.startswith('.'):
                    pdf_path = os.path.join(image_path, pdf)
                    # read in the jsonl file
                    if not os.path.exists(os.path.join(pdf_path, 'chunks.jsonl')):
                        print(f"Skipping {pdf_path}")
                        continue
                    with open(os.path.join(pdf_path, 'chunks.jsonl'), "r") as f2:
                        # read in jsonl text
                        pages = f2.readlines()
                    # write it to csv
                    for page in pages:
                        chunks, page_num = self.get_chunks_from_jsonl(page)
                        if not chunks:
                            continue
                        for chunk in chunks:
                            f.write(f'{pdf},{page_num},{chunk["chunk_index"]},"{chunk["chunk_text"]}",,,,\n')

    def convert_all_recursive_jsonls_to_csv_scoresheet(self,pdf_path):
        # go through all subdirs of images_path and read in the
        # chunks.jsonl file in each, then write it to csv
        # one csv for all jsonls
        with open("scoresheet.csv", "w") as f:
            f.write('Doc,Page,"Chunk Index",Chunk Text,"Completeness Score (1-10)","Sense Score (1-10)","Page Coverage Score (1-10) (one / page only)"\n')
            # read in the jsonl file
            filename = os.path.join(pdf_path, 'chunks_recursive.jsonl')
            if not os.path.exists(filename):
                print(f"Skipping {filename}")
                return None
            with open(filename, "r") as f2:
                # read in jsonl text
                pages = f2.readlines()
            # write it to csv
            for page in pages:
                chunks, page_num, doc_fn = self.get_chunks_from_jsonl(page)
                if not chunks:
                    continue
                for chunk in chunks:
                    # remove non-printable characters etc
                    chunk_printable = ''.join([c for c in chunk["chunk_text"] if c in string.printable])
                    chunk_printable = chunk_printable.replace("\n", "  ")
                    f.write(f'{doc_fn},{page_num},{chunk["chunk_index"]},"{chunk_printable}",,,,\n')


print("ONLY DOING FIRST 15 PAGES")
dv = DocVision("chunking_tests")
dv.convert_docs_to_pdf()
dv.convert_pdfs_path_to_images("chunking_tests/pdfs", pages=15)

DO_RECURSIVE = False
DO_GPT4 = True
if DO_RECURSIVE:
    images_path = "doc_analysis2023/chunking_tests/pdfs"
    dv.max_chunk_characters = 325  # this gives about 545 chunks, similar to the 550 of gpt4
    dv.get_chunks_for_all_docs(images_path, use_recursive_splitter=True, recursive_pages=15) #up_to_docs=5,  up_to_images=5
    dv.convert_all_recursive_jsonls_to_csv_scoresheet(images_path)

if DO_GPT4:
    images_path = "chunking_tests/pdfs/images/"
    dv.get_chunks_for_all_docs(images_path, use_recursive_splitter=False)
    dv.convert_all_jsonls_to_csv_scoresheet(images_path)
