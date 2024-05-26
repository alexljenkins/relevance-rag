from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PyMuPDFReader

def get_web_loader():
    # https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo/
    return SimpleWebPageReader(html_to_text=True)

def get_file_loader(folderpath: str):
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?h=simpledirectoryreader
    return SimpleDirectoryReader(input_dir=folderpath)

def get_pdf_loader():
    # since generic file loader failed to load 1 of the pdfs
    return PyMuPDFReader()
