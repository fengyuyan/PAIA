from llama_cpp import Llama
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import pymupdf
from pathlib import Path



def read_pdf_file_with_image(file_path, kb_path):
    doc = pymupdf.open(file_path)
    text = "\n"
    images = {}
    image_number = 0
    for page in doc:
        clip = page.rect
        clip.y0 = 75
        clip.y1 = 750
        for block in page.get_text("dict", clip=clip)["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"] == " ":
                            if text[-1] != "\n":
                                text += "\n"
                            else:
                                continue
                        else:
                            text += span["text"]
            if block["type"] == 1:
                name = f"images/img{page.number}-{image_number}.{block['ext']}"
                text += f" ![image]({name}) "
                images[str(image_number)] = name
                out = open(name, "wb")
                out.write(block["image"])
                out.close()
                image_number += 1
    return text, images


def split_text(base_file_name, text, chunk_size, chunk_overlap):
    text_chunks = {}
    text_length = len(text)
    start = 0
    i = 1
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        text_chunks[base_file_name + "_" + str(i)] = chunk
        start = end - chunk_overlap
        i += 1
    return text_chunks


def load_model(
        model_path="C:/Users/Patrick/MLExperiments/models/llama-3-8b-it/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        temperature=0.1,
        n_gpu_layers=0,
        verbose=False,
):
    model = Llama(
        model_path=model_path,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        n_ctx=8000,
        verbose=verbose,
    )
    return model


class DatabaseManager:
    def __init__(self, kb_path="db_data"):
        self._client = chromadb.PersistentClient(
            path=kb_path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        self._kb_path = kb_path

    def add_to_kb(self, kb, file_path):
        text, images = read_pdf_file_with_image(file_path, self._kb_path)
        text_chunks = split_text(kb, text, 1000, 200)
        chunk_names = list(text_chunks.keys())
        chunk_texts = list(text_chunks.values())
        metadata = [{'parent_file': file_path, } for i in range(len(chunk_names))]
        collection = self._client.get_or_create_collection(
            name=kb
            # metadata={
            #     'hnsw_space': 'cosine',
            # }
        )
        collection.upsert(
            ids=chunk_names,
            documents=chunk_texts,
            metadatas=metadata,
        )

    def get_kb_list(self):
        collections = self._client.list_collections()
        return [collection.name for collection in collections]

    def query_kb(self, query, kb):
        collection = self._client.get_or_create_collection(
            name=kb,
        )
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        return results

    def delete_doc_from_kb(self, kb, file_path):
        collection = self._client.get_or_create_collection(
            name=kb,
        )
        collection.delete(
            where={'parent_file': file_path}
        )

    def delete_kb(self, kb):
        self._client.delete_collection(kb)


class PrimariusAssistant:
    def __init__(self, model_path, ):
        self._model = load_model(model_path)

    def create_completion(self, prompt: str, context: str, kb: str, app_cfg=None):
        temperature = 0
        max_tokens = 1024
        if app_cfg:
            temperature = app_cfg["temperature"]
            max_tokens = app_cfg["max_tokens"]
        query_string = f'''Query: {prompt}
        Context: {context}'''
        response = self._model.create_chat_completion(
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You answer a user's question given some context. The first item labelled Query is the user's question.\
The second item labelled Context will contain relevant text. Do not respond with anything that is not the direct answer. Do not mention the provided\
text. Keep your answer clean and concise. Include any image tags."},
                {
                    "role": "user",
                    "content": query_string
                },
            ],
            stop=["<|end_of_text|>", "assistant\n"]
        )
        response_text = response['choices'][0]['message']['content']
        return response_text


def init_backend():
    global assistant, db_manager
    db_manager = DatabaseManager()
    assistant = PrimariusAssistant(
        "/root/PycharmProjects/PrimariusAIAssistant/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
    Path("./images").mkdir(parents=True, exist_ok=True)


def get_kb_list():
    return db_manager.get_kb_list()


def kbqa(question, kb, app_cfg=None):
    print('Starting inference...')
    context = db_manager.query_kb(question, kb)
    context_text = "\n".join(context['documents'][0])
    response = assistant.create_completion(question, context_text, kb, app_cfg=app_cfg)
    return response

def query_kb(question, kb):
    context = db_manager.query_kb(question, kb)
    context_list = [doc for doc in context['documents'][0]]
    context_text = "\n".join(context_list)
    return context_text


def add_doc(kb, file_path):
    db_manager.add_to_kb(kb, file_path)


def delete_doc_from_kb(kb, file_path):
    db_manager.delete_doc_from_kb(kb, file_path)


def delete_kb(kb):
    db_manager.delete_kb(kb)


assistant, db_manager = None, None
init_backend()
