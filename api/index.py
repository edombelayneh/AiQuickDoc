from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import boto3
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
import uuid
from flask_cors import CORS
from urllib.parse import urlparse, unquote
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO
import base64
import re

load_dotenv()

app = Flask(__name__)
CORS(app)
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

genai.configure(api_key=os.environ["Gemini_API_Key"])
model = genai.GenerativeModel("gemini-1.5-flash")

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
index_name = "summeryai"
dimension = 1024
metric = 'cosine'

# Check if the index exists before creating it
existing_indexes = pc.list_indexes()
if index_name not in existing_indexes:
    try:
        pc.create_index(
            name=index_name, dimension=dimension, metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the index: {e}")
else:
    print(f"Index '{index_name}' already exists.")

# Initialize the index
try:
    index = pc.Index(index_name)
    print(f"Index '{index_name}' initialized successfully.")
except Exception as e:
    print(f"An error occurred while initializing the index: {e}")
    raise


def extract_text_from_pdf(pdf_bytes, chunk_size=1000):
    text_chunks = []
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            text_chunks.extend(chunks)
    return text_chunks

def remove_duplicate_content(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(dict.fromkeys(sentences))

@app.route('/embed_text', methods=['POST'])
def embed_text():
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        upload_session_id = str(uuid.uuid4())
        embedding = embedding_model.encode(text).tolist()
        processed_data = [{
            "values": embedding,
            "id": f"{upload_session_id}_text_chunk",
            "metadata": {
                "upload_session_id": upload_session_id,
                "original_text": text
            }
        }]
        index.upsert(vectors=processed_data, namespace="ns1")
        return jsonify({
            "message": "Text embedded and stored successfully",
            "upload_session_id": upload_session_id,
            "summaries": []
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error embedding text: {str(e)}"}), 500

@app.route('/flashcards', methods=['POST'])
def flashcards():
    text_content = request.form.get('text_content')
    file_url = request.form.get('file_url')
    
    if not file_url and not text_content:
        return jsonify({"error": "No file URL or text provided"}), 400

    try:
        if file_url:
            parsed_url = urlparse(file_url)
            bucket_name = parsed_url.netloc.split('.')[0]
            object_key = unquote(parsed_url.path.lstrip('/'))
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            pdf_bytes = response['Body'].read()
            text_chunks = extract_text_from_pdf(pdf_bytes)
            extracted_text = ' '.join(text_chunks)
        else:
            extracted_text = text_content
        
        return jsonify({"extracted_text": extracted_text}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during text extraction: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    file_url = request.form.get('file_url')
    if not file_url:
        return jsonify({"error": "No file URL provided"}), 400

    try:
        parsed_url = urlparse(file_url)
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = unquote(parsed_url.path.lstrip('/'))
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        pdf_bytes = response['Body'].read()
        
        chunks = extract_text_from_pdf(pdf_bytes)
        upload_session_id = str(uuid.uuid4())
        
        processed_data = [
            {
                "values": embedding_model.encode(chunk).tolist(),
                "id": f"{upload_session_id}_{os.path.basename(object_key)}_chunk_{i}",
                "metadata": {
                    "file_name": os.path.basename(object_key),
                    "chunk_number": i,
                    "upload_session_id": upload_session_id
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        index.upsert(vectors=processed_data, namespace="ns1")
        
        return jsonify({
            "message": "File processed and uploaded successfully",
            "file_name": os.path.basename(object_key),
            "upload_session_id": upload_session_id,
            "file_url": file_url
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query_pinecone():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')
 
    if not question or not session_id:
        return jsonify({"error": "Question or session_id missing"}), 400
    
    try:
        embedding = embedding_model.encode(question).tolist()
        results = index.query(
            vector=embedding,
            namespace="ns1",
            top_k=3,
            include_metadata=True,
            filter={"upload_session_id": session_id}
        )
      
        response_data = [
            {
                "chunk_number": match['metadata'].get('chunk_number', "N/A"),
                "file_name": match['metadata'].get('file_name', "N/A"),
                "score": match.get('score', "N/A")
            }
            for match in results.get('matches', [])
        ]

        combined_summary = [match['metadata'].get('original_text', '') for match in results.get('matches', [])]
        final_summary = remove_duplicate_content(" ".join(combined_summary))
        
        return jsonify({
            "message": "Query successful",
            "results": response_data,
            "summary": final_summary
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error querying Pinecone: {str(e)}"}), 500

def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

@app.route('/summarize', methods=['POST'])
def summarize_file():
    text_content = request.form.get('text_content')
    file_url = request.form.get('file_url')

    if not file_url and not text_content:
        return jsonify({"error": "No file URL or text provided"}), 400

    try:
        if file_url:
            parsed_url = urlparse(file_url)
            bucket_name = parsed_url.netloc.split('.')[0]
            object_key = unquote(parsed_url.path.lstrip('/'))
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            pdf_bytes = response['Body'].read()
            text_chunks = extract_text_from_pdf(pdf_bytes)
            combined_text = " ".join(text_chunks)
        else:
            combined_text = text_content

        prompt = f"Summarize the following text concisely:\n\n{combined_text}\n\nSummary:"
        response = model.generate_content(prompt)
        summary_text = response.text

        try:
            audio_fp = generate_audio(summary_text)
            audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
        except Exception as audio_error:
            print(f"Audio generation error: {str(audio_error)}")
            audio_base64 = None
        
        return jsonify({
            "summary": summary_text,
            "audio": audio_base64
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error processing input: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)