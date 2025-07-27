import os
import threading
from flask import Flask, request, jsonify, session, stream_with_context, Response
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from pymongo import MongoClient
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime
from bson.objectid import ObjectId, InvalidId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from cryptography.fernet import Fernet, InvalidToken
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Config ---
API_KEY = 'supersecretapikey'
UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'html'}
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.secret_key = 'f4b5e6d7a8c9d0e1f2a3b4c5d6e7f8a9'

#Uploads Folder Creation
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o755)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

#Vector Store Folder Creation
VECTOR_STORES_FOLDER = 'vector_stores'
os.makedirs(VECTOR_STORES_FOLDER, exist_ok=True)
os.chmod(VECTOR_STORES_FOLDER, 0o755) 

google_api_key = os.getenv("GOOGLE_API_KEY")
app.config["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --- MongoDB ---
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['edusahayak-dev']
pdf_collection = db['documents']
chats_collection = db['chats']
unanswered_questions_collection = db['unanswered_questions']
users_collection = db['users']
feedback_collection = db['feedback']

user_locks = {}

groq_api_key = os.getenv('GROQ_API_KEY')

# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     # model_name="Llama3-8b-8192",
#     model_name="llama-3.1-8b-instant",
#     temperature=0.5,  # Adjust the temperature value as needed
#     model_kwargs={"stream":True},
#     max_tokens=6192
#     #callbacks= [StreamingStdOutCallbackHandler()]
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = ChatPromptTemplate.from_template(
    """
    User: {context}
    ---
    You are a educative support executive for Edu-Sahayak. Your task is to answer the user's question based on the provided context only.
    Guidelines:
    - Use clear and concise language.
    - Keep your responses very short and simple. Donot give more information than what is asked.
    - Always give the response in HTML format with good structure and good formatting.
    - Maintain a professional and helpful tone.
    - Avoid technical jargon unless necessary.
    - If the question is not related to the context, respond with "I am sorry, I donot have the information regarding this at present".
    Question: {input}
    Answer:
    """
)

def get_user_lock(user_id):
    if user_id not in user_locks:
        user_locks[user_id] = threading.Lock()
    return user_locks[user_id]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Key Auth Decorator ---
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('x-api-key')
        if key != API_KEY:
            return {'message': 'Unauthorized'}, 401
        return f(*args, **kwargs)
    return decorated

# --- Swagger Setup ---
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'x-api-key',
        'description': 'API Key for authorization'
    },
    'userid': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'x-user-id'
    },
    'username': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'x-username'
    }
}

api = Api(
    app,
    version='1.0',
    title='EduSahayak API',
    description='Secure API with API Key',
    doc='/swagger',
    authorizations=authorizations,
    security=['apikey', 'userid', 'username']
)

# --- Login ---
login_model = api.model('Login', {
    'username': fields.String(required=True, description='The username'),
    'password': fields.String(required=True, description='The user password')
})

@api.route('/api/login')
class Login(Resource):
    @api.expect(login_model)
    @api.doc(security=['apikey'])  # 👈 Disables API key security for this route
    @api.response(200, 'Login successful')
    @api.response(401, 'Invalid credentials or inactive user')
    def post(self):
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return {'message': 'Username and password are required.'}, 400

        user = users_collection.find_one({'username': username})

        if not user:
            return {'message': 'Invalid username or password.'}, 401

        if not user.get('active', False):
            return {'message': 'Your account is inactive. Please contact the administrator.'}, 401

        if not check_password_hash(user['password'], password):
            return {'message': 'Invalid username or password.'}, 401

        session['username'] = username
        session['user_id'] = str(user['_id'])
        session['role'] = user['role']

        return {
            'message': 'Login successful.',
            'username': username,
            'role': user['role']
        }, 200

# ---- Upload Document ---
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('name', location='form', required=True, help='Document name')
upload_parser.add_argument('version', location='form', default='1', help='Document version')
upload_parser.add_argument('description', location='form', default='Default description', help='Document description')
upload_parser.add_argument('assigned_agents', location='form', help='Comma-separated agent usernames')
upload_parser.add_argument('file', type=FileStorage, location='files', required=True, help='PDF file(s) to upload',action='append')

@api.route('/api/upload')
class FileUpload(Resource):
    @api.expect(upload_parser)
    @api.doc(
        security=['apikey', 'userid', 'username'],
        consumes=['multipart/form-data'],
        params={
            'name': {
                'in': 'formData',
                'type': 'string',
                'required': True,
                'description': 'Document name'
            },
            'version': {
                'in': 'formData',
                'type': 'string',
                'required': False,
                'default': '1',
                'description': 'Document version'
            },
            'description': {
                'in': 'formData',
                'type': 'string',
                'required': False,
                'default': 'Default description',
                'description': 'Document description'
            },
            'assigned_agents': {
                'in': 'formData',
                'type': 'string',
                'required': False,
                'description': 'Comma-separated agent usernames (e.g., agent1,agent2)'
            },
            'file': {
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'PDF file to upload'
            }
        }
    )
    @require_api_key
    def post(self):
        """
        Upload one or more PDF files with metadata.
        Requires headers: x-api-key, x-user-id, x-username
        """
        user_id = request.headers.get('x-user-id')
        username = request.headers.get('x-username', 'anonymous')

        if not user_id:
            return {'message': 'Missing user ID'}, 400

        lock = get_user_lock(user_id)
        with lock:
            args = upload_parser.parse_args()
            name = args['name']
            version = args.get('version', '1')
            description = args.get('description', 'Default description')
            raw_assigned_agents = args.get('assigned_agents', '')
            files = args['file']

            if not files:
                return {'message': 'No files provided'}, 400

            # Resolve usernames to agent ObjectIDs
            assigned_agent_usernames = raw_assigned_agents.split(',')
            assigned_agent_ids = []

            for agent_username in map(str.strip, assigned_agent_usernames):
                if agent_username:
                    agent = users_collection.find_one({'username': agent_username, 'role': 'agent'})
                    if agent:
                        assigned_agent_ids.append(str(agent['_id']))
                    else:
                        logging.warning(f"[UPLOAD] Agent not found for username: {agent_username}")

            saved_files = []

            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    file_size = os.path.getsize(filepath)
                    created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    document_data = {
                        'name': name,
                        'version': version,
                        'description': description,
                        'filename': filename,
                        'filepath': filepath,
                        'filesize': file_size,
                        'assigned_agents': assigned_agent_ids,
                        'processed_agents': [],
                        'created_by': user_id,  # Use user ID instead of just username
                        'created_at': created_time
                    }

                    pdf_collection.insert_one(document_data)
                    os.chmod(filepath, 0o644)
                    saved_files.append(filename)
                else:
                    return {'message': f'Invalid file type: {file.filename}'}, 400

            return {
                'message': 'Files uploaded successfully.',
                'uploaded_files': saved_files
            }, 200

# --- Update and Delete Document by ID ---
update_parser = reqparse.RequestParser()
update_parser.add_argument('name', type=str, required=False, help='New document name')
update_parser.add_argument('version', type=str, required=False, help='New version number')
update_parser.add_argument('description', type=str, required=False, help='New description')
update_parser.add_argument('assigned_agents', type=str, required=False, 
                          help='Comma-separated list of agent IDs')
update_parser.add_argument('file', type=FileStorage, location='files', required=False, 
                          help='New PDF file')

@api.route('/api/document/<string:document_id>')
class DocumentResource(Resource):
    @api.doc(
        security=['apikey', 'userid', 'username'],
        params={
            'document_id': {
                'description': 'MongoDB document ID',
                'in': 'path',
                'type': 'string',
                'required': True
            }
        },
        responses={
            200: 'Document deleted successfully',
            404: 'Document not found',
            500: 'Internal server error'
        }
    )
    @require_api_key
    def delete(self, document_id):
        """Delete a document by its ID
        ---
        tags:
          - Documents
        """
        try:
            doc = pdf_collection.find_one({'_id': ObjectId(document_id)})
            if not doc:
                return {'message': 'Document not found'}, 404
            
            if os.path.exists(doc.get('filepath', '')):
                os.remove(doc['filepath'])
            
            result = pdf_collection.delete_one({'_id': ObjectId(document_id)})
            return {'message': 'Document deleted successfully'}, 200
            
        except InvalidId:
            return {'message': 'Invalid document ID format'}, 400
        except Exception as e:
            return {'message': f'Error deleting document: {str(e)}'}, 500

    @api.expect(update_parser)
    @api.doc(
        security=['apikey', 'userid', 'username'], 
        consumes=['multipart/form-data'],
        responses={
            200: 'Document updated successfully',
            400: 'Invalid request',
            404: 'Document not found',
            500: 'Internal server error'
        }
    )
    @require_api_key
    def put(self, document_id):
        """Update a document by its ID (metadata and/or file)
        ---
        tags:
          - Documents
        """
        try:
            doc = pdf_collection.find_one({'_id': ObjectId(document_id)})
            if not doc:
                return {'message': 'Document not found'}, 404

            args = update_parser.parse_args()
            updates = {}

            if args['name']:
                updates['name'] = args['name']
            if args['version']:
                updates['version'] = args['version']
            if args['description']:
                updates['description'] = args['description']
            if args['assigned_agents']:
                updates['assigned_agents'] = args['assigned_agents'].split(',')

            file = args.get('file')
            if file:
                if not allowed_file(file.filename):
                    return {'message': 'Invalid file type'}, 400
                    
                if os.path.exists(doc.get('filepath', '')):
                    os.remove(doc['filepath'])
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                os.chmod(filepath, 0o644)
                
                updates.update({
                    'filename': filename,
                    'filepath': filepath,
                    'filesize': os.path.getsize(filepath),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            if updates:
                pdf_collection.update_one(
                    {'_id': ObjectId(document_id)}, 
                    {'$set': updates}
                )
                return {'message': 'Document updated successfully'}, 200
            return {'message': 'No valid fields provided for update'}, 400

        except InvalidId:
            return {'message': 'Invalid document ID format'}, 400
        except Exception as e:
            return {'message': f'Error updating document: {str(e)}'}, 500

# --- Delete All Docs Assigned to an Agent ---
@api.route('/api/documents/agent/<string:agent_id>')
class DeleteAgentDocs(Resource):
    @api.doc(
        security=['apikey', 'userid', 'username'],
        params={
            'agent_id': {
                'description': 'Agent user ID',
                'in': 'path',
                'type': 'string',
                'required': True
            }
        },
        responses={
            200: 'Documents deleted successfully',
            404: 'Agent not found',
            500: 'Internal server error'
        }
    )
    @require_api_key
    def delete(self, agent_id):
        """Delete all documents assigned to an agent
        ---
        tags:
          - Documents
        """
        try:
            # Verify agent exists
            agent = users_collection.find_one({'_id': ObjectId(agent_id), 'role': 'agent'})
            if not agent:
                return {'success': False, 'message': 'Agent not found'}, 404

            # Find all documents assigned to this agent
            docs = pdf_collection.find({'assigned_agents': agent_id})
            
            # Delete associated files
            deleted_count = 0
            for doc in docs:
                if os.path.exists(doc.get('filepath', '')):
                    os.remove(doc['filepath'])
                deleted_count += 1

            # Delete from database
            result = pdf_collection.delete_many({'assigned_agents': agent_id})
            
            return {
                'success': True,
                'message': f'Deleted {result.deleted_count} documents assigned to {agent["username"]}',
                'deleted_count': result.deleted_count
            }, 200
            
        except InvalidId:
            return {'success': False, 'message': 'Invalid agent ID format'}, 400
        except Exception as e:
            return {'success': False, 'message': str(e)}, 500

# --- Vector Embedding API ---
class EnhancedPyPDFLoader(PyPDFLoader):
    def load(self):
        documents = super().load()
        # Enhance documents with page numbers
        for i, doc in enumerate(documents):
            doc.metadata['page_number'] = i + 1  # Page numbers typically start from 1
        return documents

class EnhancedTextLoader(TextLoader):
    def lazy_load(self):
        encodings = ['utf-8', 'latin1', 'cp1252']  # Add more encodings if needed
        for encoding in encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    yield Document(page_content=text, metadata={"source": self.file_path})
                    break  # Exit the loop if successful
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError(f"Error loading {self.file_path}: Unable to decode file with any of the specified encodings.")
        
def check_file_permissions(filepath):
    if not os.access(filepath, os.R_OK):
        logging.error(f"No read permission for file: {filepath}")
        return False
    return True

def get_document_loader(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None

    if not check_file_permissions(filepath):
        return None

    try:
        if filepath.endswith('.pdf'):
            return EnhancedPyPDFLoader(filepath)
        elif filepath.endswith('.txt'):
            return EnhancedTextLoader(filepath)
        elif filepath.endswith(('.docx', '.csv', '.xlsx', '.xls')):
            return TextLoader(filepath)
        elif filepath.endswith(('.pptx', '.ppt')):
            return TextLoader(filepath)  # Extract text from pptx using a library (e.g., python-pptx)
        elif filepath.endswith('.html'):
            return TextLoader(filepath)  # Use TextLoader for HTML files as well
        else:
            return None
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {str(e)}")
        return None
    
def retrieve_documents_from_mongodb(agent_id, field='assigned_agents'):
    docs = []
    query = {field: agent_id}
    documents = pdf_collection.find(query)

    for doc in documents:
        filepath = doc.get('filepath')
        filename = doc.get('filename')

        loader = get_document_loader(filepath)  # You must define this function
        if loader:
            loaded_docs = loader.load()
            for loaded_doc in loaded_docs:
                loaded_doc.metadata.update({
                    'filename': filename,
                    'page_number': loaded_doc.metadata.get('page_number', 'Unknown')
                })
            docs.extend(loaded_docs)
    return docs

def split_documents_for_embedding(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    return text_splitter.split_documents(docs)

def vector_store_process_all_documents(agent_id):
    agent = users_collection.find_one({'_id': ObjectId(agent_id), 'role': 'agent'})
    if not agent:
        return {"success": False, "message": "Agent not found"}

    username = agent.get('username')
    docs = retrieve_documents_from_mongodb(agent_id, 'assigned_agents')
    print(f"[DEBUG] Retrieved {len(docs)} docs for agent: {agent_id}")
    
    if not docs:
        print("No documents to process")
        return {"success": False, "message": "No documents to process"}

    final_documents = split_documents_for_embedding(docs)

    # Save vector store
    vector_store_path = os.path.join(VECTOR_STORES_FOLDER, f"faiss_index_{username}")
    vectors = FAISS.from_documents(final_documents, app.config["embeddings"])  # embeddings must be preloaded
    vectors.save_local(vector_store_path)

    # Mark processed in DB
    pdf_collection.update_many(
        {'assigned_agents': agent_id, 'processed_agents': {'$ne': agent_id}},
        {'$addToSet': {'processed_agents': agent_id}}
    )

    return {"success": True, "message": f"Vector store created for {username}"}


vector_embedding_parser = reqparse.RequestParser()
vector_embedding_parser.add_argument('embed_all', type=bool, required=False, help='Process all agents (admin only)')
vector_embedding_parser.add_argument('agent_id', type=str, required=False, help='Agent ID to process')


@api.route('/api/vector-embedding')
class VectorEmbedding(Resource):
    @api.expect(vector_embedding_parser)
    @api.doc(
        description="Generate vector embeddings for uploaded documents. If 'embed_all' is true, all agents will be processed (admin only).",
        security=['apikey'],
        responses={
            200: 'Vector store created',
            400: 'Bad request',
            401: 'Unauthorized',
            500: 'Internal error'
        }
    )
    @require_api_key
    def post(self):
        """
        Trigger vector embedding creation from uploaded documents.
        """
        try:
            user_id = request.headers.get('x-user-id')
            username = request.headers.get('x-username', 'anonymous')

            if not user_id:
                return {"message": "Missing user ID"}, 401

            user = users_collection.find_one({'_id': ObjectId(user_id)})
            if not user:
                return {"message": "User not found"}, 404

            args = vector_embedding_parser.parse_args()
            embed_all = args.get('embed_all')
            agent_id = args.get('agent_id')
            response_messages = []

            if user.get('role') == 'admin':
                if embed_all:
                    agents = users_collection.find({'role': 'agent'})
                    for agent in agents:
                        result = vector_store_process_all_documents(agent['_id'])
                        response_messages.append(result["message"])
                    success = True
                    message = "Training done for all agents: " + "; ".join(response_messages)
                elif agent_id:
                    result = vector_store_process_all_documents(agent_id)
                    success = result["success"]
                    message = result["message"]
                else:
                    result = vector_store_process_all_documents(user_id)
                    success = result["success"]
                    message = result["message"]
            else:
                result = vector_store_process_all_documents(user_id)
                success = result["success"]
                message = result["message"]

            return {
                "success": success,
                "message": message,
                "current_page": "/ask" if success else "/"
            }, 200 if success else 400

        except Exception as e:
            logging.error(f"Vector embedding error: {str(e)}")
            return {
                "success": False,
                "message": f"Error in processing documents: {str(e)}",
                "current_page": "/"
            }, 500

# --- user creation api ---
class EncryptionManager:
    _instance = None
    
    def __new__(cls, key_file='secret.key'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, key_file='secret.key'):
        if not hasattr(self, 'initialized'):
            self.key = self._load_key(key_file)
            self.fernet = Fernet(self.key)
            self.initialized = True
    
    def _load_key(self, key_file):
        try:
            with open(key_file, 'rb') as file:
                return file.read()
        except FileNotFoundError:
            # Generate a new key if not found
            key = Fernet.generate_key()
            with open(key_file, 'wb') as file:
                file.write(key)
            return key
        except Exception as e:
            logging.error(f"Critical error loading encryption key: {e}")
            raise RuntimeError("Failed to initialize encryption")

    def encrypt(self, data):
        if not data:
            return None
        try:
            if isinstance(data, str):
                return self.fernet.encrypt(data.encode())
            return data  # Return as is if already encrypted
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            return None

    def decrypt(self, encrypted_data):
        if not encrypted_data:
            return None
        try:
            # Check if data is already decrypted
            if isinstance(encrypted_data, str):
                return encrypted_data
            
            decrypted = self.fernet.decrypt(encrypted_data).decode()
            return decrypted
        except InvalidToken:
            logging.warning(f"Invalid token during decryption: {encrypted_data[:30]}...")
            return encrypted_data if isinstance(encrypted_data, str) else None
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return None

encryption_mgr = EncryptionManager()

register_model = api.model('RegisterUser', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password'),
    'email': fields.String(required=True, description='Email'),
    'phone': fields.String(required=True, description='Phone number'),
    'role': fields.String(required=True, description='Role: admin, agent, user', enum=['admin', 'agent', 'user'])
})

@api.route('/api/register')
class RegisterUser(Resource):
    @api.expect(register_model)
    @api.doc(description="Register a new user with role: admin, agent, or user.")
    def post(self):
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        phone = data.get('phone')
        role = data.get('role')

        if users_collection.find_one({'username': username}):
            return {'message': 'Username already exists.'}, 400

        hashed_password = generate_password_hash(password)
        encrypted_email = encryption_mgr.encrypt(email)
        encrypted_phone = encryption_mgr.encrypt(phone)

        new_user = {
            'username': username,
            'password': hashed_password,
            'email': encrypted_email,
            'phone': encrypted_phone,
            'role': role,
            'active': True
        }

        users_collection.insert_one(new_user)
        return {'message': 'User registered successfully.'}, 201
    
# --- Get all users ---
user_model = api.model('User', {
    '_id': fields.String(description='User ID'),
    'username': fields.String(description='Username'),
    'role': fields.String(description='Role'),
    'active': fields.Boolean(description='Active status')
})

@api.route('/api/users')
class UserList(Resource):
    @api.doc(description="Get a list of all users (excluding sensitive info).")
    @api.marshal_list_with(user_model)
    def get(self):
        users = users_collection.find({}, {'username': 1, 'role': 1, 'active': 1})
        return [{**u, '_id': str(u['_id'])} for u in users]
    
# --- Delete a user by ID ---
user_delete_parser = reqparse.RequestParser()
user_delete_parser.add_argument('user_id', type=str, required=True, help='User ID to delete')

@api.route('/api/users/delete')
class UserDelete(Resource):
    @api.expect(user_delete_parser)
    @api.doc(description="Delete a user by ID.")
    def delete(self):
        args = user_delete_parser.parse_args()
        user_id = args['user_id']

        result = users_collection.delete_one({'_id': ObjectId(user_id)})
        if result.deleted_count:
            return {'message': 'User deleted successfully.'}, 200
        else:
            return {'message': 'User not found.'}, 404

# --- Ask API ---
def generate_response(question : str, retriever,retrieval_chain,timestamp):

    response_parts = []    
    def stream_response():
        chain = retrieval_chain.pick("answer")
        for token in chain.stream({"input" :question}):
            yield f"data: {token}\n\n"
            response_parts.append(token)
        model_response = "".join(response_parts)
        # model_response = "A special marriage under the Special Marriage Act, 1954, refers to a marriage between two persons who meet certain conditions, such as neither having a spouse living, being of sound mind, and not being related by affinity or kinship within the degrees of prohibited relationship."
        print(model_response)
        meta_data = for_highlight(question,retriever,retrieval_chain,timestamp,model_response)
        yield "data: [DONE]"
    return Response(stream_with_context(stream_response()), 
                   mimetype='text/event-stream')
    
def for_highlight(q, retriever,retrieval_chain,timestamp,answer):
    response = {}
    try:
        answer_source_docs = []

        if answer_source_docs:
            response={
                "response": answer,
            }
        else:
            response={
                "response": "Sorry, I could not find an answer.",
                "page_number": "Unknown",
                "filename": "Unknown",
            }
        # print(response)
        return response
    except Exception as e:
        logging.error(f"Error in processing question: {e.__traceback__.tb_lineno}") # type: ignore
        return {"response": f"Error in processing question: {str(e)}"}, 500
    
ask_parser = reqparse.RequestParser()
ask_parser.add_argument('question', type=str, required=True, help='User question')
ask_parser.add_argument('agent', type=str, required=True, help='Agent ID')

@api.route('/api/ask')
class AskQuestion(Resource):
    @api.expect(ask_parser)
    @api.doc(
            description="Ask a question based on the uploaded documents for a given agent.",
            security=['apikey', 'userid', 'username'],
        responses={
            200: 'Answer generated',
            400: 'Bad request',
            401: 'Unauthorized',
            500: 'Internal server error',
        }
    )
    @require_api_key
    def post(self):
        """
        Ask a question and get answer from the relevant documents for an agent.
        """
        try:
            args = ask_parser.parse_args()
            question = args['question']
            agent_id = args['agent']

            if not question or not agent_id:
                return {"response": "Missing question or agent ID"}, 400

            # Get agent object and resolve username
            agent = users_collection.find_one({'_id': ObjectId(agent_id), 'role': 'agent'})
            if not agent:
                return {"response": "Agent not found"}, 404

            username = agent.get('username')
            vector_store_path = os.path.join(VECTOR_STORES_FOLDER, f"faiss_index_{username}")

            if not os.path.exists(vector_store_path):
                return {"response": "Vector store not available."}, 500

            try:
                vectors = FAISS.load_local(vector_store_path, app.config["embeddings"], allow_dangerous_deserialization=True)
            except Exception as e:
                logging.error(f"Error loading vector store: {str(e)}")
                return {"response": f"Error loading vector store: {str(e)}"}, 500

            if not vectors:
                return {"response": "Vector store not available."}, 500

            # Prepare chains
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            document_chain = create_stuff_documents_chain(llm, prompt)

            retriever = vectors.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # For now just one question; you can expand to multi-question later
            response_parts = []

            chain = retrieval_chain.pick("answer")
            for token in chain.stream({"input": question}):
                response_parts.append(token)

            model_response = "".join(response_parts)

            # If you want metadata/highlighting support, call for_highlight() here
            return {"response": model_response}, 200

        except Exception as e:
            logging.error(f"Error in /api/ask: {str(e)}")
            return {"response": f"Internal server error: {str(e)}"}, 500

lesson_plan_parser = reqparse.RequestParser()
lesson_plan_parser.add_argument('grade', type=str, required=True, help='Grade/Class level')
lesson_plan_parser.add_argument('subject', type=str, required=True, help='Subject')
lesson_plan_parser.add_argument('chapter', type=str, required=True, help='Chapter or Topic')
lesson_plan_parser.add_argument('agent_id', type=str, required=True, help='Agent ID')

@api.route('/api/generate-lesson-plan')
class GenerateLessonPlan(Resource):
    @api.expect(lesson_plan_parser)
    @api.doc(description="Generate a structured lesson plan using uploaded content and vector knowledge base.",
             security=['apikey', 'userid', 'username'])
    @require_api_key
    def post(self):
        try:
            args = lesson_plan_parser.parse_args()
            grade = args['grade']
            subject = args['subject']
            chapter = args['chapter']
            agent_id = args['agent_id']

            # Fetch agent and vector store
            agent = users_collection.find_one({'_id': ObjectId(agent_id), 'role': 'agent'})
            if not agent:
                return {"message": "Agent not found."}, 404

            username = agent.get('username')
            vector_store_path = os.path.join(VECTOR_STORES_FOLDER, f"faiss_index_{username}")

            if not os.path.exists(vector_store_path):
                return {"message": "Vector store not available. Please embed first."}, 400

            # Load vector DB and get context
            vectors = FAISS.load_local(vector_store_path, app.config["embeddings"], allow_dangerous_deserialization=True)
            retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            vector_context_docs = retriever.get_relevant_documents(chapter)
            vector_context = "\n\n".join([doc.page_content for doc in vector_context_docs])

            # Get reference content from uploaded docs
            all_docs = retrieve_documents_from_mongodb(agent_id)
            combined_file_text = "\n\n".join([doc.page_content for doc in all_docs])

            # Prompt template
            lesson_prompt = ChatPromptTemplate.from_template("""
You are an experienced teacher assistant AI. Based on the following inputs, generate a structured, engaging, and pedagogically sound lesson plan.

### Inputs:
- Grade Level: {grade}
- Subject: {subject}
- Chapter/Topic: {chapter}
- Reference Content: {file_content}
- Knowledge Base Context: {vector_db_context}

---

### Output Required:
Generate a lesson plan in the following format:

1. Lesson Title:  
2. Grade & Subject:  
3. Chapter/Topic:  
4. Learning Objectives:  
5. Introduction/Engagement Activity:  
6. Core Teaching Content:  
7. Activities/Student Tasks:  
8. Assessment/Evaluation:  
9. Differentiation:  
10. Materials/Resources Needed:  
11. Homework/Follow-up:  
12. Teacher Tips or Notes:
            """)

            chain = lesson_prompt | llm
            output = chain.invoke({
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "file_content": combined_file_text[:10000],
                "vector_db_context": vector_context[:3000]
            })

            return {"lesson_plan": output.content}, 200

        except Exception as e:
            logging.error(f"Error generating lesson plan: {str(e)}")
            return {"message": f"Internal server error: {str(e)}"}, 500

# ADD THIS NEAR THE END OF YOUR EXISTING api.py
# Right before: if __name__ == '__main__':

lesson_plan_parser = reqparse.RequestParser()
lesson_plan_parser.add_argument('grade', type=str, required=True, help='Grade/Class level')
lesson_plan_parser.add_argument('subject', type=str, required=True, help='Subject')
lesson_plan_parser.add_argument('chapter', type=str, required=True, help='Chapter or Topic')
lesson_plan_parser.add_argument('agent_id', type=str, required=True, help='Agent ID')

@api.route('/api/generate-lesson-plan')
class GenerateLessonPlan(Resource):
    @api.expect(lesson_plan_parser)
    @api.doc(description="Generate a structured lesson plan using uploaded content and vector knowledge base.",
             security=['apikey', 'userid', 'username'])
    @require_api_key
    def post(self):
        try:
            args = lesson_plan_parser.parse_args()
            grade = args['grade']
            subject = args['subject']
            chapter = args['chapter']
            agent_id = args['agent_id']

            # Fetch agent and vector store
            agent = users_collection.find_one({'_id': ObjectId(agent_id), 'role': 'agent'})
            if not agent:
                return {"message": "Agent not found."}, 404

            username = agent.get('username')
            vector_store_path = os.path.join(VECTOR_STORES_FOLDER, f"faiss_index_{username}")

            if not os.path.exists(vector_store_path):
                return {"message": "Vector store not available. Please embed first."}, 400

            # Load vector DB and get context
            vectors = FAISS.load_local(vector_store_path, app.config["embeddings"], allow_dangerous_deserialization=True)
            retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            vector_context_docs = retriever.get_relevant_documents(chapter)
            vector_context = "\n\n".join([doc.page_content for doc in vector_context_docs])

            # Get reference content from uploaded docs
            all_docs = retrieve_documents_from_mongodb(agent_id)
            combined_file_text = "\n\n".join([doc.page_content for doc in all_docs])

            # Prompt template
            lesson_prompt = ChatPromptTemplate.from_template("""
You are an experienced teacher assistant AI. Based on the following inputs, generate a structured, engaging, and pedagogically sound lesson plan.

### Inputs:
- Grade Level: {grade}
- Subject: {subject}
- Chapter/Topic: {chapter}
- Reference Content: {file_content}
- Knowledge Base Context: {vector_db_context}

---

### Output Required:
Generate a lesson plan in the following format:

1. Lesson Title:  
2. Grade & Subject:  
3. Chapter/Topic:  
4. Learning Objectives:  
5. Introduction/Engagement Activity:  
6. Core Teaching Content:  
7. Activities/Student Tasks:  
8. Assessment/Evaluation:  
9. Differentiation:  
10. Materials/Resources Needed:  
11. Homework/Follow-up:  
12. Teacher Tips or Notes:
            """)

            chain = lesson_prompt | llm
            output = chain.invoke({
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "file_content": combined_file_text[:10000],
                "vector_db_context": vector_context[:3000]
            })

            return {"lesson_plan": output.content}, 200

        except Exception as e:
            logging.error(f"Error generating lesson plan: {str(e)}")
            return {"message": f"Internal server error: {str(e)}"}, 500


# Worksheet Generator API
worksheet_parser = reqparse.RequestParser()
worksheet_parser.add_argument('grade', type=str, required=True, help='Grade/Class level')
worksheet_parser.add_argument('subject', type=str, required=True, help='Subject')
worksheet_parser.add_argument('chapter', type=str, required=True, help='Chapter or Topic')
worksheet_parser.add_argument('difficulty', type=str, required=False, default='medium', help='Difficulty level (easy, medium, hard)')

@api.route('/api/generate-worksheet')
class GenerateWorksheet(Resource):
    @api.expect(worksheet_parser)
    @api.doc(description="Generate a JSON worksheet based on subject, chapter, and grade.",
             security=['apikey', 'userid', 'username'])
    @require_api_key
    def post(self):
        try:
            args = worksheet_parser.parse_args()
            grade = args['grade']
            subject = args['subject']
            topic = args['chapter']
            difficulty = args['difficulty']

            prompt_text = f"""Generate a worksheet JSON for {subject} on {topic}.

GRADE LEVEL: {grade}
DIFFICULTY: {difficulty}

REQUIREMENTS:
- Create a JSON object with a 'worksheet' array
- Exactly 9 elements in the array
- First element: Worksheet title
- Next 5 elements: Comprehension/knowledge questions
- Next 2 elements: Multiple-choice questions
- Last element: Cloze passage

EXAMPLE FORMAT:
{{
    \"worksheet\": [
        \"{subject} Worksheet: {topic} for {grade} - {difficulty} level\",
        \"Define the primary concept of {topic}\",
        \"Explain the significance of a key aspect in {topic}\",
        \"Analyze the relationship between two core ideas\",
        \"Describe the main characteristics of the subject\",
        \"Compare and contrast different perspectives\",
        \"Multiple choice: Which statement best describes...\",
        \"Multiple choice: Select the correct explanation for...\",
        \"Complete the passage about {topic} by filling in the blanks...\"
    ]
}}

Be specific, educational, and ensure clear, engaging questions appropriate for {grade} at a {difficulty} level."""

            response = llm.invoke(prompt_text)
            return {"worksheet": response.content}, 200

        except Exception as e:
            logging.error(f"Error generating worksheet: {str(e)}")
            return {"message": f"Internal server error: {str(e)}"}, 500
                           
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
