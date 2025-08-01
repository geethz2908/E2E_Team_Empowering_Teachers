from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv
 
load_dotenv()
 
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
 
db_name = 'edusahayak-dev'
 
if db_name not in client.list_database_names():
    db = client[db_name]
    print(f"Database '{db_name}' created successfully.")
else:
    db = client[db_name]
    print(f"Database '{db_name}' already exists.")
 
collections = [
    'documents',
    'chats',
    'unanswered_questions',
    'users',
    'feedback'
]
 
for collection_name in collections:
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
 
users_collection = db['users']
admin_user = {
    'username': 'admin',
    'password': generate_password_hash('admin'),
    'role': 'admin',
    'active': True
}
 
if users_collection.find_one({'username': 'admin'}):
    print("Admin user already exists.")
else:
    users_collection.insert_one(admin_user)
    print("Admin user created successfully.")
 
client.close()
