import pymongo
from pymongo import MongoClient

def test_mongo_connection():
    try:
        client = MongoClient("mongodb+srv://zaidali:IUzvdpQZ7MMaixB8@cluster0.qmehy3e.mongodb.net/Todo?retryWrites=true&w=majority")
        db = client.Todo
        
        # Check if the connection is successful
        collections = db.list_collection_names()
        print("Connected to MongoDB successfully!")
        print("Available collections:", collections)
    except Exception as e:
        print("Error connecting to MongoDB:", e)

if __name__ == "__main__":
    test_mongo_connection()
