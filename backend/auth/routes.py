from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import jwt  # For JWT token generation
from functools import wraps

load_dotenv()

auth_bp = Blueprint('auth', __name__)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME", "auth_demo")]
users_collection = db.users

# JWT Secret Key (should be in .env)
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
JWT_ALGORITHM = "HS256"

@auth_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate input
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Check if user exists
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create user
        user = {
            "email": email,
            "password": hashed_password.decode('utf-8'),  # Store as string
            "createdAt": datetime.utcnow()
        }
        result = users_collection.insert_one(user)

        return jsonify({
            "success": True,
            "user": {
                "id": str(result.inserted_id),
                "email": email
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate input
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Find user
        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({"error": "Invalid credentials"}), 401

        # Create JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(days=1)  # Token expires in 1 day
        }, JWT_SECRET, algorithm=JWT_ALGORITHM)

        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": str(user['_id']),
                "email": user['email']
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Example of a protected route (requires valid JWT)
@auth_bp.route('/protected', methods=['GET'])
def protected():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header missing or invalid"}), 401

        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        user_id = payload['user_id']
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "message": f"Hello {user['email']}, this is a protected route!",
            "user": {
                "id": str(user['_id']),
                "email": user['email']
            }
        })

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500