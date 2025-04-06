from flask import request, jsonify
from werkzeug.security import check_password_hash
from app.models import User  # Import your User model
from app import app  # Import your Flask app instance

@app.route('/api/login', methods=['POST'])  # Note: methods should be a list ['POST']
def login():
    data = request.get_json()  # Fixed method name (get_json, not get.json)
    user = User.get_by_email(data['email'])
    
    # Check if user exists and password matches
    if not user or not User.check_password(user, data['password']):  # Fixed password check
        return jsonify({"error": "Invalid credentials"}), 401  # Fixed status code (401, not 40!)
    
    return jsonify({"message": "Logged in successfully"})