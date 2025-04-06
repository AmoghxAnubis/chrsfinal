from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User:
    def __init__(self, email, password=None):
        self.email = email
        self.password = generate_password_hash(password) if password else None

    @staticmethod
    def get_by_email(email):
        from app import mongo  # Avoid circular imports
        return mongo.db.users.find_one({"email": email})

    def check_password(self, password):
        return check_password_hash(self.password, password)