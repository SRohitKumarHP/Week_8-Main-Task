import sqlite3
import hashlib
import secrets
from pathlib import Path


DB_PATH = Path("app/users.db")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password, salt=None):

    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000
    )

    return salt.hex(), password_hash.hex()


def verify_password(
    password,
    salt_hex,
    hash_hex
):

    salt = bytes.fromhex(salt_hex)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000
    )

    return secrets.compare_digest(
        password_hash.hex(),
        hash_hex
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    # Create default Admin if it does not exist
    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    existing_admin = cursor.fetchone()

    if existing_admin is None:

        salt, password_hash = hash_password(
            "admin123"
        )

        cursor.execute(
            """
            INSERT INTO users
            (username, password_hash, salt, role)
            VALUES (?, ?, ?, ?)
            """,
            (
                "admin",
                password_hash,
                salt,
                "Admin"
            )
        )

        connection.commit()

    connection.close()


# ============================================================
# USER AUTHENTICATION
# ============================================================

def authenticate_user(
    username,
    password
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT username, password_hash, salt, role
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    user = cursor.fetchone()

    connection.close()

    if user is None:

        return None

    db_username, password_hash, salt, role = user

    if verify_password(
        password,
        salt,
        password_hash
    ):

        return {
            "username": db_username,
            "role": role
        }

    return None


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    username,
    password,
    role
):

    connection = get_connection()
    cursor = connection.cursor()

    salt, password_hash = hash_password(
        password
    )

    try:

        cursor.execute(
            """
            INSERT INTO users
            (username, password_hash, salt, role)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                password_hash,
                salt,
                role
            )
        )

        connection.commit()

        success = True

    except sqlite3.IntegrityError:

        success = False

    connection.close()

    return success


# ============================================================
# GET ALL USERS
# ============================================================

def get_users():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, username, role, created_at
        FROM users
        ORDER BY username
        """
    )

    users = cursor.fetchall()

    connection.close()

    return users


# ============================================================
# DELETE USER
# ============================================================

def delete_user(
    username,
    current_username=None
):

    # --------------------------------------------------------
    # Prevent deleting currently logged-in user
    # --------------------------------------------------------

    if current_username is not None:

        if username == current_username:

            return (
                False,
                "You cannot delete the account you are currently logged in with."
            )


    connection = get_connection()
    cursor = connection.cursor()


    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT id, role
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    user = cursor.fetchone()


    if user is None:

        connection.close()

        return (
            False,
            "User does not exist."
        )


    user_id, role = user


    # --------------------------------------------------------
    # Prevent deleting final Admin
    # --------------------------------------------------------

    if role == "Admin":

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM users
            WHERE role = 'Admin'
            """
        )

        admin_count = cursor.fetchone()[0]

        if admin_count <= 1:

            connection.close()

            return (
                False,
                "The last Admin account cannot be deleted."
            )


    # --------------------------------------------------------
    # Delete user
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM users
        WHERE id = ?
        """,
        (user_id,)
    )

    connection.commit()

    connection.close()


    return (
        True,
        f"User '{username}' deleted successfully."
    )