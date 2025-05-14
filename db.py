# Database related all Functions here

import os
import mysql.connector
from mysql.connector import pooling, Error, errorcode
from dotenv import load_dotenv

load_dotenv(override = True)


pool = None

def get_connection():

    """
    Making the connection with the database using MySQL Pooling method
    """

    global pool

    try:
        if pool is None:
            pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name = "bookkeeping_pool",
            pool_size = 5,
            host = os.getenv("DB_HOST"),
            database = os.getenv("DB_NAME"),
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD"),
            port = os.getenv("DB_PORT"),
            connect_timeout = 30,
            auth_plugin = "mysql_native_password",
            raise_on_warnings = True,
            # use_pure = True,
            # ssl_disabled = True


        )

        return pool.get_connection()
    
    except mysql.connector.Error as e:

        if e.errno == errorcode.CR_CONN_HOST_ERROR:
            raise Exception("Server IP is not whitelisted. Please add this IP to the database whitelist")
        
        else:
            raise Exception(f"Database connection failed {str(e)}")
    


def execute_query(query: str) ->str:
    """
    Execute any SQL query using this function (behind the hood - connection pool)
    """
    conn = None
    cursor = None  # Correct variable name

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        return result

    except Exception as e:
        return [f"Error: {str(e)}"]

    finally:
        # Important to close the connection
        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


def fetch_schema():
    
    """
    Fetching the database schema, Later will pass this on to LLM
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES")
        tables = [t[0] for t in cursor.fetchall()]

        schema_text = ""

        for table in tables:
            cursor.execute(f"SHOW COLUMNS FROM {table}")
            columns = [ col[0] for col in cursor.fetchall()]

            schema_text += f"{table}({','.join(columns)})\n"

        return schema_text
    
    except Exception as e:
        return f"Error : {str(e)}"
    
    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()

# print(execute_query("SHOW TABLES"))






