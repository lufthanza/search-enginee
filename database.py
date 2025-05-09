import sqlite3
import os
import hashlib
from datetime import datetime

class DocumentDatabase:
    def __init__(self, db_path="documents.db"):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # Initialize database
        if not self.initialize_database():
            raise Exception("Failed to initialize database")

    def connect(self):
        """Create database connection"""
        try:
            if self.conn is None:
                # Ensure database file exists
                if not os.path.exists(self.db_path):
                    # Create empty database file
                    with open(self.db_path, 'w') as f:
                        pass
                
                self.conn = sqlite3.connect(self.db_path)
                self.cursor = self.conn.cursor()
                print(f"Successfully connected to database at {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error connecting to database: {e}")
            return False

    def close(self):
        """Close database connection"""
        try:
            if self.conn:
                self.conn.close()
                print("Database connection closed")
        except sqlite3.Error as e:
            print(f"Error closing database connection: {e}")
        except Exception as e:
            print(f"Unexpected error closing database connection: {e}")
        finally:
            self.conn = None
            self.cursor = None

    def check_column_exists(self, table_name, column_name):
        """Check if a column exists in a table"""
        if not self.connect():
            return False
            
        try:
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in self.cursor.fetchall()]
            return column_name in columns
        except sqlite3.Error as e:
            print(f"Error checking column existence: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking column existence: {e}")
            return False
        finally:
            self.close()

    def add_column_if_not_exists(self, table_name, column_name, column_type):
        """Add a column to a table if it doesn't exist"""
        if not self.check_column_exists(table_name, column_name):
            if not self.connect():
                return False
                
            try:
                self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                print(f"Error adding column: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error adding column: {e}")
                return False
            finally:
                self.close()
        return True

    def initialize_database(self):
        """Create necessary tables if they don't exist and handle migrations"""
        print("Initializing database...")
        if not self.connect():
            print("Failed to connect to database during initialization")
            return False
            
        try:
            # Create documents table if it doesn't exist
            print("Creating documents table if not exists...")
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    file_hash TEXT UNIQUE,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    processed BOOLEAN DEFAULT 0
                )
            ''')
            
            # Create processed_content table if it doesn't exist
            print("Creating processed_content table if not exists...")
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    sentence_index INTEGER,
                    original_text TEXT,
                    processed_text TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            self.conn.commit()
            print("Database initialization completed successfully")
            return True
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error initializing database: {e}")
            return False
        finally:
            self.close()

    def get_file_hash(self, content):
        """Generate hash for file content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def document_exists(self, file_hash):
        """Check if document already exists in database"""
        if not self.connect():
            return False
            
        try:
            self.cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            result = self.cursor.fetchone()
            return result is not None
        except sqlite3.Error as e:
            print(f"Error checking document existence: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking document existence: {e}")
            return False
        finally:
            self.close()

    def add_document(self, filename, content, file_size):
        """Add new document to database"""
        file_hash = self.get_file_hash(content)
        
        # Check if document already exists
        if self.document_exists(file_hash):
            print(f"Document {filename} already exists in database")
            return None
        
        if not self.connect():
            return None
            
        try:
            # Insert into documents table
            self.cursor.execute('''
                INSERT INTO documents (filename, content, file_hash, file_size, processed)
                VALUES (?, ?, ?, ?, 0)
            ''', (filename, content, file_hash, file_size))
            
            document_id = self.cursor.lastrowid
            self.conn.commit()
            print(f"Successfully added document {filename} with ID {document_id}")
            return document_id
        except sqlite3.Error as e:
            print(f"Error adding document: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error adding document: {e}")
            return None
        finally:
            self.close()

    def add_processed_content(self, document_id, sentence_index, original_text, processed_text):
        """Add processed content for a document"""
        if not self.connect():
            return False
            
        try:
            self.cursor.execute('''
                INSERT INTO processed_content (document_id, sentence_index, original_text, processed_text)
                VALUES (?, ?, ?, ?)
            ''', (document_id, sentence_index, original_text, processed_text))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding processed content: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error adding processed content: {e}")
            return False
        finally:
            self.close()

    def get_all_documents(self):
        """Get all documents from database"""
        if not self.connect():
            return []
            
        try:
            self.cursor.execute("SELECT id, filename, content FROM documents")
            results = self.cursor.fetchall()
            print(f"Retrieved {len(results)} documents from database")
            return results
        except sqlite3.Error as e:
            print(f"Error getting all documents: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error getting all documents: {e}")
            return []
        finally:
            self.close()

    def get_document_by_id(self, document_id):
        """Get specific document by ID"""
        if not self.connect():
            return None
            
        try:
            self.cursor.execute("SELECT id, filename, content FROM documents WHERE id = ?", (document_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting document by ID: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting document by ID: {e}")
            return None
        finally:
            self.close()

    def get_processed_content(self, document_id):
        """Get processed content for a document"""
        if not self.connect():
            return []
            
        try:
            self.cursor.execute('''
                SELECT sentence_index, original_text, processed_text 
                FROM processed_content 
                WHERE document_id = ?
                ORDER BY sentence_index
            ''', (document_id,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting processed content: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error getting processed content: {e}")
            return []
        finally:
            self.close()

    def mark_document_processed(self, document_id):
        """Mark document as processed"""
        if not self.connect():
            return False
            
        try:
            self.cursor.execute("UPDATE documents SET processed = 1 WHERE id = ?", (document_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error marking document as processed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error marking document as processed: {e}")
            return False
        finally:
            self.close()

    def get_unprocessed_documents(self):
        """Get all unprocessed documents"""
        if not self.connect():
            return []
            
        try:
            self.cursor.execute("SELECT id, filename, content FROM documents WHERE processed = 0")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting unprocessed documents: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error getting unprocessed documents: {e}")
            return []
        finally:
            self.close()

    def clear_processed_content(self, document_id):
        """Clear processed content for a document"""
        if not self.connect():
            return False
            
        try:
            self.cursor.execute("DELETE FROM processed_content WHERE document_id = ?", (document_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error clearing processed content: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error clearing processed content: {e}")
            return False
        finally:
            self.close()

    def get_document_stats(self):
        """Get statistics about documents in database"""
        if not self.connect():
            return {
                'total_documents': 0,
                'total_size': 0,
                'processed_documents': 0
            }
            
        try:
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    SUM(file_size) as total_size,
                    COUNT(CASE WHEN processed = 1 THEN 1 END) as processed_documents
                FROM documents
            """)
            stats = self.cursor.fetchone()
            result = {
                'total_documents': stats[0] if stats[0] is not None else 0,
                'total_size': stats[1] if stats[1] is not None else 0,
                'processed_documents': stats[2] if stats[2] is not None else 0
            }
            print(f"Database stats: {result}")
            return result
        except sqlite3.Error as e:
            print(f"Error getting document stats: {e}")
            return {
                'total_documents': 0,
                'total_size': 0,
                'processed_documents': 0
            }
        except Exception as e:
            print(f"Unexpected error getting document stats: {e}")
            return {
                'total_documents': 0,
                'total_size': 0,
                'processed_documents': 0
            }
        finally:
            self.close() 