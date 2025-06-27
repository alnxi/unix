"""
Data Encryption and Security Utilities

This module provides utilities for encrypting data at rest and in transit,
secure key management, and cryptographic hashing with best security practices.

Features:
- AES encryption for data at rest
- Secure key generation and management
- Data hashing with salt
- File encryption/decryption
- Memory-safe operations
"""

import os
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    algorithm: str = "AES-256-GCM"
    key_length: int = 32  # 256 bits
    use_salt: bool = True
    salt_length: int = 16
    iterations: int = 100000  # PBKDF2 iterations


class DataEncryption:
    """Secure data encryption and decryption utilities."""
    
    def __init__(self, config: EncryptionConfig = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for encryption")
            
        self.config = config or EncryptionConfig()
        
    def generate_key(self, password: Optional[str] = None, 
                    salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password or random."""
        if password:
            # Derive key from password using PBKDF2
            if salt is None:
                salt = os.urandom(self.config.salt_length)
                
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                iterations=self.config.iterations,
            )
            key = kdf.derive(password.encode())
            return key, salt
        else:
            # Generate random key
            return os.urandom(self.config.key_length)
            
    def encrypt_data(self, data: Union[str, bytes], key: bytes) -> Dict[str, Any]:
        """Encrypt data using AES-GCM."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            # Generate random IV
            iv = os.urandom(12)  # 96-bit IV for GCM
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv)
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Return encrypted data with metadata
            return {
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
                "algorithm": self.config.algorithm
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
            
    def decrypt_data(self, encrypted_data: Dict[str, Any], key: bytes) -> bytes:
        """Decrypt data using AES-GCM."""
        try:
            # Extract components
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            iv = base64.b64decode(encrypted_data["iv"])
            tag = base64.b64decode(encrypted_data["tag"])
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
            
    def encrypt_file(self, file_path: Union[str, Path], 
                    output_path: Union[str, Path], key: bytes) -> None:
        """Encrypt a file."""
        file_path = Path(file_path)
        output_path = Path(output_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Read file data
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Encrypt data
            encrypted_data = self.encrypt_data(data, key)
            
            # Write encrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(encrypted_data, f)
                
            logger.info(f"File encrypted: {file_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
            
    def decrypt_file(self, encrypted_file_path: Union[str, Path],
                    output_path: Union[str, Path], key: bytes) -> None:
        """Decrypt a file."""
        encrypted_file_path = Path(encrypted_file_path)
        output_path = Path(output_path)
        
        if not encrypted_file_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")
            
        try:
            # Read encrypted data
            with open(encrypted_file_path, 'r') as f:
                encrypted_data = json.load(f)
                
            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_data, key)
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
                
            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise


class SecureDataManager:
    """Secure data management with encryption and access control."""
    
    def __init__(self, storage_dir: str = "secure_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.encryption = DataEncryption()
        self.keys = {}  # In production, use proper key management
        
    def store_data(self, data_id: str, data: Union[str, bytes, Dict], 
                  password: Optional[str] = None) -> str:
        """Store data securely with encryption."""
        try:
            # Generate or derive key
            if password:
                key, salt = self.encryption.generate_key(password)
                key_info = {
                    "type": "password_derived",
                    "salt": base64.b64encode(salt).decode('utf-8')
                }
            else:
                key = self.encryption.generate_key()
                key_info = {"type": "random"}
                
            # Store key securely (in production, use proper key management)
            self.keys[data_id] = {
                "key": base64.b64encode(key).decode('utf-8'),
                "info": key_info
            }
            
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
                
            # Encrypt data
            encrypted_data = self.encryption.encrypt_data(data_bytes, key)
            
            # Store encrypted data
            data_file = self.storage_dir / f"{data_id}.enc"
            with open(data_file, 'w') as f:
                json.dump(encrypted_data, f)
                
            logger.info(f"Data stored securely: {data_id}")
            return data_id
            
        except Exception as e:
            logger.error(f"Failed to store data {data_id}: {e}")
            raise
            
    def retrieve_data(self, data_id: str, password: Optional[str] = None) -> bytes:
        """Retrieve and decrypt stored data."""
        try:
            # Check if data exists
            data_file = self.storage_dir / f"{data_id}.enc"
            if not data_file.exists():
                raise FileNotFoundError(f"Data not found: {data_id}")
                
            if data_id not in self.keys:
                raise ValueError(f"No key found for data: {data_id}")
                
            # Get key
            key_data = self.keys[data_id]
            if key_data["info"]["type"] == "password_derived":
                if not password:
                    raise ValueError("Password required for password-derived key")
                    
                salt = base64.b64decode(key_data["info"]["salt"])
                key, _ = self.encryption.generate_key(password, salt)
            else:
                key = base64.b64decode(key_data["key"])
                
            # Load encrypted data
            with open(data_file, 'r') as f:
                encrypted_data = json.load(f)
                
            # Decrypt data
            decrypted_data = self.encryption.decrypt_data(encrypted_data, key)
            
            logger.info(f"Data retrieved: {data_id}")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {data_id}: {e}")
            raise
            
    def delete_data(self, data_id: str) -> bool:
        """Securely delete stored data."""
        try:
            data_file = self.storage_dir / f"{data_id}.enc"
            
            if data_file.exists():
                # Secure deletion by overwriting
                file_size = data_file.stat().st_size
                with open(data_file, 'r+b') as f:
                    # Overwrite with random data
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                    
                # Remove file
                data_file.unlink()
                
            # Remove key
            if data_id in self.keys:
                del self.keys[data_id]
                
            logger.info(f"Data securely deleted: {data_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data {data_id}: {e}")
            return False
            
    def list_stored_data(self) -> List[str]:
        """List all stored data IDs."""
        data_files = list(self.storage_dir.glob("*.enc"))
        return [f.stem for f in data_files]


# Utility functions
def generate_encryption_key() -> str:
    """Generate a random encryption key."""
    if not CRYPTOGRAPHY_AVAILABLE:
        raise ImportError("cryptography library is required")
        
    key = Fernet.generate_key()
    return key.decode('utf-8')


def encrypt_data(data: Union[str, bytes], password: str) -> str:
    """Encrypt data with password (simple interface)."""
    encryption = DataEncryption()
    key, salt = encryption.generate_key(password)
    
    encrypted = encryption.encrypt_data(data, key)
    encrypted["salt"] = base64.b64encode(salt).decode('utf-8')
    
    return base64.b64encode(json.dumps(encrypted).encode()).decode('utf-8')


def decrypt_data(encrypted_data: str, password: str) -> bytes:
    """Decrypt data with password (simple interface)."""
    encryption = DataEncryption()
    
    # Decode and parse
    encrypted_dict = json.loads(base64.b64decode(encrypted_data))
    salt = base64.b64decode(encrypted_dict.pop("salt"))
    
    # Derive key
    key, _ = encryption.generate_key(password, salt)
    
    # Decrypt
    return encryption.decrypt_data(encrypted_dict, key)


def hash_data(data: Union[str, bytes], salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Hash data with salt using SHA-256."""
    if isinstance(data, str):
        data = data.encode('utf-8')
        
    if salt is None:
        salt = os.urandom(16)
        
    hash_obj = hashlib.sha256(salt + data)
    digest = hash_obj.hexdigest()
    salt_b64 = base64.b64encode(salt).decode('utf-8')
    
    return digest, salt_b64


def secure_compare(a: str, b: str) -> bool:
    """Secure string comparison to prevent timing attacks."""
    return secrets.compare_digest(a, b)