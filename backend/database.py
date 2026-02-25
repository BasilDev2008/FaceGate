from sqlalchemy import create_engine, Column, String, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker # taking to the database
from datetime import datetime 
import numpy as np
Base = declarative_base()

class User(Base):
    __tablename__ = 'users' # name of the table
    id = Column(String, primary_key = True) # creates unique ID for each user
    name = Column(String, nullable = False) # full name of user
    role = Column(String, nullable = False)
    face_embedding = Column(LargeBinary, nullable = False)
    voice_embedding = Column(LargeBinary, nullable = False)
    created_at = Column(DateTime, default = datetime.utcnow) # time of registery
class AttendanceRecord(Base):
    __tablename__ = 'attendance'
    id = Column(String, primary_key = True)
    user_id = Column(String, nullable = False)
    timestamp = Column(DateTime, default = datetime.utcnow)
    status = Column(String, nullable = False) # late or present
    confidence_face = Column(String, nullable = False)
    confidence_voice = Column(String, nullable = False)
class Database:
    def __init__(self):
        self.engine = create_engine('sqlite:///facegate.db') # create database file
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind = self.engine)
        self.session = Session()
    def add_unser(self, id, name, role, face_embedding, voice_embedding):
        face_bytes = face_embedding.numpy().tobytes()
        voice_bytes = voice_embedding.numpy().tobytes()
        user = User(
            id = id,
            name = name,
            role = role,
            face_embedding = face_bytes,
            voice_embedding = voice_bytes,
        )
        self.session.add(user)
        self.session.commit()
    def get_all_users(self):
        return self.session.query(User).all()
    def get_embedding(self, user):
        face_embedding = np.frombuffer(user.face_embedding, dtype = np.float32)
        voice_embedding = np.frombuffer(user.voice_embedding, dtype = np.float32)
        return face_embedding, voice_embedding
    def log_attendance(self, user_id, name, status, confidence_face, confidence_voice):
        import uuid
        record = AttendanceRecord(
            id = str(uuid.uuid4()),
            user_id = user_id,
            name = name,
            status = status,
            confidence_face = str(confidence_face),
            confidence_voice = str(confidence_voice)
        )
        self.session.add(record)
        self.session.commit()
    def get_attendance(self):
        return self.session.query(AttendanceRecord).all()
    