# from app import db
# from extensions import db

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ARRAY

Base = declarative_base()

class Element(Base):
    __tablename__ = 'element'

    element_id = Column(Integer, primary_key=True)
    title = Column(String(1024))
    tags = Column(String(1024))

    def convert_to_dict(self):
        out_dict = {'element_id': self.element_id, 'title': self.title, 'tags': self.tags}
        return out_dict
    
    def __repr__(self):
        return 'Element #{} title {}>'.format(self.element_id, self.title)

class Activity(Base):
    __tablename__ = 'activity'
    
    activity_id = Column(Integer, primary_key=True)
    element_id = Column(Integer)
    user_id = Column(Integer)
    rating = Column(Integer)
    status = Column(Integer)

    def convert_to_dict(self):
        out_dict = {'activity_id': self.activity_id, 'element_id': self.element_id, 'user_id': self.user_id,'rating': self.rating, 'status': self.status}
        return out_dict
    
    def __repr__(self):
         return 'Activity #{} on element {} from user {}>'.format(self.activity_id, self.element_id, self.user_id)
