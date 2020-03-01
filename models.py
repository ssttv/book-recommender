from app import db

class Element(db.Model):
    element_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(1024))
    tags = db.Column(db.String(1024))

    def convert_to_dict(self):
        out_dict = {'element_id': self.element_id, 'title': self.title, 'tags': self.tags}
        return out_dict
    
    def __repr__(self):
        return 'Element #{} title {}>'.format(self.element_id, self.title)

class Activity(db.Model):
    activity_id = db.Column(db.Integer, primary_key=True)
    element_id = db.Column(db.Integer)
    user_id = db.Column(db.Integer)
    rating = db.Column(db.Integer)
    status = db.Column(db.Integer)

    def convert_to_dict(self):
        out_dict = {'activity_id': self.activity_id, 'element_id': self.element_id, 'user_id': self.user_id,'rating': self.rating, 'status': self.status}
        return out_dict
    
    def __repr__(self):
         return 'Activity #{} on element {} from user {}>'.format(self.activity_id, self.element_id, self.user_id)
