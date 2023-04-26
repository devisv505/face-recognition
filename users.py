import pickle


class Users(object):

    def __init__(self):
        with open("db/known_face_names", "rb") as f:
            self.known_face_names = pickle.load(f)

    def get_all(self):
        return self.known_face_names
