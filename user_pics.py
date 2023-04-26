import pickle


class UserPics(object):

    def __init__(self):
        with open("db/known_face_images", "rb") as f:
            self.known_face_images = pickle.load(f)

    def get_all_pics(self, user_name):
        return self.known_face_images[user_name]
