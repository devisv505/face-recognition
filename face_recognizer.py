import os
import pickle
from os import walk

import face_recognition
import numpy as np


class Recognizer(object):

    def __init__(self):
        self.filenames = os.listdir("static/images")
        # self.filenames = next(walk("static/images"), (None, None, []))[2]
        self.filenames.remove(".DS_Store")

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = {}
        self.next_user_number = {'user_': 1}

    def upload(self):
        img_count = 0
        img_cnt = len(self.filenames)

        for filename in self.filenames:
            fn = "./static/images/" + filename
            img_count = img_count + 1
            image_data = face_recognition.load_image_file(fn)
            face_locations = face_recognition.face_locations(image_data)
            face_encodings = face_recognition.face_encodings(image_data, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        self.known_face_images[name].append(fn)
                    else:
                        name = self.assign_name(name)
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        self.known_face_images[name] = [fn]
                else:
                    name = self.assign_name(name)
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    self.known_face_images[name] = [fn]

            print("Loading " + str(img_count) + "/" + str(img_cnt))

        with open("db/known_face_encodings", "wb") as fp:
            pickle.dump(self.known_face_encodings, fp)

        with open("db/known_face_names", "wb") as fp:
            pickle.dump(self.known_face_names, fp)

        with open("db/known_face_images", "wb") as fp:
            pickle.dump(self.known_face_images, fp)

    def assign_name(self, name):
        name = 'user_' + str(self.next_user_number['user_'])
        self.next_user_number['user_'] += 1
        return name

# import pickle
# from os import walk
#
# import face_recognition
# import numpy as np
#
#
# class Recognizer(object):
#
#     def __init__(self):
#         self.filenames = next(walk("static/images"), (None, None, []))[2]
#         self.filenames.remove(".DS_Store")
#
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_face_images = {}
#         print(self.filenames)
#
#     def upload(self):
#         count = 1
#         img_count = 0
#         img_cnt = len(self.filenames)
#
#         for filename in self.filenames:
#             fn = "./static/images/" + filename
#
#             img_count = img_count + 1
#             image_data = face_recognition.load_image_file(fn)
#             face_locations = face_recognition.face_locations(image_data)
#             face_encodings = face_recognition.face_encodings(image_data, face_locations)
#
#             for face_encoding in face_encodings:
#                 matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                 name = "Unknown"
#
#                 if len(self.known_face_encodings) > 0:
#                     # Or instead, use the known face with the smallest distance to the new face
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = self.known_face_names[best_match_index]
#                         self.known_face_images[name].append(fn)
#                     else:
#                         name = "user_" + str(count)
#                         count = count + 1
#                         self.known_face_encodings.append(face_encoding)
#                         self.known_face_names.append(name)
#                         self.known_face_images[name] = [fn]
#                 else:
#                     name = "user_" + str(count)
#                     count = count + 1
#                     self.known_face_encodings.append(face_encoding)
#                     self.known_face_names.append(name)
#                     self.known_face_images[name] = [fn]
#
#             print("Loading " + str(img_count) + "/" + str(img_cnt))
#
#         with open("db/known_face_encodings", "wb") as fp:
#             pickle.dump(self.known_face_encodings, fp)
#
#         with open("db/known_face_names", "wb") as fp:
#             pickle.dump(self.known_face_names, fp)
#
#         with open("db/known_face_images", "wb") as fp:
#             pickle.dump(self.known_face_images, fp)


# import pickle
# from os import walk
#
# import face_recognition
# import numpy as np
#
#
# class Recognizer(object):
#
#     def __init__(self):
#         self.filenames = next(walk("static/images"), (None, None, []))[2]
#         self.filenames.remove(".DS_Store")
#
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_face_images = {}
#         print(self.filenames)
#
#     def upload(self):
#         count = 1
#         img_count = 0
#         img_cnt = len(self.filenames)
#
#         for filename in self.filenames:
#             fn = "./static/images/" + filename
#
#             img_count = img_count + 1
#             image_data = face_recognition.load_image_file(fn)
#             face_locations = face_recognition.face_locations(image_data)
#             face_encodings = face_recognition.face_encodings(image_data, face_locations)
#
#             for face_encoding in face_encodings:
#                 matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                 name = "Unknown"
#
#                 if len(self.known_face_encodings) > 0:
#                     # Or instead, use the known face with the smallest distance to the new face
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = self.known_face_names[best_match_index]
#                         self.known_face_images[name].append(fn)
#                     else:
#                         name = "user_" + str(count)
#                         count = count + 1
#                         self.known_face_encodings.append(face_encoding)
#                         self.known_face_names.append(name)
#                         self.known_face_images[name] = [fn]
#                 else:
#                     name = "user_" + str(count)
#                     count = count + 1
#                     self.known_face_encodings.append(face_encoding)
#                     self.known_face_names.append(name)
#                     self.known_face_images[name] = [fn]
#
#             print("Loading " + str(img_count) + "/" + str(img_cnt))
#
#         with open("db/known_face_encodings", "wb") as fp:
#             pickle.dump(self.known_face_encodings, fp)
#
#         with open("db/known_face_names", "wb") as fp:
#             pickle.dump(self.known_face_names, fp)
#
#         with open("db/known_face_images", "wb") as fp:
#             pickle.dump(self.known_face_images, fp)


# import pickle
# from os import walk
#
# import face_recognition
# import numpy as np
#
#
# class Recognizer(object):
#
#     def __init__(self):
#         self.filenames = next(walk("static/images"), (None, None, []))[2]
#         self.filenames.remove(".DS_Store")
#
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_face_images = {}
#         print(self.filenames)
#
#     def upload(self):
#         count = 1
#         img_count = 0
#         img_cnt = len(self.filenames)
#
#         for filename in self.filenames:
#             fn = "./static/images/" + filename
#
#             img_count = img_count + 1
#             image_data = face_recognition.load_image_file(fn)
#             face_locations = face_recognition.face_locations(image_data)
#             face_encodings = face_recognition.face_encodings(image_data, face_locations)
#
#             for face_encoding in face_encodings:
#                 matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                 name = "Unknown"
#
#                 if len(self.known_face_encodings) > 0:
#                     # Or instead, use the known face with the smallest distance to the new face
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = self.known_face_names[best_match_index]
#                         self.known_face_images[name].append(fn)
#                     else:
#                         name = "user_" + str(count)
#                         count = count + 1
#                         self.known_face_encodings.append(face_encoding)
#                         self.known_face_names.append(name)
#                         self.known_face_images[name] = [fn]
#                 else:
#                     name = "user_" + str(count)
#                     count = count + 1
#                     self.known_face_encodings.append(face_encoding)
#                     self.known_face_names.append(name)
#                     self.known_face_images[name] = [fn]
#
#             print("Loading " + str(img_count) + "/" + str(img_cnt))
#
#         with open("db/known_face_encodings", "wb") as fp:
#             pickle.dump(self.known_face_encodings, fp)
#
#         with open("db/known_face_names", "wb") as fp:
#             pickle.dump(self.known_face_names, fp)
#
#         with open("db/known_face_images", "wb") as fp:
#             pickle.dump(self.known_face_images, fp)
