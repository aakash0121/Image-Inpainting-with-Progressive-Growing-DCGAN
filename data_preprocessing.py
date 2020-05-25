from mtcnn.mtcnn import MTCNN
import mtcnn
from os import listdir
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
from matplotlib import pyplot

# print version
# print(mtcnn.__version__)

# load an image as an rgb numpy array
def load_image(filename):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	img = asarray(image)
	return img

def extract_face(model, img, required_size=(128, 128)):
    # detect face in the image
    faces = model.detect_faces(img)

    # skipping cases where a face could not be found
    if len(faces) == 0:
        return None
    
    # extract details of the face
    x1, y1, w, h = faces[0]['box']

    x1, y1 = abs(x1), abs(y1)

    # making cordinates
    x2, y2 = x1 + w, y1 + h

    # retrive face region
    face_region = img[y1:y2, x1:x2]

    image = Image.fromarray(face_region)

    # converting image to required size
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory, num_faces):
    # prepare Model
    model = MTCNN()

    # intializing list of faces
    faces = list()

    # enumerating images
    for filename in listdir(directory):
        # load the image
        img = load_image(directory + filename)

        # get face
        face = extract_face(model, img)

        if face is None:
            continue
        
        # store into faces
        faces.append(face)
        print(len(faces), face.shape)

        # stop once num_faces condition satisfies
        if len(faces) >= num_faces:
            break
        
    return asarray(faces)

# directory that contains all images
directory = 'img_align_celeba/'

# load and extract all faces
all_faces = load_faces(directory, 40000)
print('Loaded: ', all_faces.shape)

# save in compressed format
savez_compressed('img_align_celeba_128.npz', all_faces)