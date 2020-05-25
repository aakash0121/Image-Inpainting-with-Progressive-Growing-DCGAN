from numpy import load
import matplotlib.pyplot as plt 

# plot a list of loaded faces
def plot_faces(faces, n=10):
    for i in range(n*n):
        # define subplot
        plt.subplot(n, n, i+1)

        plt.axis("off")

        # show raw pixel
        plt.imshow(faces[i].astype('uint8'))

    plt.show()

# load the modified face dataset
data = load('img_align_celeba_128.npz')
faces = data['arr_0']
print('Loaded: ', faces.shape)
plot_faces(faces, 10)