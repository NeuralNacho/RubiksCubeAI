import numpy as np

class Cube:
    def __init__(self, n):
        self.dim = n
        face = np.ones((n,n), dtype = int)
        # viewing each face so that unfolds into hotizontal t shape like so: -|--
        self.up_face = np.multiply(face, 0)
        self.down_face = np.multiply(face, 1)
        self.front_face = np.multiply(face, 2)
        self.back_face = np.multiply(face, 3) 
        self.right_face = np.multiply(face, 4)
        self.left_face = np.multiply(face, 5)

    def clockwise_right(self):
        self.right_face = np.rot90(self.right_face, -1)
        temp = self.up_face[:,-1].copy()  # gives last column
        self.up_face[:,-1] = self.front_face[:,-1]
        self.front_face[:,-1] = self.down_face[:,0][::-1]  # [::-1] reverses array
        self.down_face[:,0] = self.back_face[:,-1][::-1]
        self.back_face[:,-1] = temp

    def clockwise_left(self):
        self.left_face = np.rot90(self.left_face, -1)
        temp = self.up_face[:,0].copy()
        self.up_face[:,0] = self.back_face[:,0]
        self.back_face[:,0] = self.down_face[:,-1][::-1]
        self.down_face[:,-1] = self.front_face[:,0][::-1]
        self.front_face[:,0] = temp

    def clockwise_front(self):
        self.front_face = np.rot90(self.front_face, -1)
        temp = self.up_face[-1].copy()
        self.up_face[-1] = self.left_face[-1]
        self.left_face[-1] = self.down_face[-1]
        self.down_face[-1] = self.right_face[-1]
        self.right_face[-1] = temp

    def clockwise_back(self):
        self.back_face = np.rot90(self.back_face, -1)
        temp = self.up_face[0].copy()
        self.up_face[0] = self.right_face[0]
        self.right_face[0] = self.down_face[0]
        self.down_face[0] = self.left_face[0]
        self.left_face[0] = temp

    def clockwise_up(self):
        self.up_face = np.rot90(self.up_face, -1)
        temp = self.back_face[-1].copy()
        self.back_face[-1] = self.left_face[:,-1][::-1]
        self.left_face[:,-1] = self.front_face[0]
        self.front_face[0] = self.right_face[:,0][::-1]
        self.right_face[:,0] = temp

    def clockwise_down(self):
        self.down_face = np.rot90(self.down_face, -1)  # checked in my head that this is correct
        temp = self.back_face[0].copy()
        self.back_face[0] = self.right_face[:,-1]
        self.right_face[:,-1] = self.front_face[-1][::-1]
        self.front_face[-1] = self.left_face[:,0]
        self.left_face[:,0] = temp[::-1]
    
    def anticlockwise_right(self):
        self.right_face = np.rot90(self.right_face)
        temp = self.up_face[:,-1].copy()  # gives last column
        self.up_face[:,-1] = self.back_face[:,-1]
        self.back_face[:,-1] = self.down_face[:,0][::-1]
        self.down_face[:,0] = self.front_face[:,-1][::-1]
        self.front_face[:,-1] = temp

    def anticlockwise_left(self):
        self.left_face = np.rot90(self.left_face)
        temp = self.up_face[:,0].copy()
        self.up_face[:,0] = self.front_face[:,0]
        self.front_face[:,0] = self.down_face[:,-1][::-1]
        self.down_face[:,-1] = self.back_face[:,0][::-1]
        self.back_face[:,0] = temp

    def anticlockwise_front(self):
        self.front_face = np.rot90(self.front_face)
        temp = self.up_face[-1].copy()
        self.up_face[-1] = self.right_face[-1]
        self.right_face[-1] = self.down_face[-1]
        self.down_face[-1] = self.left_face[-1]
        self.left_face[-1] = temp

    def anticlockwise_back(self):
        self.back_face = np.rot90(self.back_face)
        temp = self.up_face[0].copy()
        self.up_face[0] = self.left_face[0]
        self.left_face[0] = self.down_face[0]
        self.down_face[0] = self.right_face[0]
        self.right_face[0] = temp

    def anticlockwise_up(self):
        self.up_face = np.rot90(self.up_face, 1)
        temp = self.back_face[-1].copy()
        self.back_face[-1] = self.right_face[:,0]
        self.right_face[:,0] = self.front_face[0][::-1]
        self.front_face[0] = self.left_face[:,-1]
        self.left_face[:,-1] = temp[::-1]

    def anticlockwise_down(self):
        self.down_face = np.rot90(self.down_face)
        temp = self.back_face[0].copy()
        self.back_face[0] = self.left_face[:,0][::-1]
        self.left_face[:,0] = self.front_face[-1]
        self.front_face[-1] = self.right_face[:,-1][::-1]
        self.right_face[:,-1] = temp

# github test2