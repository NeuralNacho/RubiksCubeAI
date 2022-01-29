from tkinter import *
from Cube import *

class RenderEnv(Tk):
    def __init__(self, cube):
        super().__init__()
        self.cube = cube
        self.title("Rubik's Cube")
        self.geometry('500x400+700+100')
        self.resizable(True, True)
        self.canvas = Canvas(self, bg = '#191919', height = self.winfo_height(), width = self.winfo_width())
        self.face_size = min(self.winfo_width() / 6, self.winfo_height() / 5)
        self.draw_cube()
        self.canvas.pack(fill = 'both', expand = True)
        self.bind('<Configure>', lambda event: self.on_resize())

    def draw_cube(self):
        self.draw_face(self.cube.left_face, [self.face_size, 2 * self.face_size])
        self.draw_face(self.cube.back_face, [2 * self.face_size, self.face_size])
        self.draw_face(self.cube.up_face, [2 * self.face_size, 2 * self.face_size])
        self.draw_face(self.cube.front_face, [2 * self.face_size, 3 * self.face_size])
        self.draw_face(self.cube.right_face, [3 * self.face_size, 2 * self.face_size])
        self.draw_face(self.cube.down_face, [4 * self.face_size, 2 * self.face_size])

    def draw_face(self, face, coords):  # face will be a numpy array of integers. coords will start in top left corner
        colour_dict = {0: "#FFFF00", 1: "#FFFFFF", 2: "#00FF00", 
                    3: "#0000FF", 4: "#FFA500", 5: "#FF0000"}
        for i in range(self.cube.dim):
            for j in range(self.cube.dim):
                x = coords[0] + i * (self.face_size / self.cube.dim)
                y = coords[1] + j * (self.face_size / self.cube.dim)
                self.canvas.create_rectangle(x,  y, x + (self.face_size / self.cube.dim), y + (self.face_size / self.cube.dim), fill = colour_dict[face[j][i]])               

    def on_resize(self):
        self.face_size = min(self.winfo_width() / 6, self.winfo_height() / 5)
        self.canvas.delete('all')
        self.draw_cube()
