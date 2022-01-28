import numpy as np
from tkinter import *
import Cube

class Window(Tk):
    def __init__(self):
        super().__init__()
        self.cube = Cube.Cube(2)
        self.title("Rubik's Cube")
        self.geometry('500x400+700+100')
        self.resizable(True, True)
        self.canvas = Canvas(self, bg = '#191919', height = self.winfo_height(), width = self.winfo_width())
        self.face_size = min(self.winfo_width() / 6, self.winfo_height() / 5)
        self.draw_cube()
        self.canvas.pack(fill = 'both', expand = True)  # these arguments make the canvas take up extra space when resizing the self

        self.bind('u', lambda event: self.update_cube(self.cube.clockwise_up))  # lambda since passing function passes the return value not the actual function
        self.bind('d', lambda event: self.update_cube(self.cube.clockwise_down))
        self.bind('l', lambda event: self.update_cube(self.cube.clockwise_left))
        self.bind('r', lambda event: self.update_cube(self.cube.clockwise_right))
        self.bind('f', lambda event: self.update_cube(self.cube.clockwise_front))
        self.bind('b', lambda event: self.update_cube(self.cube.clockwise_back))
        self.bind('U', lambda event: self.update_cube(self.cube.anticlockwise_up))  # works with shift key
        self.bind('D', lambda event: self.update_cube(self.cube.anticlockwise_down))
        self.bind('L', lambda event: self.update_cube(self.cube.anticlockwise_left))
        self.bind('R', lambda event: self.update_cube(self.cube.anticlockwise_right))
        self.bind('F', lambda event: self.update_cube(self.cube.anticlockwise_front))
        self.bind('B', lambda event: self.update_cube(self.cube.anticlockwise_back))
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
    
    def update_cube(self, rotation):
        rotation()
        self.draw_cube()

    def on_resize(self):
        self.face_size = min(self.winfo_width() / 6, self.winfo_height() / 5)
        self.canvas.delete('all')
        self.draw_cube()

if __name__ == '__main__':
    window = Window()
    window.mainloop() # watches for events on tkinter objects

# github comment