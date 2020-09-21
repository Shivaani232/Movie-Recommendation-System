from tkinter import*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
import back as back

window=Tk()
window.geometry("300x300")
window.title('Movie Recommender')
window.configure(bg="purple")

name=StringVar()

movie_name=Label(window,text="      Movie Name      ",
                    fg="white",
                    bg="blue")
movie_name.pack(side=LEFT)
name_input=Entry(window,textvariable=name)
name_input.pack(side=RIGHT)
movie_names=Button(window,text="  Submit  ",
                   fg="white",
                   bg="black",
                   command=lambda :back.make_recommendation(name.get()))
movie_names.pack(side=BOTTOM)

window.mainloop()
