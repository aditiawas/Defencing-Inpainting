from tkinter import *
import easygui
import os
import shutil
import phase1
import phase2
from skimage.io import imread, imsave
from inpainter import Inpainter

def runTask(picture):
  r2.destroy()
  img = phase1.readimg(picture)
  phase1.processbpf(img)
  phase2.processmask(picture)
  print("Defencing process complete")

  dir = os.path.dirname(__file__)
  foldername = os.path.join(dir, 'Images')
  os.chdir(foldername)
  image = imread('applyon.jpg')
  mask = imread('finalmask.jpg', as_gray=True)

  output_image = Inpainter(image,mask).inpaint()
  imsave('result.jpg', output_image, quality=100)

class Tapp(Frame):
  def __init__(self,master):
    super(Tapp,self).__init__(master)
    self.pack()
    self.wid()
    self.red = ""
    self.tar = ""

  def wid(self):
    self.id=StringVar()
    self.id.set(None)

    Button(self,text="Upload image", command= self.add_data).pack(padx=30,pady=30,side=TOP)

  def add_data(self):
    picture=easygui.fileopenbox(msg="Select picture")
    runTask(picture)

r2=Tk()
r2.title("Defencing")
r2.geometry("500x200")
a=Tapp(r2)
r2.mainloop()