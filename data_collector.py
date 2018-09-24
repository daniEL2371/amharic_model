from tkinter import *

def onSkip():
    pass

def onNext():
    pass

def onPrev():
    pass

window = Tk()

frame = Frame(window)
frame.pack()
row = 1
lblTitle = Label(frame, text="Amharic Word Feature Collector")
lblTitle.grid(row=row, column=1)
lblWord = Label(frame, text="Word")
lblWordIndex = Label(frame, text="4/100")
row += 1
lblWord.grid(row=row, column=1)
lblWordIndex.grid(row=row, column=2)

row += 1
btnNext = Button(frame, text="Next")
btnNext.grid(row=row, column=3)
btnPrev = Button(frame, text="Previous")
btnPrev.grid(row=row, column=1)
btnSkip = Button(frame, text="Skip")
btnSkip.grid(row=row, column=2)



# lblWordIndex.pack()
# lblTitle.pack()
# lblWord.pack()
# btnNext.pack()
# btnPrev.pack()
# btnSkip.pack()

window.mainloop()