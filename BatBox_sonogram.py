#Code to generate a sonogram from a wav file recorded by frequency division (/10).
#Credit: Nicolas Oudart
#Version: 1 - 28/11/2021

#Libraries:
from tkinter import filedialog as fd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert

#Size of the window used by the short-time Fourier transform (s):
wint = 1e-3

#Import the wav file:
filename = fd.askopenfilename()

#Fs = Sampling frequency
#BatSound = Data from the wav file
Fs, BatSound = wavfile.read(filename)

#dt = time step corresponding to Fs
#T = time period corresponding to the entire file duration
dt = 1/Fs
T = len(BatSound)*dt

#Size of the window in samples:
wint = round((wint/T)*len(BatSound))
#Frequency bandwidth corresponding to the time delay dt:
F = 1/(dt*2)

#Initialize the sonogram:
BatSono = np.zeros((len(range(wint,len(BatSound)-wint,1)),wint*10))

#Initialize the counter of frequency spectra composing the sonogram:
numspec = 0

#Sliding window to perform the "short-time Fourier transform":
for ii in range(wint,len(BatSound)-wint,1):

    #Hilbert transform to obtain a complex signal, select samples in the window:
    BatSample = hilbert(BatSound[ii-wint:ii+wint])
    #Frequency spectrum -> Fast Fourier Transform (FFT) with windowing (Hann), and zero-padding (X10):
    Spec = abs(np.fft.fft(BatSample[::2]*np.hanning(wint),wint*10))
    #Add this spectrum to the sonogram:
    BatSono[numspec][:] = Spec
    #Next spectrum, increment the counter:
    numspec = numspec + 1

#Normalize the sonogram:
BatSono = BatSono/np.max(BatSono)

#Adapt the orientation of the radargram:
BatSono = np.transpose(BatSono)
BatSono = np.flipud(BatSono)

#Calculate the time and frequency steps (dt and df), in ms and kHz:
df = 10*(F/np.shape(BatSono)[0])
df = df/1000
dt = dt*1000

#Display the figure (20-80 kHz):
plt.imshow(BatSono,extent=[0,np.shape(BatSono)[1]*dt,0,np.shape(BatSono)[0]*df],aspect='auto',cmap='jet')
plt.gca().set_ylim(20,80)
plt.title('SQY - 11/08/2021')
plt.xlabel('Time delays (ms)')
plt.ylabel('Frequency (kHz)')
plt.colorbar()
plt.grid(which='both',axis='both')
plt.show()
