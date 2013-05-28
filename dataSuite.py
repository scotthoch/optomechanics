import numpy as np


def preBinFFT(real, imag, tstp, binsize, bintype='mean'):
    data=bindat(real,binsize,bintype)+1j*bindat(imag,binsize,bintype)
    freqs=FixFftFreq(np.fft.fftfreq(data.size,tstp*binsize))
    cplxfft=FixFftFreq(np.fft.fft(data))
    return [freqs,cplxfft]

def postBinFFT(real, imag, tstp, binsize, bintype='mean'):
    data=real+1j*imag
    freqs=FixFftFreq(np.fft.fftfreq(data.size,tstp))
    cplxfft=FixFftFreq(np.fft.fft(data))
    freqs=bindat(freqs,binsize,'mean')
    cplxfft=(bindat(np.real(cplxfft),binsize,bintype)+
                    1j*bindat(np.imag(cplxfft),binsize,bintype))
    return [freqs,cplxfft]

def noBinFFT(real, imag, tstp):
    data=real+1j*imag
    freqs=FixFftFreq(np.fft.fftfreq(data.size,tstp))
    cplxfft=FixFftFreq(np.fft.fft(data))
    return [freqs,cplxfft]

def FixFftFreq(fftarray):
    '''This will cut fftarray in half and switch the order of the halvs.
    It is mostly intended for use by other functions in the Fourier class
    but why not make it accessable'''
    holder=[t for t in np.split(np.array(fftarray),2)[1]]
    holder.extend([t for t in np.split(np.array(fftarray),2)[0]])
    return holder

def bindat(data,binsize=50, type="mean"):
    '''a simple low pass filter that takes nearest neighboring points and maps them to
    a single point.  The value of that point is the "mean" "max" or "min" of that set'''
    numBins=np.array(data).size//binsize
    dataChunks=np.array(data[:binsize*numBins]).reshape((-1,binsize))
    if type == "mean":
        binnedDat=dataChunks.mean(axis=1)
    elif type == "max":
        binnedDat=dataChunks.max(axis=1)
    elif type == "min":
	    binnedDat=dataChunks.min(axis=1)
    else:
	    print "Please enter a valid binning type: mean, min, or max"
	    return
    return binnedDat

def undoHF2Filter(freq,mag,pole,tau):
        undone=mag*(np.abs(((1+(2*3.14159*1j*np.abs(freq)*tau)))**pole)**2) #Filter in HF2 Labview code
        return undone
