import matplotlib.pyplot as plt 
import numpy as np
from scipy.io import wavfile
from numpy.fft import fft, ifft
import scipy.signal as sig


class AmAud1:

	wavfn = '' # string <- load_timeseries()
	label = ''
	#--#
	pcm0 = np.zeros(0) # array of int then float <- load_timeseries()
	pcm_offset = -1 # <- load_timeseries() and set_pcm_cut_p()
	pcm_samples = 0 # <- load_timeseries() and set_pcm_cut_p()
	sps = -1 # int <- load_timeseries()
	pcm0ms = np.zeros(0) # array of float <- withComputed_timeseries_ms()
	isScaled_pcm = False
	#--#
	mags = np.zeros(0)
	fft_data = np.zeros(0)
	frqs = np.zeros(0)
	frqsz = 0
	phas = np.zeros(0)
	xmags = np.zeros(0)
	#--#
	peak_frqs = np.zeros(0)
	peak_mags = np.zeros(0)
	peak_phas = np.zeros(0)
	proms_ = np.zeros(0)
	isNormalizedFft = False
	maxnext_peak_dist = -1
	pick_strategy = ''
	fft_data_sharpened = np.zeros(0)
	ifft_data_sharpened = np.zeros(0)
	main_peak_x = -1
	peak_frqs_0 = -1
	
	def __init__(self,label,libdef):
		self.label = label
		self.libdef = libdef
		
	def __repr__(self):

		try:
			get_str_peak_main= "main_peak={} at frq={}".format(self.mags[self.main_peak_x],self.frqs[self.main_peak_x])
		except:
			get_str_peak_main= "main_peak={} at frq={}".format('n/a','n/a')
			
		try:
			get_str_peaks= "peaks={},{},..{} at frqs={},{},..{} - freq0={}".format(
				self.peak_mags[0],self.peak_mags[1],self.peak_mags[-1],
				self.peak_frqs[0],self.peak_frqs[1],self.peak_frqs[-1],
				self.peak_frqs_0)
		except:
			get_str_peaks= "peaks={},{},..{} at frqs={},{},..{} - freq0={}".format('n/a','n/a','n/a', 'n/a','n/a','n/a','n/a')
		
		lines = [
			"="*20,
			"LABEL={}".format(self.label),
			"len(pcm)={} sps={} pcm_win=[{},{}] pcm_winsz={} isScaled_pcm={}".format(
				len(self.pcm0),self.sps,self.pcm_offset,self.pcm_offset+self.pcm_samples,self.pcm_samples, self.isScaled_pcm),
			"pcm_dtype={} pcm_min={} pcm_max={}".format(type(self.pcm0[0]),min(self.pcm0),max(self.pcm0)),
			"len(mags/frqs/phas)={} frq band={}".format(len(self.mags),self.frqsz),
			get_str_peak_main,
			get_str_peaks,
			"pick_strategy={} no. peaks={} maxnext_peak_dist={} ".format(self.pick_strategy, len(self.peak_mags), self.maxnext_peak_dist)
		]
		return "\n".join(lines)
			#len(self.mags),
			#len(self.xmags),

	def __str__(self):
		return self.__repr__()            
	
	def help(self):
		print('*'*40)
		print('help')
		print('.util_plot_pcm_window()')
		print('.util_plot_spectrum_freqband(0,44100/4,sharpened=False,normalized=False)')
		print(".util_print_equab('a',3,False,False) <- (note,nOctave,bSharp,bFlat)")
		print('*'*40)
		
		
	def load_wav_file(self,wavfn):
		'''using: load_timeseries()'''
		if self.sps > 0: return self
		assert type(wavfn)==str
		print('name:',wavfn)
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
		fs, snd = wavfile.read(wavfn)
		print('freq',fs,'snd class',type(snd),'ndim',snd.ndim,'shape',snd.shape)
		if snd.ndim == 1:
			print('MONO','min',min(snd),'max',max(snd))
			self.load_timeseries(wavfn,snd,fs)
		else:
			ch2take=0
			for ch in range(snd.ndim):
				print('channel',ch,'min',min(snd[:,ch]),'max',max(snd[:,ch]))
			self.load_timeseries(wavfn,snd[:,ch2take],fs)
		return self

	def load_timeseries(self,name,ts,sps):
		'''using: withComputed_timeseries_ms()'''
		self.wavfn = name # 'load_timeseries()'
		#assert sps == 44100
		
		self.sps = sps
		self.pcm0 = ts
		print('len(pcm0)',len(self.pcm0),self.pcm0[0])
		#default
		self.pcm_offset = 0
		self.pcm_samples = len(ts)
		
		self.withComputed_timeseries_ms()
		return self

	def withComputed_timeseries_ms(self):
		cache = len(self.pcm0ms)>0
		if cache: return self
		time = np.arange(0, len(self.pcm0), 1)
		time = (time / self.sps) * 1000
		self.pcm0ms = time
		print([ 'time minmax [ms]', min(time), max(time) ])
		return self
	
	def minmaxscaled(self):
		cache = self.isScaled_pcm 
		if cache: return self
		dtype = type(self.pcm0[0])
		absmax = max(abs(self.pcm0))
		print('original dtype:',dtype,'range:',min(self.pcm0),max(self.pcm0),'absmax',absmax)

		if dtype == np.int16:
			self.pcm0 = self.pcm0 / absmax  # (2.**15)
			self.isScaled_pcm = True
			return self
		if dtype == np.int32:
			self.pcm0 = self.pcm0 / absmax # (2.**31)
			self.isScaled_pcm = True
			return self
		if dtype == np.float64:
			self.pcm0 = self.pcm0 / absmax
			self.isScaled_pcm = True
			return self
		assert False, 'unsupported dtype'
		# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html



	def util_plot_pcm_window(self,*,subsamplingstep):
		'''using: util_get_pcm2plot()'''
		print(self.label,': plotting PCM','(plt.stem)','subsamplingstep=',subsamplingstep,
			  'duration(s)=',round(self.pcm_samples/self.sps,1))
		print(self.label,': plotting PCM','from',self.pcm_offset,'to',self.pcm_offset+self.pcm_samples)
		print(self.label,': plotting PCM',
			  'from',round(self.pcm0ms[self.pcm_offset],1),'ms',
			  'to',  round(self.pcm0ms[self.pcm_offset+self.pcm_samples-1],1),'ms')
		fig = plt.figure(figsize = (15, 5))
		
		x,y = self.util_get_pcm2plot(subsamplingstep=subsamplingstep)
		plt.stem(x,y,'b', markerfmt=" ", basefmt="-b")
		
		#plt.plot(self.pcm0ms[self.pcm_offset:self.pcm_offset+self.pcm_samples],
		#         self.pcm0[self.pcm_offset:self.pcm_offset+self.pcm_samples], color='b')
		plt.ylabel('Amplitude', fontsize=16)
		plt.xlabel('Time (ms)', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.title(self.label)
		plt.show()
		return

	def set_pcm_cut_s(self,samples_start_s,samples_dur_s):
		'''using: set_pcm_cut_p()'''
		pcm_offset = int(self.sps*samples_start_s)
		pcm_samples = int(self.sps*samples_dur_s)
		self.set_pcm_cut_p(pcm_offset,pcm_samples)
		print('samples_to_take: {} ({}s)'.format(pcm_samples,samples_dur_s))
		print('first_sample_to_take: {} ({}s)'.format(pcm_offset,samples_start_s))
		return self
		
	def set_pcm_cut_p(self,pcm_offset,pcm_samples):
		assert len(self.pcm0) > pcm_offset + pcm_samples # samples2take
		self.pcm_offset = pcm_offset
		self.pcm_samples = pcm_samples
		return self
	
	def util_get_pcm2plot(self,subsamplingstep):
		subsamplingstep=int(subsamplingstep)
		# the purpose of subsamplingstep is just for plotting fewer datapoints and saving memory/notebook size
		x = self.pcm0ms[self.pcm_offset:self.pcm_offset+self.pcm_samples]
		y = self.pcm0[self.pcm_offset:self.pcm_offset+self.pcm_samples]
		x = x[0::subsamplingstep]
		y = y[0::subsamplingstep]
		return (x,y)

	# FFT #
	
	def withSpectrum(self):
		'''using: util_get_pcm2plot()'''
		# caching:
		#if self.mags is not None and self.pcm_offset == pcm_offset and self.pcm_samples == pcm_samples:
		#    return self
		#assert sps == 44100
		#assert type(x_) == np.ndarray and x_.ndim == 1
		#if False: print('get_spectrum_data','len(x):',len(x_)) # 441
		#x_ = self.pcm0[self.pcm_offset:self.pcm_offset + self.pcm_samples]
		x_,y_ = self.util_get_pcm2plot(subsamplingstep=1)
		assert len(x_) == len(y_)
		#self.pcm_offset = pcm_offset
		#self.pcm_samples = pcm_samples
		X_ = fft(y_) # input = 1d Real array, output = 1d Complex array.
		self.fft_data = X_
		assert X_.ndim == 1 and len(X_)==len(x_)
		#print(type(X_[0]))
		assert type(X_[0])== np.complex128 # (97020+0j)
		N = len(X_) # 441
		assert len(X_) == len(x_)
		n_ = np.arange(N) # [0 1 2 3 4 ... 440]
		T = N/self.sps # 0.01
		freq_ = n_/T # [  0. 100. 200. 300. 400. ... 44000.0]
		#freq = 0-sbs divided in samples2take bins -> size(bin) is sps/samples2take
		assert freq_[0]==0
		self.frqsz = freq_[1]
		#if False:
		mag_ = np.abs(X_)
		#xlim_sup = int(sps/2)
		#fft_data = X_
		# https://dsp.stackexchange.com ...
		# /questions/72005/calculate-the-magnitude-and-phase-of-a-signal-at-a-particular-frequency-in-pytho
		#mag_ = np.mag(fft_data)
		phase_ = np.angle(self.fft_data)
		assert len(mag_) == len(X_)

		x_of_first_item_with_val_gt_x = np.argmax(freq_>44100/2)
		print('withSpectrum()', 'max_freq returned by the FFT=',freq_[-1])
		firsthalf = int(len(mag_)/2)
		##
		self.mags = mag_[0:firsthalf]
		self.frqs = freq_[0:firsthalf]
		print('withSpectrum()', 'max_freq after cut=',self.frqs[-1])
		self.phas = phase_[0:firsthalf]
		assert firsthalf*2 == len(mag_) and len(mag_)==len(freq_) and len(mag_)==len(phase_)
		assert firsthalf == len(self.mags) and len(self.mags)==len(self.frqs) and len(self.mags)==len(self.phas)
		
		idx_max = np.argmax(self.mags)
		self.main_peak_x = idx_max # self.frqs[idx_max]
		print('withSpectrum()', 'fft freq bands:',len(self.frqs))
		print('withSpectrum()', 'width:',self.sps/self.pcm_samples,'Hz')
		print('withSpectrum()', 'global peak:',self.frqs[self.main_peak_x],'Hz')
		
		return self
	
		#plt.stem(freq, mag, 'b', markerfmt=" ", basefmt="-b")
		#plt.xlim(0, xlim_sup)
	
	def util_plot_spectrum_freqband(self, plot_hz_inf,plot_hz_sup,*,sharpened,normalized):
		'''using: nothing'''
		if plot_hz_sup <= 0:
			plot_hz_sup = self.sps/2
		print('fft hz inf:',plot_hz_inf)
		print('fft hz sup:',plot_hz_sup)
		#samples_ = self.pcm0
		#self.withSpectrum()
		
		#if 0:
		   # updated_y = samples_[ offset : offset+samples2take ]
		   # print(len(updated_y),samples2take)
		   # assert len(updated_y) == samples2take
			#freq, mag = get_spectrum_data(x_=updated_y,sps=sps) 
		# freq = 0-sbs divided in samples2take bins -> size(bin) is sps/samples2take
		#assert len(freq) == samples2take and len(mag) == samples2take

		plot_hz_inf_i = int(plot_hz_inf * self.pcm_samples / self.sps)
		plot_hz_sup_i = int(plot_hz_sup * self.pcm_samples / self.sps)
		print('plot_hz_inf_i:',plot_hz_inf_i)
		print('plot_hz_sup_i:',plot_hz_sup_i)
		# plot_hz_inf : plot_hz_inf_i = sps : len(freq)
		# plot_hz_inf_i = plot_hz_inf 
		# 0.0 1.0 44099.0
		# 0.0 10.0 44090.0
		print('len:',len(self.frqs),'freq12..n',self.frqs[0],self.frqs[1],self.frqs[-1])
		print('PLOTTING Spectrum...')
		fig = plt.figure(figsize = (15, 5))

		if sharpened:
			#yy = np.log(self.xmags) # neper
			yy = self.xmags
		else:
			#yy = np.log(self.mags) # neper
			yy = self.mags

		if normalized:
			assert False, 'not supported yet'
			xx = self.frqs * 440. / self.frqs[0]
		else:
			xx = self.frqs
			
		devnull=plt.stem(
			xx[plot_hz_inf_i:plot_hz_sup_i], 
			np.log(yy[plot_hz_inf_i:plot_hz_sup_i]), 'b', markerfmt=" ", basefmt="-b")
		#plt.ylabel('Log Magnitude', fontsize=16)
		plt.ylabel('Magnitude (log)', fontsize=16)
		plt.xlabel('Freq (Hz)', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.title(self.label)
		plt.xlim((0,44100/5))
		return # self # (freq, mag)
		
	def util_parsenote(self,note):
		bSharp = "#" in note[1:]
		bFlat = "b" in note[1:]
		note = note.upper()
		if bSharp or bFlat:
			return self.util_print_equab( note[0], int(note[2:]), bSharp,bFlat)
		else:
			return self.util_print_equab( note[0], int(note[1:]), bSharp,bFlat)
		

	def util_print_equab(self,note,nOctave,bSharp,bFlat):
		'''using: nothing'''
		# https://en.wikipedia.org/wiki/Piano_key_frequencies
		
		# facts:
		st = 2**(1/12)
		A4 = 440.0
		C5 = A4 * st**3
		sts = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
		#inv_sts = {v: k for k, v in sts.items()}
		octaves = nOctave - 5
		Cn = C5 * 2**octaves
		
		note_unspecified_therefore_print_the_full_octave = note =="*"
		if note_unspecified_therefore_print_the_full_octave:
			devnull = [ print('{:2d} {:8.2f}'.format(si,Cn * st**si)) for si in range(12+1) ]
			return None
		note = note.upper()
		assert note in sts.keys()
		RET = Cn * st**sts[note]
		if bSharp: return RET * st
		if bFlat: return RET / st
		#print(RET)
		return RET

	def util_get_peaks_above_infrasound(self,peaks_x):
		remove_infrasound = True
		infrasound_thr = 20.0
		if remove_infrasound:
			peaks_x0 = peaks_x[0]
			while self.frqs[peaks_x0] < infrasound_thr:
				print('peaks_x0',peaks_x0,'has infrasound freq:',self.frqs[peaks_x0],'<- REMOVING')
				peaks_x = peaks_x[1:] # remove the first peak
				peaks_x0 = peaks_x[0]
				print('NEW peaks_x0',peaks_x0,'has freq:',self.frqs[peaks_x0])

			assert self.frqs[peaks_x0] >= infrasound_thr
		return peaks_x
	
	
	def util_get_dist_peakx_peaknext(self,peaks_x,peaks_xmax):
		nextpeak = 0
		for p in peaks_x:
			if p > peaks_xmax:
				nextpeak = p
				break
		dist = nextpeak - peaks_xmax
		assert dist>=1
		print('nextpeak after peaks_xmax=',peaks_xmax,'is',nextpeak,'and dist=',dist)
		return dist
		
	def util_get_peaks(self,hyperparams,pick_strategy,distance):
		'''using: sig.find_peaks()'''
		peaks_x0 = -1
		# promin = hyperparams['promin1']
		if pick_strategy == 'singlepass':
			# returns all the peaks above infrasound with a min distance and a min prominence
			peaks_x, proms_ = sig.find_peaks(self.mags, prominence = hyperparams['promin1'],distance=distance) 
				# , distance=self.sps/10000) # 441
			print('len(peaks)',len(peaks_x))
			print('peaks_x',peaks_x)
			print('freqs of peaks_x',self.frqs[peaks_x])
			print('mags of peaks_x',self.mags[peaks_x])
			print('prominences',proms_)
			self.proms_ = proms_
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			return peaks_x

		if pick_strategy == 'lowtrim':
			# returns all the peaks above infrasound with a min distance and a min prominence
			peaks_x, proms_ = sig.find_peaks(self.mags, prominence = hyperparams['promin2'],distance=distance) 
				# , distance=self.sps/10000) # 441
			print('len(peaks)',len(peaks_x))
			print('peaks_x',peaks_x)
			print('freqs of peaks_x',self.frqs[peaks_x])
			print('mags of peaks_x',self.mags[peaks_x])
			print('prominences',proms_)
			self.proms_ = proms_
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			return peaks_x

		if pick_strategy == 'lowtrim2':
			# returns all the peaks above infrasound with a min distance and a min prominence
			print('pass1','doing find_peaks()','prominence =',hyperparams['promin2'],'distance=',distance)
			peaks_x, proms_ = sig.find_peaks(self.mags, prominence = hyperparams['promin2'],distance=distance)
				# , distance=self.sps/10000) # 441
			print('pass1','len(peaks)',len(peaks_x))
			print('pass1','peaks_x',peaks_x)
			print('pass1','freqs of peaks_x',self.frqs[peaks_x])
			print('pass1','mags of peaks_x',self.mags[peaks_x])
			print('pass1','prominences',proms_)
			self.proms_ = proms_
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			
			# pass 2
			#peaks_x, proms_ = sig.find_peaks(self.mags, prominence = hyperparams['promin2'] /2 ,distance=peaks_x[0] * 0.9) 
			print('pass2','doing find_peaks()','prominence =',hyperparams['promin1'],'distance=',peaks_x[0],' * ',hyperparams['safe_dist_factor'],'=',peaks_x[0]*hyperparams['safe_dist_factor'])
			peaks_x, proms_ = sig.find_peaks(self.mags, prominence = hyperparams['promin1'] ,distance=peaks_x[0] * hyperparams['safe_dist_factor']) 
			print('pass2','len(peaks)',len(peaks_x))
			print('pass2','peaks_x',peaks_x)
			print('pass2','freqs of peaks_x',self.frqs[peaks_x])
			print('pass2','mags of peaks_x',self.mags[peaks_x])
			print('pass2','prominences',proms_)
			self.proms_ = proms_
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			
			return peaks_x
		
		if pick_strategy == 'verymax':
			# assume_max_magn_peak_is_the_fundamental (not always true!)
			# returns all the peaks above infrasound with a min distance = max peak and a min prominence
			peaks_x0 = np.argmax(self.mags) # index of the max mags
			safe_dist_factor = hyperparams['safe_dist_factor']
			peaks_x = self.util_get_peaks(hyperparams,'singlepass',safe_dist_factor*peaks_x0)
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			return peaks_x

		if pick_strategy == 'verymaxnextdist':
			# take the max peak PM: main_peak_x
			# take, with a moderate promin, the next peak PMN
			# take, with a moderate promin and the dist 0.9(PMN - PM)
			peaks_x = self.util_get_peaks(hyperparams,'verymax',0)
			peaks_xmax = peaks_x[0]
			peaks_x = self.util_get_peaks(hyperparams,'lowtrim',hyperparams['safe_dist2'])
			self.maxnext_peak_dist = self.util_get_dist_peakx_peaknext(peaks_x,peaks_xmax)
			peaks_x = self.util_get_peaks(hyperparams,'lowtrim',hyperparams['safe_dist_factor'] * self.maxnext_peak_dist)
			peaks_x = self.util_get_peaks_above_infrasound(peaks_x)
			return peaks_x
				
			
		#if pick_strategy == 'veryfirst':
		#    peaks_x0 = peaks_x[0]
		
		assert peaks_x0 >= 0,'invalid pick_strategy value'
			
		#print('self.frqs[peaks_x0]',self.frqs[peaks_x0])

	def util_boxplot_of_peak_freq_dist(self):
		
		series_base = self.peak_frqs[:-1]
		series_shifted = self.peak_frqs[1:]
		assert len(series_base) == len(series_shifted)
		arr_dists = series_shifted - series_base
		print(self.label,arr_dists)
		fig = plt.figure(figsize = (10, 5))
		ax = plt.boxplot(arr_dists,vert = 0)
		plt.title(self.label)
		
	
	
	def with_sharpened_mag(self,hyperparams):
		'''using: util_get_picks()'''
		print(self.label,'with_sharpened_mag()','BEGIN',hyperparams)
		promin1 = hyperparams['promin1'] # self.sps/1000
		dist = hyperparams['safe_dist2']
		#promin2 = hyperparams['promin2'] # self.sps/1000
		strategy = hyperparams['pick_strategy']
		self.pick_strategy = strategy
		# take the max peak PM: main_peak_x
		# take, with a moderate promin, the next peak PMN
		# take, with a moderate promin and the dist 0.9(PMN - PM)
		
		
		peaks_x = self.util_get_peaks(hyperparams,strategy,dist) # THE ONLY CALL
		
		peaks_x0 = peaks_x[0]
		
		#print('^'*20)
		#print('1. max peak',self.main_peak_x)
		#print('2. peaks_x - first pass',peaks_x)
		#print('4. dist',dist)
		#print('5. peaks_x - first pass',peaks_x)
		#print('6. peaks_x0', peaks_x0)
		
		
		
		# OVERRIDING
		
		#peaks_x, _ = sig.find_peaks(self.mags, prominence = promin2)
		#pthr = self.main_peak
		#peak_mags_all = self.mags[peaks_x]
		
		
		
		self.peak_frqs = self.frqs[peaks_x]
		self.peak_mags = self.mags[peaks_x]
		self.peak_phas = self.phas[peaks_x]

		#normalization:
		assert self.peak_frqs[-1] < self.sps/2
		
		self.normalize_fft(peaks_x0)
				
		print('peaks:','len',len(peaks_x))
		print('peaks_x0:',peaks_x0,'peaks_x0 mags:',self.mags[peaks_x0],'peaks_x0 freq:',self.frqs[peaks_x0])
		print('peaks_x 0,1,last',peaks_x[0],peaks_x[1],peaks_x[-1])
		print(self.frqs[peaks_x])
		
		self.xmags = self.util_get_flattened_a(self.mags,peaks_x)
		self.fft_data_sharpened = self.util_get_flattened_a(self.fft_data,peaks_x)
		#fft_data_sharpened2 = np.concatenate((self.fft_data_sharpened, np.flip(self.fft_data_sharpened)))
		self.ifft_data_sharpened = ifft(self.fft_data_sharpened)
		# https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html
		
		return self

	def normalize_fft(self,peaks_x0):
		if self.isNormalizedFft: return self
		self.peak_frqs_0 = self.peak_frqs[0]
		self.peak_frqs = self.peak_frqs * (440.0 / self.frqs[peaks_x0]) # the first one will be 440 Hz
		print(max(self.peak_frqs)) # 346500.0
		self.peak_mags = self.peak_mags / self.mags[peaks_x0]
		#self.peak_phas = self.peak_phas
		self.isNormalizedFft = True
		return self
		

		
	def util_get_flattened_a(self,a,peaks_x):
		ra = np.zeros(len(a))
		np.put(ra, peaks_x, a[peaks_x]) # like: for every index in peaks_x, set a[index] = a[index] 
		return ra
	
	def util_get_synthesized(self,seconds, a440factor):
		'''generate 3 seconds of synthetic signal'''
		peak_frqs = self.peak_frqs
		peak_mags = self.peak_mags
		peak_phas = self.peak_phas
		samplerate = self.sps
		t = np.linspace(0., 1.*seconds, num=seconds*samplerate)
		print('output datapoints:',len(t),'dcheck:',44100*(t[1]-t[0]))
		amplitudex = np.iinfo(np.int16).max

		data = np.zeros(3*samplerate) # 3 seconds -> 3*44k datapoints
		for compon in range(len(peak_frqs)):
			fs = peak_frqs[compon] * a440factor # / self.sps # added:  / self.sps
			ph = peak_phas[compon]
			amplitude = peak_mags[compon] / max(peak_mags) * amplitudex
			print('component {:2d} freq {:8.2f} amplitude {:8.2f} phase {:6.2f}'.format(compon,fs,amplitude,ph))
			
			data_c = amplitude * np.sin(2. * np.pi * fs * t + ph)

			data = data + data_c
		return data
		
	def util_write_a440_timber(self,ofname, seconds, a440factor):
		
		data = self.util_get_synthesized(seconds, a440factor)
		
		print('data[first]',data[0],'data[last]',data[-1],'min',min(data),'max',max(data))
		#return data.astype(np.int16)
		wavfile.write(ofname+'.'+self.label+'.wav', self.sps, data.astype(np.int16))
		return self
	
	def util_write_sharpened_ifft(self):
		amplitudex = np.iinfo(np.int16).max
		self.ifft_data_sharpened = self.ifft_data_sharpened / max(self.ifft_data_sharpened) * amplitudex
		wavfile.write('sharpened_ifft'+'.'+self.label+'.wav', 
					  self.sps, self.ifft_data_sharpened.astype(np.int16))
		
	
	def util_plot_footprint(self):
		'''using: nothing'''
		print(self.peak_frqs[0],self.peak_frqs[-1])
		#assert self.peak_frqs[-1] < self.sps/2
		print(self.label,self.peak_frqs)
		print(self.label,self.peak_mags)
		if False:
			devnull = plt.plot(self.peak_frqs, self.peak_mags, marker='x', linestyle='-')
		else:
			ax =plt.stem(
						self.peak_frqs, 
						self.peak_mags, 'b', markerfmt=" ", basefmt="-b")
			#ax.set_xlim([0,44100/4])
			
		plt.title('footprint of '+self.label)

	def util_get_footprint(self):
		'''using: nothing'''
		return {'pkf':list(self.peak_frqs), 'pkm':list(self.peak_mags),'pkp':list(self.peak_phas)}
		