"""
Create a class RamanMap with attribute designed for a bokeh plotting app
Methds:
Initialization:
    __init__(self, input_file, resolution = 1600): read input info file
    getRaman(self): read asc file and format it into pd.DataFrame
    selectPeaks(self): Find peaks in self.mean spectrum
                       and returns top prominant n_peaks
    
sliceMap(self, i): Prepare slice #i for plotting from self.df
includes maps of intensity, max_shift, example spectra (min, med, max)
    
Misc Methods
    idVal(self, df,value = None): returns the index of a value in a pd.Series
                                  default values id the median
    indexer(self): returns indices of pixels [[(0,0), (0,1).],[...(49,49]])
                    size = dim*dim
    min_med_max(self, series): return min, median, max of a pd.Series


imageKeepOrder(self, image): tries to maintain order of predicted labels
                        by making sure clusters are numbered in the same
                        order of appearance in the image array
prepare_training_data(self, training_features = 25, x = False)
predictorOptimizer(self, n_features): run kmean over n_clusters, returns inertia
predictor(self, n_mat = 3, n_pca = 2): returns dictionary of {
                                     predicted lables, pca.explained_variance_}


Plotting Methods:
meanPlotter(self): plots a figure of self.mean, self.var and selected peaks
slicePlotter(self, i): plots #ith map of maxima positions, intensities
                                  and corresponding min_med_max spectra


Attributs:
    self.resolution = number of 1D pixels in CCD sensor
                defined in __init__, default = 1600
    self.n_peaks = #peaks to be detected, defined from info.txt in __init__
    self.file = full path of raw data file
    self.dim = 1 diminsion of map
    self.darkCounts = np.nanmin(df.values)
    self.var = df.var()
    self.mean = df.mean()
    self.df = df, map data in a pd.DataFrame
    self.array = self.df.values, np.array of data
    self.selected_waves = wavenumbers of selected peaks
    self.groups = list of ranges of wavelengths centered at self.selected_waves,
                   width = 4*width of detected peak
    self.selected_peaks = pd.DataFrame of selected peaks
                        [index   prominences   peak_heights   widths   waves]
    self.indices = indices of pixels [[(0,0), (0,1).],[...(49,49]])
    self.sliced_maps: selected ith maps of intensities and max_shifts
    self.sliced_spectra: corresponding example spectra 
    
    self.n_features = min(training_features, self.n_peaks)
    self.x = x, boolean training_data is scales
    self.training_data: the traning data to use in self.predictor
    self.n_mat = n_mat: number of materials in clustering model
    self.n_pca = n_pca: number of PCA in clustering model

@author: Amjad Al Taleb
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

class RamanMap:
    
    def __init__(self, input_file, resolution = 1600):
        '''
        Instantiate, run essential methods and assign some attributes
        '''
        file = open(input_file, 'r+')
        folder = file.readline().rstrip('\n')
        file_name= file.readline().rstrip('\n')
        file_ext = file.readline().rstrip('\n')
        nx = file.readline().rstrip('\n')
        file.close()
        folder = folder.replace('\\', '\\\\')
        if not folder.endswith('\\\\'):
            folder = folder + '\\\\'
        # assign attributes
        self.file = folder + file_name + '.' + file_ext
        self.resolution = resolution
        self.n_peaks = int(nx)
        
        print('reading ', self.file.replace('\\\\', '\\')  ,' into DataFrame')

        # run essential methods
        self.getRaman()  #Read file
        self.selectPeaks() #find peaks in self.mean
        self.sliceMap(0)   #prepare first slice for plotting
# =============================================================================
# ~  Core methods
# =============================================================================
    def getRaman(self):
        """
        Read asc file and return DataFrame
        Inputs: file name & file path
        Outputs: DataFrame & waves
        DataFrame has the structure (rows = points, cols = wavenumbers (1600 points))
        waves is a list 
        """
        f = self.file
        resolution = self.resolution
        sneak = pd.read_csv(f, delimiter = '\t', header = None, nrows = 1, index_col = 0)
        print('sneak returned a DataFrame with shape' +str(sneak.shape))
        
        def getStep(array):
            '''
            stripped down np.diff1d() = np.subtract(array[1:] , array[:-1])
            step size is not unique since floating points cause problems
            ix returns the index of the most frequent step size
            return the rounded value
            '''
            y,ix = np.unique(np.subtract(array[1:] , array[:-1]), return_counts = True)
            return round(y[ix.argmax()],3)
        
        if sneak.shape[1] == 0:
            print('*_*_*_*_* This file useses "," as delimiter and as decimal separator *_*_*_*_*')
            delimiter = ','
            sneak = pd.read_csv(f, delimiter = delimiter, header = None, nrows = 1, index_col = [0,1])
            dim = int(np.sqrt(sneak.shape[1]))
            ncols = int(dim**2)
            index = pd.read_csv(f, delimiter = delimiter, header = None, nrows = resolution, usecols = [0,1], dtype= {0:str, 1:str})
            df = pd.read_csv(f, delimiter = delimiter, header = None, nrows = resolution, usecols = range(2,ncols+2))# +2 since two are index
            df.index = (index[0]+'.'+index[1]).astype(float)
            df.columns = np.arange(0,ncols,1)
            df = df.T
    
        else:
            delimiter = '\t'
            if type(sneak.index.values[0]) == str:
                decimal = ','
                print('using decimal %s' %decimal)
            else:
                 decimal = '.'
                 print('using decimal %s' %decimal)
            if (sneak.shape[1] < 5) and (sneak.shape[1] > 0):
                print('This file is formatted in two columns.')
                count = sum(1 for line in open(f))
                dim = int(np.sqrt(count))
                nrows = int(dim**2)
                print('File containes %g lines, loading the first %g into DataFrame' %(count, nrows))
                df = pd.read_csv(f, delimiter = delimiter, header = None, decimal = decimal, nrows = nrows, names = ['wave','Int'], index_col = False)
                desc = df.wave.astype('category').describe()
                print(desc)
                res = int(desc[1]) #number of wavelengths
                waves = df.wave.values[0:res]
                array_size = int(desc[3])
                dim = int(np.sqrt(array_size))
                mat = df.Int.values.reshape((array_size,res))
                df = pd.DataFrame(data = mat, columns = waves) #rows are single measurements, cols are wavelengths
                self.step = getStep(waves)
                self.waves = np.around(waves, 3)
#                shift = df.iloc[0].idxmax()
#                if shift < 30: #need incident laser wavelength to correct this, for now assume a shift from 0 no more than 30 otherwise the reference is not defined
#                    print('Laser 0 is shifted by %s cm-1' % round(shift,3))
#                    waves = np.around(waves - shift,1)
#                    df.columns = waves
            else:
                dim = int(np.sqrt(sneak.shape[1]))
                ncols = int(dim**2)
                df = pd.read_csv(f, delimiter = delimiter, header = None, index_col = 0, usecols = range(ncols+1), decimal = decimal, nrows = resolution).T
                print('Finished reading. DataFrame shape:', df.shape)
                waves = df.columns.values
                self.step = getStep(waves)
                self.waves = np.around(waves, 3)
                df.columns = self.waves
                mat = df.values
            #################################
            self.dim = dim
            self.darkCounts = np.nanmin(df.values)
            self.var = df.var()
            self.mean = df.mean()
            self.array = mat
            self.df = df
# =============================================================================
# =============================================================================
    def selectPeaks(self):
        cut = pd.DataFrame(data = self.mean).describe().loc['25%'].values[0]
        peak_indices, peaks = find_peaks(self.mean, height = cut, width = 2)
#        self.selected_waves_total = self.mean.iloc[peak_indices].index.values
        df_peaks = pd.DataFrame(peaks, index = peak_indices)[['prominences','peak_heights','widths']].sort_values(by = 'prominences', ascending=False).head(self.n_peaks)
        self.selected_waves = self.waves[df_peaks.index.values]
        df_peaks['waves'] = self.selected_waves
        fwhm = df_peaks['widths'].apply(lambda x: int(round(x,0)))
#        self.fwhm = fwhm
        self.groups = [self.waves[i[0]-i[1]:i[0]+i[1]]  for i in zip(df_peaks.index.values.round().astype('int'),2*fwhm.values)]
#        self.selected_waves_mean = self.waves[peak_indices]
        self.selected_peaks = df_peaks

# =============================================================================
# ~  basic methods used in self.sliceMap()
# =============================================================================
    def idVal(self, df,value = None):
        '''
        get location of a value from a pandas.Series
        by default get the location of the median
        
        for a given value, find the closes available item in the Series
        by substracting this value from the Series.asArray() and get the
        minimum of the Abs() of the diferences
        not sure why add 1 was there
        '''
        array = np.asarray(df)
        if value == None:
            value = np.median(array)    
        idx = (np.abs(array - value)).argmin()
        return idx+1 # I don't remember why I had to add 1 here
# =============================================================================
    def indexer(self):
        '''
        Creates a 2D-list of tuples to use as index for hover in images
        '''
        sublist = []
        totallist = []
        for i in itertools.product(range(self.dim), repeat = 2):
            if i[1] == 0:
                if len(sublist) != 0:
                    totallist.append(sublist)
                sublist = []
            sublist.append(i[::-1])
        totallist.append(sublist)
        self.indices = totallist
# =============================================================================
    def min_med_max(self, series):
        mn = series.idxmin()
        md = self.idVal(series)
        mx = series.idxmax()
        return mn, md, mx
# =============================================================================
    def sliceMap(self, i):
        '''
        Prepare slice #i for plotting from self
        Inputs: i, number of slice
        Outputs: spectra, maps
            spectra: a dictionary, each item is a partial spectrum
                       {key:pd.Series}
            maps: dictionary of images to plot by bokeh.plotting.figure.image
                    {key:[np.ndarray]}
                   
        '''        
        dim = self.dim
        sub_df = self.df[self.groups[i]]
        intensities = sub_df.mean(axis =  1)-self.darkCounts
        g_max = sub_df.idxmax(axis = 1)-self.selected_waves[i]
        y_d_mn, y_d_md, y_d_mx = [sub_df.loc[i] for i in self.min_med_max(g_max)]
        y_i_mn, y_i_md, y_i_mx = [sub_df.loc[i] for i in self.min_med_max(intensities)]
        ### a compact version for later versions
#        s1 = pd.concat([sub_df.loc[i] for i in rm.min_med_max(g_max)], axis = 1)
#        s2 = pd.concat([sub_df.loc[i] for i in rm.min_med_max(intensities)], axis = 1)
#        spectra = pd.concat([s1,s2], axis = 1)
        self.indexer()
        spectra = dict(
        x = sub_df.columns.values,
        y_i_mx= y_i_mx,
        y_i_md= y_i_md,
        y_i_mn= y_i_mn,
        y_d_mx= y_d_mx,
        y_d_md= y_d_md,
        y_d_mn= y_d_mn,
        )
        maps = dict(
            g_max = [g_max.values.reshape(dim,dim)],
            intensities = [intensities.values.reshape(dim,dim)],
            index = [self.indices],
            )
        self.sliced_spectra = spectra
        self.sliced_maps = maps
        return spectra, maps
# =============================================================================
# ~    Clustering Methods
# ~    Still in experimental level
# =============================================================================
    def imageKeepOrder(self, image):
        '''
        tries to maintain order of predicted labels by making sure clusters
        are numbered in the same order of appearance in the image array
        '''
        dic = np.unique(image)
        j = 0
        dic2 = []
        for i in image:
            if i == 0:
                if i not in dic2:
                    dic2.append(i)
            if i != j:
                if i not in dic2:
                    dic2.append(i)
                if len(dic2) == len(dic): break
                j = i
        dictionary = dict(zip(dic2, dic))
        newimage = []
        for i in range(len(image)):
            newimage.append(dictionary[image[i]])
        return np.array(newimage)
# =============================================================================
    def prepare_training_data(self, training_features = 25, x = False):
        '''
        returns array shape = (dim**2, n_features*2)
        columns are intensities of maxima shift for each pixel over all #n_feature slices
        
        Output is scaled or not if x
        '''
        self.n_features = min(training_features, self.n_peaks)
        self.x = x
        ints = []
        g_xs = []   #in the future g_max and intensity training data might be separated
        for i in range(self.n_features):
            sub_df = self.df[self.groups[i]]
            intensities = sub_df.sum(axis =  1)
            g_max = sub_df.idxmax(axis = 1)-self.selected_waves[i]
#            if i < training_features:
            ints.append(intensities)
            g_xs.append(g_max)
        
        if x == True: #scale data before or after pca, requires better separation of pcas
            i_scaler = StandardScaler()
            i_s = i_scaler.fit_transform(np.array(ints).T)
            x_scaler = StandardScaler()
            x_s = x_scaler.fit_transform(np.array(g_xs).T)
            self.training_data = np.hstack([np.column_stack((i_s[:,n],x_s[:,n])) for n in range(self.n_features)])
        else:
#            self.training_data = np.array(ints).T
            self.training_data = np.hstack([np.column_stack((np.array(ints).T[:,n],np.array(g_xs).T[:,n])) for n in range(self.n_features)])
#        i_scaler = PowerTransformer()
#        i_scaler = MinMaxScaler()
#        i_scaler = QuantileTransformer()
        
# =============================================================================
    def predictorOptimizer(self, n_features):
        '''
        Used to return the inertia of kmeans
        Input: training data, type: numpy.array 
        Output: dictionary {number of clusters n_clusters , inertia of kmeans(n_clusters)}
        '''
        self.prepare_training_data(n_features)
#        self.n_features = n_features
        inertia = []
        n_clusters = range(1,15)
        for i in n_clusters:
            kmeans = KMeans(n_clusters=i)
            model = kmeans.fit(self.training_data)
            inertia.append(model.inertia_)
        # Create a data frame with two lists, num_clusters and distortions
        return dict(num_clusters = n_clusters, inertia= inertia)

# =============================================================================
    def predictor(self, n_mat = 3, n_pca = 2):
        '''
        Inputs:
            traning data(np.arrya)
            n_mat: number of materials to categorize
            n_pca: number of Principal Components used in fitting
            dim: dimention of image = sqrt(len(training data))
        Output: 2 dictionaries:
            dict1: labels of categorized materials
                image: result of predictions reshaped for image plotting in Bokeh
                index: indexer(dim) 2D list of tuples
            dict2:
                x: range of number of principal components
                top: variance of principal components
        '''
        dim = self.dim
        self.n_mat = n_mat
        n_pca = min(self.n_features, n_pca)
        self.n_pca = n_pca
        pca = PCA(n_components = n_pca)
        transformed_data = pca.fit_transform(self.training_data)
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters = n_mat)
#        image = self.imageKeepOrder(kmeans.fit_predict(transformed_data))
        pipeline = make_pipeline(scaler, kmeans)
        pipeline.fit(transformed_data)
        image = self.imageKeepOrder(pipeline.predict(transformed_data))
        return dict(image = [image.reshape(dim,dim)], index = [self.indices]), dict(x = range(n_pca), top = pca.explained_variance_)

# =============================================================================
# ~     Plotting Methods
# =============================================================================
    def meanPlotter(self):
        plt.figure(figsize=(12,8))
        plt.plot(self.mean)
        
        plt.yscale('log')
        plt.title("Map's Average Intensity")
        plt.ylabel('Intensity (CPS/pixel)')
        plt.xlabel('Wavelength (nm)')
        plt.plot(self.selected_peaks['waves'],self.selected_peaks['peak_heights'], 'ko')
        plt.twinx()
        plt.plot(self.var, 'r-')
        plt.plot(self.selected_peaks['waves'],self.selected_peaks['prominences'], 'ro')
        for i, r in self.selected_peaks.iterrows():
            plt.annotate(round(r['waves'],1), (r['waves'], r['prominences']*1.2), rotation = 90, ha='center', va = 'bottom')
        plt.yscale('log')
        plt.ylabel('Prominence').set_color('red')
        plt.tick_params('y', colors = 'r')
        plt.show()
        
# =============================================================================
    def slicePlotter(self, i):
        self.sliceMap(i)
        colors = ['r-','b-','k-']
        labels = ['max','med','min']
        ints = [i for i in self.sliced_spectra.keys()][1:4]
        maxs = [i for i in self.sliced_spectra.keys()][4:]

        plt.figure(figsize=(11.69,4.14))
        
        plt.subplot(1, 4, 1)
        plt.imshow(self.sliced_maps['g_max'][0],cmap='seismic')
        plt.title('max @ '+ str(self.selected_waves[i]))
        plt.colorbar()
        
        plt.subplot(1, 4, 2)
        plt.imshow(self.sliced_maps['intensities'][0],cmap='seismic')
        plt.title('Intensity')
        plt.colorbar()
        
        plt.subplot(1,4,3)
        [plt.plot(self.sliced_spectra[maxs[i]], colors[i], label = labels[i]) for i in range(3)]
        plt.legend()
       
        plt.subplot(1,4,4)
        [plt.plot(self.sliced_spectra[ints[i]], colors[i], label = labels[i]) for i in range(3)]
        plt.legend()
        
        plt.tight_layout()
        plt.show()