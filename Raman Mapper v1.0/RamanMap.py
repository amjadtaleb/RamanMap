import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

class RamanMap:
    
    def __init__(self, input_file, resolution = 1600):
        file = open(input_file, 'r+')
        folder = file.readline().rstrip('\n')
        file_name= file.readline().rstrip('\n')
        file_ext = file.readline().rstrip('\n')
        nx = file.readline().rstrip('\n')
        file.close()
        folder = folder.replace('\\', '\\\\')
        if not folder.endswith('\\\\'):
            folder = folder + '\\\\'
        
        self.file = folder + file_name + '.' + file_ext
        
        print('reading ', self.file.replace('\\\\', '\\')  ,' into DataFrame')

        self.resolution = resolution
        self.n_peaks = int(nx)

        self.getRaman()  #Read file
        self.grouper()
        self.sliceMap(0)



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
            self.array = mat
            self.df = df

    def selectPeaks(self):
        cut = pd.DataFrame(data = self.var).describe().loc['25%'].values[0]
        peak_indices, peaks = find_peaks(self.var, height = cut, width = 2)
        df_peaks = pd.DataFrame(peaks, index = peak_indices)[['prominences','widths']]
        selected_peaks = df_peaks.sort_values(by = 'prominences', ascending=False).head(self.n_peaks)
        selected_waves = self.waves[selected_peaks.index]
        selected_peaks['waves'] = selected_waves
        self.selected_peaks = selected_peaks
        self.selected_waves  = selected_waves 

    def grouper(self):
        '''
        Inputs are the outputs of deviationReport()
        Outputs:
        groups is a list wavenumbers containing n_peaks of 1D arrays eatch array has a length 2*fwhm
        selected_waves are the indices of the n_peaks
        '''
        self.selectPeaks()
        fwhm = self.selected_peaks['widths'].apply(lambda x: int(round(x,0)))
        self.groups = [self.waves[i[0]-i[1]:i[0]+i[1]]  for i in zip(self.selected_peaks.index.values,2*fwhm.values)]


    def devPlotter(self):
        plt.figure(figsize=(12,8))
        plt.plot(self.waves, self.var)
        plt.yscale('log')
        plt.title('Variance per wavelength')
        plt.ylabel('Variance (%)')
        plt.xlabel('Wavelength (nm)')
        plt.twinx()
        plt.plot(self.selected_peaks['waves'],self.selected_peaks['prominences'], 'ro')
        for i, r in self.selected_peaks.iterrows():
            plt.annotate(round(r['waves'],1), (r['waves'], r['prominences']*1.2), rotation = 90, ha='center', va = 'bottom')
        plt.yscale('log')
        plt.ylabel('Prominence').set_color('red')
        plt.tick_params('y', colors = 'r')
        plt.show()

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
        return idx #+1 I don't remember why I had to add 1 here
    
    def indexer(self):
        '''
        Creates a 2D-list of tuples to use as index for hover in images
        '''
        sublist = []
        totallist = []
        for i in itertools.product(range(self.dim), repeat = 2):
    #        print(i)
            if i[1] == 0:
                if len(sublist) != 0:
                    totallist.append(sublist)
                sublist = []
            sublist.append(i[::-1])
        totallist.append(sublist)
        self.indices = totallist
    
    def min_med_max(self, series):
        mn = series.idxmin()
        md = self.idVal(series)
        mx = series.idxmax()
        return mn, md, mx
    

        
        
    def plotter_plot(self, i):
        sub_df = self.df[self.groups[i]]
        intensities = sub_df.sum(axis =  1)
        g_max = sub_df.idxmax(axis = 1)-self.selected_waves[i]
        
        plt.figure(figsize=(11.69,4.14))
        
        plt.subplot(1, 4, 1)
        plt.imshow(g_max.values.reshape(self.dim,self.dim),cmap='seismic')
        plt.title('max @ '+ str(self.selected_waves[i]))
        plt.colorbar()
        plt.subplot(1, 4, 2)
        
        plt.imshow(intensities.apply(lambda x: (x-self.darkCounts)/len(self.groups[i])).values.reshape(self.dim,self.dim))
        plt.title('Intensity')
        plt.colorbar()
        
        plt.subplot(1,4,3)
        mx, mn, mm = self.min_med_max(g_max)
        plt.plot(sub_df.loc[mx], 'r-', label = 'max')
        plt.plot(sub_df.loc[mn], 'b-', label = 'min')
        plt.plot(sub_df.loc[mm], 'k-', label = 'med')
        plt.legend()
       
        plt.subplot(1,4,4)
        mx2,mn2,mm2= self.min_med_max(intensities)
        plt.plot(sub_df.loc[mx2], 'r-', label = 'max')
        plt.plot(sub_df.loc[mn2], 'b-', label = 'min')
        plt.plot(sub_df.loc[mm2], 'k-', label = 'med')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def sliceMap(self, i):
        dim = self.dim
        sub_df = self.df[self.groups[i]]

        intensities = sub_df.mean(axis =  1)
        g_max = sub_df.idxmax(axis = 1)-self.selected_waves[i]
        y_d_mn, y_d_md, y_d_mx = [sub_df.loc[i] for i in self.min_med_max(g_max)]
        y_i_mn, y_i_md, y_i_mx = [sub_df.loc[i] for i in self.min_med_max(intensities)]
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

######## Clustering functions   ###############################################
###############################################################################
    def imageKeepOrder(self, image):
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

    def prepare_training_data(self, training_features = 25):
#        global_data = {}
        training_data = []
        for i in range(self.n_peaks):
            sub_df = self.df[self.groups[i]]
            intensities = sub_df.sum(axis =  1)
#            g_max = sub_df.idxmax(axis = 1)-self.selected_waves[i]
            if i < training_features:
                training_data.append(intensities)
        self.training_data = np.array(training_data).T

    def predictorOptimizer(self, n_features):
        '''
        Used to return the inertia of kmeans
        Input: training data, type: numpy.array 
        Output: dictionary {number of clusters n_clusters , inertia of kmeans(n_clusters)}
        '''
        self.prepare_training_data(n_features)
        scaler = StandardScaler()
        training_data_scaled = scaler.fit_transform(self.training_data)
        inertia = []
        n_clusters = range(1,15)
        for i in n_clusters:
            kmeans = KMeans(n_clusters=i)
            model = kmeans.fit(training_data_scaled)
            inertia.append(model.inertia_)
        # Create a data frame with two lists, num_clusters and distortions
        return dict(num_clusters = n_clusters, inertia= inertia)
    

    
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
        pca = PCA(n_components = n_pca)
        transformed_data = pca.fit_transform(self.training_data)
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters = n_mat)
        pipeline = make_pipeline(scaler, kmeans)
        pipeline.fit(transformed_data)
        image = self.imageKeepOrder(pipeline.predict(transformed_data))
        return dict(image = [image.reshape(dim,dim)], index = [self.indices]), dict(x = range(n_pca), top = pca.explained_variance_)
#        return image

#rm = RamanMap('info.txt') #instanciate a RamanMap
## =============================================================================
## ~ example usage:
#for i in range(2):
#    s1,sx = rm.sliceMap(i)
#    plt.figure(figsize=(11.69,4.14))
#    plt.subplot(1, 4, 1)
#    plt.imshow(sx['g_max'][0])
#    plt.colorbar()
#    plt.subplot(1, 4, 2)
#    plt.plot(pd.DataFrame([s1[key] for key in s1 if len(key) > 2 and key[2] == 'i']).T)
#    plt.subplot(1, 4, 3)
#    plt.imshow(sx['intensities'][0])
#    plt.colorbar()
#    plt.subplot(1, 4, 4)
#    plt.plot(pd.DataFrame([s1[key] for key in s1 if len(key) > 2 and key[2] == 'd']).T)
#    plt.tight_layout()
#    plt.show()
## =============================================================================
