import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm 
import sys,os
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


def get_summary_statistics(X, list_feat = None, file_dir = None):

	""" Get summay statistics on X

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    file_dir : str, directory to save descriptive file. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")
    """

	if isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
			X = pd.DataFrame(X, columns = list_feat)
		except TypeError :
			sys.exit("Please fill in the list_feat argument")

	# Start by counting NaN
	df_desc = X.isnull().sum().to_frame()
	df_desc.columns = ["countNaN"]
	# Add mean
	df_desc["mean"] = X.mean()
	# Add standard dev
	df_desc["std"] = X.std()
	# Add mode
	df_desc["mode"] = X.mode().T
	# Add Cardinality
	data = [len(X[feat].unique()) for feat in X.columns.values]
	df_desc["Cardinality"] = data
	# Data types 
	df_desc["DataType"] = X.dtypes

	if not file_dir :
		file_dir = raw_input("Enter file directory location: ")
		# Create directory if needed
		if not os.path.exists(file_dir):
			os.makedirs(file_dir)
		if file_dir[-1] != "/" :
			file_dir+="/"
		df_desc.to_csv(file_dir + "descriptive_statistics.csv", float_format = "%.2f")

def plotPCA(X, y=None, n_components=None, plot_indices = (0,1),
			 fig_dir=None, fig_title = None, fig_format = ".png",
			 marker_size=50, alpha = 1, legend_location = "best",
			 verbose=False):

	""" Plot the PCA of data provided as input

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    n_components : int, (default None => keep all components).
        number of components to keep.

    plot_indices : tuple of int : indices of princ components to plot

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_title : str,  figure title

    fig_format : str, choose your favorite (.eps, .png)

    marker_size : int, size of scatter plot markers

    alpha : float (0 to 1) transparency of the plot markers

    legend_location : see matplotlib doc for possible values, default = best

    verbose : bool , if True, will show explained variance ratio

    """

	if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
		X = X.values

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values
		labels = np.unique(y)

	# Interactive plotting mode
	plt.ion()

	# Deal with possible NaN
	print "Removing columns with NaN"
	X = X[:,~np.isnan(X).any(axis=0)]

	print "Fitting PCA..."
	pca = PCA(n_components=n_components)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	if verbose :
		print('explained variance ratio (first two components): %s'
		      % str(pca.explained_variance_ratio_))

	# Set up the matplotlib figure
	fig, ax = plt.subplots(figsize=(15,15))

	if isinstance(labels, np.ndarray) :
		list_color=cm.Accent(np.linspace(0,1,len(np.unique(labels))))
		for index, label in enumerate(labels):
			color = list_color[index]
			plt.scatter(X_r[y==label, plot_indices[0]], 
						X_r[y==label, plot_indices[1]], 
						c=color, label="Class " + str(label), s = marker_size, alpha = alpha)
	else :
		plt.scatter(X_r[:, plot_indices[0]], 
					X_r[:, plot_indices[1]], 
					c="k", s = marker_size, alpha = alpha)

	plt.xlabel("Princ Comp # %s" % plot_indices[0] )
	plt.ylabel("Princ Comp # %s" % plot_indices[1] )
	plt.legend(loc=legend_location)			
	plt.show()
	raw_input("Inspect figure then press any key: ")

	if not fig_dir :
		fig_dir = raw_input("Enter figure directory location: ")
		# Create directory if needed
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		if not fig_title :
			fig_title = raw_input("Enter fig title (without .XXX): ")
		if fig_dir[-1] != "/" :
			fig_dir+="/"
		plt.savefig(fig_dir + fig_title + fig_format)


def plotCorr(X, list_feat = None, fontsize = 20, mask = False,
				fig_dir=None, fig_format = ".png"):

	""" Plot the correlation matrix of data provided as input

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    fontsize : int : fontsize for the axes tick labels

    mask : bool, if True, plot a triangular matrix

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
			X = pd.DataFrame(X, columns = list_feat)
		except TypeError :
			sys.exit("Please fill in the list_feat argument")

	# Define list of features 
	list_feat = X.columns.values

	# Interactive mode
	plt.ion()

	# Set up the matplotlib figure
	fig, ax = plt.subplots(figsize=(15,12))

	# plotting the correlation matrix
	R = X.corr().values
	dim = R.shape[0]

	# Optionally : plot triangular correlation matrix
	if mask :
		m =  np.tri(R.shape[0], k=-1)
		R = np.ma.array(R, mask=m)

	# Specify tick parameters
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left = "off",
    right = "off") 

	# Add xticks + 0.5 offset to center labels
	plt.xticks(np.arange(0.5,dim + .5),range(0,dim), fontsize = fontsize)
	plt.yticks(np.arange(0.5,dim + .5),range(0,dim), fontsize = fontsize)

	# Specify x tick labels and rotate them
	ax.set_xticklabels(list_feat, rotation=90)
	ax.set_yticklabels(list_feat, rotation=0)

	plt.pcolor(R, cmap="coolwarm", vmin=-1, vmax=1)
	plt.colorbar()

	plt.show()
	raw_input("Inspect figure then press any key: ")

	if not fig_dir :
		fig_dir = raw_input("Enter figure directory: ")
		# Create directory if needed
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		if fig_dir[-1] != "/" :
			fig_dir+="/"
		plt.savefig(fig_dir + "CorrelationMatrix" + fig_format)

def plot1D(X, list_feat = None, 
			bins= 100, color="steelblue", alpha=0.7,
			histtype = "stepfilled",
			fig_dir=None, fig_format = ".png"):

	""" Plot 1D histogram for each column in X

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    bins : int : binning for the matplotlib histogram

    color : str, color for the histogram

    alpha : float, transparency for the histogram

    histtype : str, matplotlib histogram type

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
			X = pd.DataFrame(X, columns = list_feat)
		except TypeError :
			sys.exit("Please fill in the list_feat argument")

	# Define list of features 
	list_feat = X.columns.values

	for feat in list_feat :
		print feat
		x = X[feat].values
		print "Removing NaN for the plot"
		x = x[np.logical_not(np.isnan(x))]
		# Plot and save histogram
		plt.hist(x, bins = bins, histtype = histtype, color = color, alpha = alpha)
		plt.xlabel(feat)
		if not fig_dir :
			fig_dir = raw_input("Enter figure directory: ")
			# Create directory if needed
			if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)
			if fig_dir[-1] != "/" :
				fig_dir+="/"
		plt.savefig(fig_dir + feat + fig_format)
		plt.clf()


def plot1Dbyclass(X, y, list_feat = None, single_out = [],
				bins= 100, alpha=0.7, histtype = "stepfilled", stacked = True,
				logscale = True, fontsize = 20, legend_location = "upper center",
				bbox_to_anchor=(0.5, 1.25),
				fig_dir=None, fig_format = ".png"):

	""" Plot 1D histogram for each column in X, with a class separation

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    single_out : list of int, a list of classes to single out in the plot 
    			 (i.e. plot all other classes vs singled out classes)

    bins : int : binning for the matplotlib histogram

    alpha : float, transparency for the histogram

    histtype : str, matplotlib histogram type

    stacked : bool, whether or not to stack the histograms

    logscale : bool, if True use logscale on Y axis 

    fontsize : int, fontsize for X axis label 

    legend_location : str, see matplotlib documentation for possible values 
    				  best is to keep it at its default parameters

    bbox_to_anchor : tuple of floats, where to anchor the legend box

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
			X = pd.DataFrame(X, columns = list_feat)
		except TypeError :
			sys.exit("Please fill in the list_feat argument")

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values

	# Define list of features 
	list_feat = X.columns.values

	labels = np.unique(y)
	list_color=cm.Accent(np.linspace(0,1,len(labels)))

	if single_out == [] :
		# Plot with different color for each class
		for feat in list_feat :
			list_hist = [X[feat].values[y==label] for label in labels]
			# Remove possible NaN
			list_hist = [x[np.logical_not(np.isnan(x))] for x in list_hist]
			n,_,_ = plt.hist(list_hist, bins = bins, histtype=histtype, 
								color=list_color, stacked = stacked,
								label = ["Class %s" % i for i in labels])
			# Improve plot boundaries
			n = np.ravel(n)
			plt.xlim([np.min(X[feat]) - np.abs(np.std(X[feat])),\
					 np.max(X[feat]) + np.abs(np.std(X[feat]))])
			plt.ylim([np.min(n[np.nonzero(n)]),\
					 np.max(n) + np.abs(np.std(n))])
			# Add legend and labels
			plt.legend(loc = legend_location, 
						bbox_to_anchor=bbox_to_anchor,
          				ncol=len(labels)/2, fancybox=True)
			plt.xlabel(feat, fontsize = fontsize)

			# Adjust white space for better visibility
			plt.subplots_adjust(top=0.8)

			if logscale :
				plt.yscale("log")
			if not fig_dir :
				fig_dir = raw_input("Enter figure directory: ")
				# Create directory if needed
			if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)
			if fig_dir[-1] != "/" :
				fig_dir+="/"
			plt.savefig(fig_dir + feat + "_byclass" + fig_format)
			plt.clf()
			plt.close()

	elif single_out != [] :
		# Single out class labels in the single_out list wrt all other classes
		# Remap labels :
		new_y = np.copy(y)
		d_map = {}
		for label in labels :
			if label in single_out :
				d_map[label] = 1
			else :
				d_map[label] = 0
		for k, v in d_map.iteritems(): new_y[y==k] = v

		for feat in list_feat :
			list_hist = [X[feat].values[new_y==label] for label in [0, 1]]
			not_single = [label for label in labels if label not in single_out]
			# Remove possible NaN
			list_hist = [x[np.logical_not(np.isnan(x))] for x in list_hist]
			n,_,_ = plt.hist(list_hist, bins = bins, histtype=histtype, 
								color=list_color[:2], stacked = stacked,
								label = ["Class " + "-".join(map(str, not_single)),\
										 "Class " + "-".join(map(str, single_out))] )
			# Improve plot boundaries
			n = np.ravel(n)
			plt.xlim([np.min(X[feat]) - np.abs(np.std(X[feat])),\
					 np.max(X[feat]) + np.abs(np.std(X[feat]))])
			plt.ylim([np.min(n[np.nonzero(n)]),\
					 np.max(n) + np.abs(np.std(n))])
			# Add legend and labels
			plt.legend(loc = legend_location, 
						bbox_to_anchor=bbox_to_anchor,
          				ncol=len(labels)/2, fancybox=True)
			plt.xlabel(feat, fontsize = fontsize)

			# Adjust white space for better visibility
			plt.subplots_adjust(top=0.8)

			if logscale :
				plt.yscale("log")
			if not fig_dir :
				fig_dir = raw_input("Enter figure directory: ")
				# Create directory if needed
			if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)
			if fig_dir[-1] != "/" :
				fig_dir+="/"
			plt.savefig(fig_dir + feat + "_byclass_" + "-".join(map(str,single_out)) + "_vs _rest" + fig_format)
			plt.clf()
			plt.close()

def plotViolinbyclass(X, y, list_feat = None,
				alpha=1, fontsize = 20, logscale = False,
				showmeans=False, showmedians=False, showextrema=False,
				fig_dir=None, fig_format = ".png"):

	""" Violin Plot for each column of X with a class separation

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    alpha : float, transparency for the histogram

    logscale : bool, if True use logscale on Y axis 

    fontsize : int, fontsize for X axis label 

    showmeans/showmedians/showextrema : bool, whether to show means/medians/extrema

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
			X = pd.DataFrame(X, columns = list_feat)
		except TypeError :
			sys.exit("Please fill in the list_feat argument")

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values

	# Define list of features 
	list_feat = X.columns.values

	labels = np.unique(y)
	list_color=cm.Accent(np.linspace(0,1,len(labels)))

	# Plot with different color for each class
	for feat in list_feat :

		data = [X[feat].values[y == label] for label in labels]

		# Set up the matplotlib figure
		fig, ax = plt.subplots(figsize=(15,12))
		# plot violin plot
		print "Computing KDEs..."
		violin_parts = ax.violinplot(data, showmeans=showmeans,
						showmedians=showmedians,
						showextrema= showextrema)
		# Change color
		for index, pc in enumerate(violin_parts['bodies']):
			pc.set_facecolor(list_color[index])
			pc.set_edgecolor('black')

		# Adjust plot boundaries
		plt.ylim([np.min(X[feat].values) - np.abs(np.std(X[feat].values)),\
				  np.max(X[feat].values) + np.abs(np.std(X[feat].values))])

		# adding horizontal grid lines
		ax.yaxis.grid(True)
		ax.set_xticks([c+1 for c in range(len(data))])
		ax.set_xlabel('Class', fontsize = fontsize)
		ax.set_ylabel(feat + " values", fontsize = fontsize)

		# add x-tick labels
		plt.setp(ax, xticks=[c+1 for c in range(len(data))],
		         xticklabels=map(str, labels))

		if logscale :
			plt.yscale("log")

		if not fig_dir :
			fig_dir = raw_input("Enter figure directory: ")
			# Create directory if needed
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		if fig_dir[-1] != "/" :
			fig_dir+="/"
		plt.savefig(fig_dir + feat + "_violinbyclass" + fig_format)
		plt.clf()
		plt.close()


def plot2D(X, feat1 = None, feat2 = None,
				alpha=0.7, color = "steelblue", marker_size = 100,
				logXscale = False, logYscale = False, fontsize = 20,
				fig_dir=None, fig_format = ".png"):

	""" Plot the 2D scatter plot of two features of interest

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    feat1/feat2 : str, name of the features to plot against each other

    alpha : float, set the transparency of the scatter plot

    color : str, the color of the bar histogram

    marker_size : int, size of the scatter plot marker

    logX/Yscale : bool, whether to use a log scale

    fontsize : int, the size of the X axis legend 

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
		assert feat1 != None, "Provide feat1"
		assert feat2 != None, "Provide feat2"
		X = X[[feat1, feat2]].values
		
	elif isinstance(X, np.ndarray) :
		assert X.shape[1]==2, "X should be of dim (n_samples, 2)"
		assert feat1 != None, "Provide feat1"
		assert feat2 != None, "Provide feat2"

	# Remove possible NaNs
	print "Removing possible NaNs"
	list_not_nan = np.logical_and(np.logical_not(np.isnan(X[:,0])), 
						np.logical_not(np.isnan(X[:,1])))
	X = X[list_not_nan]

	# Plot
	plt.scatter(X[:,0], X[:,1], c=color, s = marker_size, alpha = alpha)
	# Add labels to axes
	plt.xlabel(feat1, fontsize = fontsize)
	plt.ylabel(feat2, fontsize = fontsize)

	if logXscale :
		plt.xscale("log")
	if logYscale :
		plt.yscale("log")

		if fig_dir[-1] != "/" :
			fig_dir+="/"

	if not fig_dir :
		fig_dir = raw_input("Enter figure directory: ")
		# Create directory if needed
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)
	if fig_dir[-1] != "/" :
		fig_dir+="/"

	plt.savefig(fig_dir + feat2 + "_vs_" + feat1 + fig_format)
	plt.clf()
	plt.close()


def plot2Dbyclass(X, y, feat1 = None, feat2 = None, single_out = [],
				alpha=0.7, marker_size = 100,
				logXscale = False, logYscale = False, fontsize=20,
				legend_location = "upper center", bbox_to_anchor=(0.5, 1.25),
				fig_dir=None, fig_format = ".png"):

	""" Plot the 2D scatter plot of two features of interest
	    Add a different color for each class

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    feat1/feat2 : str, name of the features to plot against each other

    alpha : float, set the transparency of the scatter plot

    marker_size : int, size of the scatter plot marker

    logX/Yscale : bool, whether to use a log scale

    fontsize : int, the size of the X axis legend 

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
		assert feat1 != None, "Provide feat1"
		assert feat2 != None, "Provide feat2"
		X = X[[feat1, feat2]].values
		
	elif isinstance(X, np.ndarray) :
		assert X.shape[1]==2, "X should be of dim (n_samples, 2)"
		assert feat1 != None, "Provide feat1"
		assert feat2 != None, "Provide feat2"

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values
	
	# Interactive plotting
	plt.ion()

	# Remove possible NaNs
	print "Removing possible NaN"
	list_not_nan = np.logical_and(np.logical_not(np.isnan(X[:,0])), 
						np.logical_not(np.isnan(X[:,1])))
	X = X[list_not_nan]
	y = y[list_not_nan]

	# Get list of labels and list of color
	labels = np.unique(y)
	list_color=cm.Accent(np.linspace(0,1,len(labels)))

	if single_out == [] :
		# Plot with different color for each class
		for i, label in enumerate(labels) :
			plt.scatter(X[:,0][y==label], X[:,1][y==label], 
				c=list_color[i], 
				label = "Class %s" % label, 
				s = marker_size, 
				alpha = alpha)

		# Add legend and labels
		plt.legend(loc = legend_location, 
						bbox_to_anchor=bbox_to_anchor,
          				ncol=len(labels)/2, fancybox=True)
		plt.xlabel(feat1, fontsize = fontsize)
		plt.ylabel(feat2, fontsize = fontsize)
		# Adjust white space for better visibility
		plt.subplots_adjust(top=0.8)

		if logXscale :
			plt.xscale("log")
		if logYscale :
			plt.yscale("log")

		plt.show()
		raw_input("Inspect plot then press any key: ")

		if not fig_dir :
			fig_dir = raw_input("Enter figure directory: ")
			# Create directory if needed
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		if fig_dir[-1] != "/" :
			fig_dir+="/"

		plt.savefig(fig_dir + feat2 + "_vs_" + feat1 + "_byclass" + fig_format)


	elif single_out != [] :
		# Single out class labels in the single_out list wrt all other classes
		# Remap labels :
		new_y = np.copy(y)
		d_map = {}
		for label in labels :
			if label in single_out :
				d_map[label] = 1
			else :
				d_map[label] = 0
		for k, v in d_map.iteritems(): new_y[y==k] = v
		not_single = [l for l in labels if l not in single_out]

		# Plot with different color for each class
		plt.scatter(X[:,0][new_y==0], X[:,1][new_y==0], 
			c=list_color[0], 
			label = "Classe(s) " + "-".join(map(str, not_single)), 
			s = marker_size, 
			alpha = alpha)
		plt.scatter(X[:,0][new_y==1], X[:,1][new_y==1], 
			c=list_color[1], 
			label = "Classe(s) " + "-".join(map(str, single_out)), 
			s = marker_size, 
			alpha = alpha)

		# Add legend and labels
		plt.legend(loc = legend_location, 
						bbox_to_anchor=bbox_to_anchor,
          				ncol=len(labels)/2, fancybox=True)
		plt.xlabel(feat1, fontsize = fontsize)
		plt.ylabel(feat2, fontsize = fontsize)
		# Adjust white space for better visibility
		plt.subplots_adjust(top=0.8)

		if logXscale :
			plt.xscale("log")
		if logYscale :
			plt.yscale("log")

		plt.show()
		raw_input("Inspect plot then press any key :")

		if not fig_dir :
			fig_dir = raw_input("Enter figure directory: ")
			# Create directory if needed
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		if fig_dir[-1] != "/" :
			fig_dir+="/"

		plt.savefig(fig_dir + feat2 + "_vs_" + feat1 +"_byclass_"\
		 + "-".join(map(str,single_out)) + "_vs _rest" + fig_format)

def plotFeatImpXGB(X, y, list_feat = None,
				objective = "reg:linear", num_rounds = 100,
				eta = 0.1, max_depth = 8, silent = 1,
				yticklabelsize = 20, fontsize = 20, color = "steelblue",
				file_dir = None, fig_dir=None, fig_format = ".png"):

	""" Plot feature importance (xgboost) for each feature in X
	Save feature importance to a csv file

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    objective : xgboost objective, see xgboost documentation

    num_rounds : int, number of boosting rounds

    eta : float, step size shrinkage 

    max_depth : int, max depth of a tree

    silent : 0/1 : 0 : print messages, 1 : silent

    yticklabelsize : int, the size of the y axis tick labels 

    fontsize : int, the size of the X axis legend 

    color : str, the color of the bar histogram

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values

	if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
		list_feat = X.columns.values
		X = X.values

	elif isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
		except TypeError :
			sys.exit("Please fill in the list_feat argument")	
	
	# XGB parameters
	xgb_params = {"objective": objective, 
				  "eta": eta, 
				  "max_depth": max_depth, 
				  "silent": silent}

	# Prepare xgboost matrix
	xgtrain = xgb.DMatrix(X, label=y)
	print "Computing scores..."
	bst = xgb.train(xgb_params, xgtrain, num_rounds)
	
	importance = bst.get_fscore()
	# This gets feature importance as {f0 : 10, f3 : 23, f33 : 4}
	# Let's map this back to feature names
	list_f, list_s = [], []
	for key in importance.keys() :
		# Convert fXX to feature name :
		feat = list_feat[int(key[1:])]
		# Add this to feature and score list
		list_f.append(feat)
		list_s.append(importance[key])
	
	data = np.array([list_f, list_s]).T

	df_feat = pd.DataFrame(data[:,0], columns=["Feature"])
	df_feat["Score"] = data[:,1].astype(int)
	df_feat["Score"] = df_feat["Score"] / df_feat["Score"].sum()
	# Sort by score 
	df_feat = df_feat.sort_values("Score", ascending=False)
	# Save to a file
	if not file_dir :
		file_dir = raw_input("Enter directory to save feat importance file: ")
	if file_dir[-1] != "/" :
		file_dir+="/"
	# Create directory if needed
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	df_feat.to_csv(file_dir + "feature_importance_xgb.csv", index = False)
	
	mpl.rcParams["ytick.labelsize"] = yticklabelsize
	plt.figure()
	df_feat.plot(kind="barh", x="Feature", y="Score", color = color, legend=False, figsize=(15, 15))
	plt.xlabel("XGBoost Relative importance", fontsize = 20)
	plt.ylabel("")

	if not fig_dir :
		fig_dir = raw_input("Enter figure directory: ")
	if fig_dir[-1] != "/" :
		fig_dir+="/"
	# Create directory if needed
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)
	plt.savefig(fig_dir + "feature_importance_xgboost" + fig_format)


def plotFeatImpKBest(X, y, list_feat = None,
				score_func = "regression", yticklabelsize = 20,
				fontsize = 20, color = "steelblue",
				file_dir = None, fig_dir=None, fig_format = ".png"):

	""" Plot feature importance (F-score ANOVA) for each feature in X
	Save feature importance to a csv file

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape = (n_samples, n_features)

    y : pd.Series or pd.DataFrame or np.ndarray, shape = (n_samples,)
    	the array of target values for a classification task

    list_feat : list of str, the name of each column of X. No need 	
    			to specify it if X is a pd.DataFrame

    score_func : either "regression" or "classification,
    			 it's the score function to use to rank features

    yticklabelsize : int, the size of the y axis tick labels 

    fontsize : int, the size of the X axis legend 

    color : str, the color of the bar histogram

    fig_dir : str, directory to save figure. Specify the path relative to 
    	current directory (e.g. "./Figures/") 
    	or an absolute path  (e.g. "~/Desktop/")

    fig_format : str, choose your favorite (.eps, .png)
    """

	if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
		y = y.values

	if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
		list_feat = X.columns.values
		X = X.values

	elif isinstance(X, np.ndarray) :
		try :
			assert len(list_feat) == X.shape[1], \
			"Length of list_feat does not match X.shape[1]"
		except TypeError :
			sys.exit("Please fill in the list_feat argument")	

	if score_func == "regression":
		score_func = f_regression
	elif score_func == "classification":
		score_func = f_classif
	else :
		sys.exit("Wrong choice of classification function\
				\nCheck scikit slectKBest documentation\
				\n Choose between f_regression and f_classif")

	# Use SelectKBest
	KB = SelectKBest(score_func, len(list_feat))
	print "Computing scores..."
	KB.fit(X, y)

	# Matrix with F score and pvalues
	F = np.zeros((len(list_feat),2))

	# Print output
	for i in range(len(list_feat)):
		F[i][0], F[i][1] = KB.scores_[i], KB.pvalues_[i]

	# Save to a dataframe
	df_feat = pd.DataFrame(F, columns=["Score", "pvalue"])
	# Use relative importance to score
	df_feat["Score"] = df_feat["Score"] / df_feat["Score"].sum()
	df_feat["Feature"] = np.array(list_feat)
	df_feat = df_feat.sort_values("Score", ascending=False)
	df_feat = df_feat[["Feature","Score","pvalue"]]
	# Save to a file
	if not file_dir :
		file_dir = raw_input("Enter directory to save feat importance file: ")
	if file_dir[-1] != "/" :
		file_dir+="/"
	# Create directory if needed
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	df_feat.to_csv(file_dir + "feature_importance_KBest.csv", index = False)

	mpl.rcParams["ytick.labelsize"] = yticklabelsize
	plt.figure()
	df_feat.plot(kind="barh", x="Feature", y="Score", color = color, legend=False, figsize=(15, 15))
	plt.xlabel("KBest Relative importance", fontsize = 20)
	plt.ylabel("")

	if not fig_dir :
		fig_dir = raw_input("Enter figure directory: ")
	# Create directory if needed
	if fig_dir[-1] != "/" :
		fig_dir+="/"
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)
	plt.savefig(fig_dir + "feature_importance_KBest" + fig_format)