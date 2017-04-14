import numpy as np
def get_veg_metric( veg_dd, domain_name, veg_name, aggregator=np.mean, axis=1 ):
	'''
	veg_dd = [dict] dictionary of dataframes of veg_counts metrics from alfresco_postprocessing
		* typically generated by `veg_dd = plotobj.get_metric_dataframes('veg_counts')`
	domain_name = [str] domain name
	veg_name = [str] vegetation string name -- must be EXACT... 
	aggregator = [function] function that will take an array as its only argument and return a single
		metric derived from this function.  i.e. np.mean, np.sum, np.min, etc... 
	axis = [int] 0  =aggregate across rows in the column; 1 = aggregate across columns in rows

	returns:
	pandas.Series object with the result either across years or replicates depending on axis=0 or axis=1
	'''
	return veg_dd[ domain_name ][ veg_name ].apply( aggregator, axis=axis )

def generate( ax, meanline, lower, upper, label, linecolor, rangecolor, alpha, line_zorder, *args, **kwargs ):
	'''
	overlay lines and ranges...

	ax = mpl axes object to use for plotting lines and ranges
	meanline = pandas.Series with index of years and summarized line values (mean)
	lower = pandas.Series with index of years and summarized line values (min) for fill_between
	upper = pandas.Series with index of years and summarized line values (max) for fill_between
	linecolor = matplotlib compatible color for the line object
	rangecolor = matplotlib compatible color for the fill_between object
	alpha = transparency level
	'''
	# plot line
	years = meanline.index.astype( int )
	ax.plot( np.array(years), np.array(meanline), lw=0.8, label=label, color=linecolor, alpha=1, zorder=line_zorder )
	# fill between axes
	ax.fill_between( np.array(years), np.array(lower), np.array(upper), facecolor=rangecolor, alpha=alpha, linewidth=0.0, label='range' )
	return ax

def get_veg_ratios( veg_dd, domain_name='AOI_SERDP', group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] ):
	'''
	calculate ratios from lists of veg types.
	'''
	agg1 = sum([ veg_dd[ domain_name ][ i ] for i in group1 ])
	agg2 = sum([ veg_dd[ domain_name ][ i ] for i in group2 ])
	return agg1 / agg2


if __name__	 == '__main__':
	import matplotlib, os, glob
	matplotlib.use( 'Agg' )
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches
	import numpy as np
	import pandas as pd
	from alfresco_postprocessing import plot
	from matplotlib import rcParams
	from functools import partial

	# set some rcparams
	rcParams[ 'xtick.direction' ] = 'out'
	rcParams[ 'ytick.direction' ] = 'out'
	rcParams[ 'xtick.labelsize' ] = 'small'
	rcParams[ 'ytick.labelsize' ] = 'small'
	# rcParams[ 'xtick.size' ] = 'out'
	# rcParams[ 'ytick.size' ] = 'out'
	rcParams[ 'figure.titlesize' ]  = 'medium'
	rcParams[ 'axes.titlesize' ] = 'medium'

	# read in the data
	models = [ 'CCSM4','MRI-CGCM3' ] # 'GFDL-CM3', 'GISS-E2-R', 'IPSL-CM5A-LR', 
	scenario = 'rcp85'
	# json = '/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/Data/ALFRESCO_SERDP_highcalib/post_processing_outputs/GISS-E2-R_rcp85.json'
	# json_alt = '/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/Data/ALFRESCO_SERDP_highcalib/post_processing_outputs/GISS-E2-R_rcp85_AltFMO.json'
	# json_no = '/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/Data/ALFRESCO_SERDP_highcalib/post_processing_outputs/GISS-E2-R_rcp85_NoFMO.json'

	# # # # NEW LISTING OF THE DATA
	json_path = '/workspace/Shared/Users/jschroder/ALFRESCO_SERDP/Full_calibration/output_json'
	# files = pd.Series( glob.glob( os.path.join( json_path, '*.json') ) )
	# files_grouped = { i:j for i,j in files.groupby( [ os.path.basename( i ).split( '.' )[ 0 ].split( '_' )[0] for i in files ] ) }

	# _ = [ os.path.join( json_path, model + '_rcp_85_' + '' ) for model in models ]
	file_groups = {}
	for model in models:
		group = [ 	os.path.join( json_path, model + '_rcp85_AltFMO.json' ), 
					os.path.join( json_path, model + '_rcp85.json' ) ] # os.path.join( json_path, model + '_rcp85_NoFMO.json' ),
		file_groups[ model ] = group

	# veg names from the alf package:
	from alfresco_postprocessing import veg_name_dict
	veg_names = veg_name_dict.values()

	# # # # # # # # # # # # # # # #
	domain_name = 'AOI_SERDP'
	aggregator = np.mean
	axis = 1

	# args_list = [(json_no, 'darkblue', 'lightblue', 1),(json_alt, 'darkred', 'crimson', 2), (json, 'black','lightgrey', 3)]
	# args_list = [(json_no, 'darkblue', 'grey', 1),(json_alt, 'darkred', 'grey', 2), (json, 'black','grey', 3)]
	# args_list = [('darkblue', 'grey', 1),('darkred', 'grey', 2), ('black','grey', 3)]
	pale_red = '#d9544d'
	denim_blue = '#3b5b92'
	args_list = [ (pale_red, pale_red, 2), (denim_blue, denim_blue, 3) ]

	for model in models:
		# setup 
		figsize = ( 14, 9 )
		fig, ax = plt.subplots( 1 )

		# setup spines
		ax.spines[ "top" ].set_visible( False ) 
		ax.spines[ "bottom" ].set_visible( True )
		ax.spines[ "right" ].set_visible( False )
		ax.spines[ "left" ].set_visible( True )

		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		
		# loop through the files:	
		for j, args in zip( file_groups[ model ], args_list ):
			k, l, m = args
			# open em up
			plotobj = plot.Plot( j, model, scenario )

			# data getter:
			veg_dd = plotobj.get_metric_dataframes( 'veg_counts' )
			if isinstance( veg_dd[ domain_name ][ 'Black Spruce' ], pd.DataFrame ):
				# get the C:D ratios
				ratios = get_veg_ratios( veg_dd, domain_name=domain_name, group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] )
				
				# subset the data frame to the years of interest
				# ratios = ratios.loc['1950':'2100']
				
				modeled_mean = ratios.mean( axis=1 )
				# modeled_min = ratios.min( axis=1 )
				# modeled_max = ratios.max( axis=1 )

				# # THE INTERQUARTILE RANGE 5/95 PERCENTILE
				f = partial( np.percentile, q=5 ) # 5th percentile
				modeled_min = ratios.apply( f, axis=1 )
				f = partial( np.percentile, q=95 ) # 95 percentile
				modeled_max = ratios.apply( f, axis=1 )

				# get labels to pass through from the json filenames << - this is hacky and will break
				label = os.path.basename( j ).split( '.' )[0].split( '_' )[-1]
				
				# label renamer: special case due to numerous id changes
				switch = { u'rcp85':'Current FMPO', u'AltFMO':'Alternative FMPO' }

				if label in switch.keys():
					label = switch[ label ]

				# plot it
				generate( ax, modeled_mean, modeled_min, modeled_max, label=label, linecolor=k, rangecolor=l, alpha=0.5, line_zorder=m )

		# legend
		handles, labels = ax.get_legend_handles_labels()
		handles = handles[:2]
		labels = labels[:2]

		# flip the order
		handles = handles[::-1]
		labels = labels[::-1]

		# ax.legend( handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
		ax.legend( handles, labels, fontsize='small', loc='upper center', ncol=2, borderaxespad=0., fancybox=False, bbox_to_anchor=(0.5, 1.02) ) # bbox_to_anchor=(0., 1, 1., 0 ), mode="expand",
		# ax.legend( handles, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

		# labels
		plt.xlabel( 'Year' )
		plt.ylabel( 'Ratio' )

		# title
		years = plotobj.years.astype( np.int )

		# switch for the output file naming to follow what Tom wants
		model_name_switch = { 'CCSM4':'NCAR-CCSM4', 'MRI-CGCM3':'MRI-CGCM3' }
		model = model_name_switch[ model ]

		plot_title = 'Conifer:Deciduous Ratios %s %d-%d \n ALFRESCO, %s, RCP 8.5 \n Upper Tanana Hydrological Basin' % ( 'SERDP', years.min(), years.max(), model )
		figsize = ( 14, 8 )
		# fig.suptitle('test title', fontsize=18, y=1.08 )
		plt.title( plot_title, y=1.08 )

		# limits
		plt.xlim( years.min(), years.max() )
		
		# ticks
		ticks = range( 1920, 2101, 20 )
		ticks.insert( 0, 1901 )
		plt.xticks( ticks, rotation=0 ) # make this dynamic!
		# plt.xticks( np.append( years[::10], 2100 ), rotation=45 ) # make this dynamic!

		# plt.minorticks_on()
		# ax.minorticks_on()

		# save it/close
		plt.savefig( '/workspace/Shared/Users/malindgren/AMY_SERDP/ALFRESCO_CD_ratios_'+model+'_'+scenario+'_iqr.png', figsize=figsize, dpi=600, bbox_inches='tight', pad_inches=0.2 )
		plt.close( 'all' )
