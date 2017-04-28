
import pandas as pd
import numpy as np
import glob, os, ast, sys,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
pd.options.mode.chained_assignment = None  # default='warn'
import alfresco_postprocessing as ap
import seaborn.apionly as sns


rcParams[ 'xtick.direction' ] = 'out'
rcParams[ 'ytick.direction' ] = 'out'
rcParams[ 'xtick.labelsize' ] = 'small'
rcParams[ 'ytick.labelsize' ] = 'small'
rcParams[ 'figure.titlesize' ] = 'small'
rcParams[ 'axes.titlesize' ] = 'small'
rcParams[ 'axes.spines.top' ] = 'False'
rcParams[ 'axes.spines.right' ] = 'False'
rcParams[ 'savefig.dpi' ] = 150
rcParams[ 'figure.figsize'] = 14 , 8
year_range = (1950,2100)



class Scenario( object ):
    '''
    class for storing data attributes and methods to abstract some of the
    ugliness of plotting the ALFRESCO Post Processing outputs.
    '''
    def __init__( self, json_fn, model, scenario, caption , color,*args, **kwargs ):
        '''
        Arguments:
        ----------
        json_fn = [str] path to the alfresco_postprocessing output TinyDB JSON database file
        model = [str] name of the model being processed (used in naming)
        scenario = [str] name of the scenario being processed (used in naming)

        Returns:
        --------
        object of type alfresco_postprocessing.Plot
                
        '''
        from tinydb import TinyDB
        self.json_fn = json_fn
        self.db = TinyDB( self.json_fn )
        self.records = self.db.all()
        self.model = '_'.join(model.split('_')[0:-1]).upper()
        self.scenario = scenario
        self.years = self._get_years()
        self.replicates = self._get_replicates()
        self.domains = self._get_domains()
        self.caption = caption
        self.metrics = self._get_metric_names()
        self.color = color
        self.mscenario = model.split('_')[-1].upper()
        self.patch = mpatches.Patch([], [], linewidth=1.2, color= self.color , label=self.caption )
        self.line = mlines.Line2D([], [], linewidth=1.2, color=self.color, label= self.caption )
        for metric in self.metrics:
             setattr(self, metric, self.get_metric_dataframes(metric))  

    def _get_years( self ):
        if 'fire_year' in self.records[0].keys():
            years = np.unique( [ rec['fire_year'] for rec in self.records ] ).astype( np.int )
        else :
            years = np.unique( [ rec['year'] for rec in self.records ] ).astype( np.int )

        years.sort()
        return years.astype( str )
    def _get_replicates( self ):
        replicates = np.unique( [ rec['replicate'] for rec in self.records ] )
        replicates.sort()
        return replicates
    def _get_domains( self ):
        record = self.records[0]
        metric = record.keys()[0]
        return record[ metric ].keys()
    def _get_metric_names( self ) :
        record = self.records[0]
        metric = [value for value in record if 'year' not in value if 'rep' not in value]
        return metric
    def get_metric_dataframes( self , metric_name ):
        '''
        output a dict of pandas.DataFrame objects representing the 
        data of type metric_name in key:value pairs of 
        domainname:corresponding_DataFrame

        Arguments:
        ----------
        metric_name = [str] metric name to be converted to pandas DataFrame obj(s).

        Returns:
        --------
        dict of pandas DataFrame objects from the output alfresco TinyDB json file
        for the desired metric_name
        '''
        from collections import defaultdict
        if 'fire_year' in self.records[0].keys():
            metric_select = ap.get_metric_json( self.db, metric_name )
        else :
            metric_select = ap.get_metric_json_hist( self.db, metric_name )

        panel = pd.Panel( metric_select )

        dd = defaultdict( lambda: defaultdict( lambda: defaultdict ) )
        for domain in self.domains:
            if metric_name != 'veg_counts': # fire
                dd[ domain ] = panel[ :, domain, : ]
                dd[ domain ].index = dd[ domain ].index.astype(int)

            if metric_name == 'veg_counts': # veg
                df = panel[ :, domain, : ]
                vegtypes = sorted( df.ix[0,0].keys() )
                new_panel = pd.Panel( df.to_dict() )
                for vegtype in vegtypes:
                    # subset the data again into vegetation types
                    dd[ domain ][ vegtype ] = new_panel[ :, vegtype, : ]
                    dd[ domain ][vegtype].index = dd[ domain ][vegtype].index.astype(int)
        return dict(dd)

def upcase( word ):
    _tmp = [i.title() for i in word.split('_')]
    _tmp = " ".join(_tmp)
    return _tmp

def get_veg_ratios( veg_dd, domain ,year_range = (1950,2100), group1=['White Spruce', 'Black Spruce'], group2=['Deciduous'] ):
    '''
    calculate ratios from lists of veg types.
    '''
    begin,end = year_range
    agg1 = sum([ veg_dd[ domain ][ i ].ix[begin:end] for i in group1 ])
    agg2 = sum([ veg_dd[ domain ][ i ].ix[begin:end]for i in group2 ])
    return agg1 / agg2

def fill_in(ax , df ,colors ,low_percentile = 5 , high_percentile = 95 , alpha = 0.2 ) :
    
    x = df.index.unique()

    ax.fill_between(x, df.groupby(df.index).apply(np.percentile, low_percentile ), \
    df.groupby(df.index).apply(np.percentile, high_percentile), alpha= alpha, color=colors)

    return ax

def df_processing(dictionnary , std_arg = False , cumsum_arg = False , *args):

    def _process_df(scen_arg , df , std_arg , cumsum_arg):

        if cumsum_arg == True :
            df = df.apply( np.cumsum, axis=0 )
        else : pass

        df['date'] = df.index
        df['scenario']= scen_arg


        if std_arg == True :
            df = pd.melt(df, id_vars=["date", "scenario",'std'], var_name="condition")
        else :
            df = pd.melt(df, id_vars=["date", "scenario"], var_name="condition")

        return df

    _tmp = [_process_df( k , v , std_arg , cumsum_arg) for k , v in dictionnary.iteritems()]

    df = pd.concat(_tmp, ignore_index= True)
    df = df.drop('condition', 1)
    df = df.rename(columns = {'scenario':'condition'})
    df = df.sort_values(by=['condition','date'])
    df = df.reset_index(drop=True)

    return df

def df_processing2(dictionnary , std_arg = False , cumsum_arg = False , *args):

    def _process_df(scen_arg , df , std_arg , cumsum_arg):

        if cumsum_arg == True :
            df = df.apply( np.cumsum, axis=0 )
        else : pass

        if std_arg == True :
            df['std'] = df.std(axis=1)
        else : pass
            
        return df

    _tmp = [_process_df( k , v , std_arg , cumsum_arg) for k , v in dictionnary.iteritems()]
    return _tmp[0]

def underscore_fix(string) :
    string = string.replace("_"," ")
    return string

def ticks(ax , decade=False) :

    if decade == False :

        # Getting ticks every ten years
        n = 10 # every n ticks... from the existing set of all
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [ l.get_text() for l in ax.xaxis.get_ticklabels() ]
        ax.xaxis.set_ticks( ticks[::n] )
        ax.xaxis.set_ticklabels( ticklabels[::n] )
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    else : 

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    return ax

def Alec_boxplot(scenario1 , observed , output_path , pdf, model , graph_variable, year_range , domain , *args):


    begin, end = year_range 

    if graph_variable == 'avg_fire_size' :
        plot_title = 'Average Size of Fire %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

    elif graph_variable == 'number_of_fires' :
        plot_title = 'Total Number of Fires %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel = 'Number of Fires'

    elif graph_variable == 'total_area_burned' :
        plot_title = 'Total Area Burned %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

    else : 'Error with Title'

    data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain].ix[begin : end] for scen_arg in [scenario1]}

    df = df_processing(data)

    ax = df.boxplot(by='date' , rot = 90 , grid=False )

    ax = ticks(ax)
    #get ride of the automatic title
    fig = ax.get_figure()
    fig.suptitle('')
    plt.title(plot_title)

    plt.xlabel( 'Years' )

    if graph_variable == 'total_area_burned' :
        plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')'  )
    elif graph_variable == 'avg_fire_size' :
        plt.ylabel( 'Average size in ('+'$\mathregular{km^2}$' + ')'  )
    elif graph_variable == 'number_of_fires' :
        plt.ylabel( 'Number of fires'  )

    output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_boxplot', domain,graph_variable, model , str(begin), str(end)]) + '.png' )

    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def decade_plot(scenario1 , observed , output_path , pdf, model , graph_variable, year_range , domain , *args):
    """  Takes a graphvariable/metric and create a bar plot by decade. The error values are calculated by calculating the variance amoung each reps
    for every year, then the error is calculated by taking the square root of the mean of those variance as shown here
    http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation#26647"""


    begin, end = year_range 
    end = end-1

    if graph_variable == 'avg_fire_size' :
        plot_title = 'Average Size of Fire Summed per Decade %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

    elif graph_variable == 'number_of_fires' :
        plot_title = 'Total Number of Fires per Decade %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel = 'Number of Fires'

    elif graph_variable == 'total_area_burned' :
        plot_title = 'Total Area Burned %d-%d \n ALFRESCO, %s, %s, %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
        ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

    else : 'Error with Title'

    def std_calc(df):
        mean = df.mean(axis=1).groupby(df.index // 10 * 10).sum()
        error = df.var(axis=1).groupby(df.index // 10 * 10).apply(lambda d: np.sqrt(d.mean()))
        df = pd.concat([mean,error],axis=1)
        df.columns = ['mean','error']
        return df

    # Handling the historical, oserved data
    obs_domain = observed.__dict__[graph_variable][domain].ix[begin : ]
    obs_domain['std'] = 0

    # Building the two dataframes needed for the plotting
    data = {scen_arg.scenario :std_calc(scen_arg.__dict__[graph_variable][domain].ix[begin : end]) for scen_arg in [scenario1]}
    means = pd.concat([data['scenario1']['mean'],obs_domain['observed'].groupby(obs_domain.index // 10 * 10).sum()],axis=1)
    error = pd.concat([data['scenario1']['error'],obs_domain['std'].groupby(obs_domain.index // 10 * 10).sum()],axis=1)


    #plotting
    ax = means.plot(kind='bar',yerr= error.values.T, error_kw={'ecolor':'grey','linewidth':1},legend=False, color = [scenario1.color , observed.color], title=plot_title,  grid=False, width=0.8 )

    #Create label for axis
    plt.ylabel( ylabel )
    plt.xlabel( 'Decade' )
    plt.ylim(ymin=0 ) 

    ax = ticks(ax , decade = True)
    
    plt.legend(handles = [ scenario1.patch , observed.patch],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

    output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_barplot_decade', domain,graph_variable, model , str(begin), str(end)]) + '.png' )

    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def bar_plot(scenario1 , observed , output_path , pdf, model , graph_variable,year_range , domain , *args):
    # take a graphvariable, average over reps for a year and sums it over a decade.



    begin, end = year_range 
    end = end-1
    plot_title = '%s %d-%d \n ALFRESCO, %s, %s, %s' % (upcase(graph_variable), begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )

    if graph_variable == 'avg_fire_size' :
        ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

    elif graph_variable == 'number_of_fires' :
        ylabel = 'Number of Fires'

    elif graph_variable == 'total_area_burned' :
        ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

    else : 'Error with Title'

    def std_calc(df):
        df['std'] = df.std(axis=1)
        return df

    #Handling the historical, oserved data
    obs_domain = observed.__dict__[graph_variable][domain].ix[begin : ]

    data = {scen_arg.scenario :std_calc(scen_arg.__dict__[graph_variable][domain].ix[begin : end]) for scen_arg in [scenario1]}

    df = df_processing(data , std_arg = True)

    df = df.groupby(["condition", "date"]).mean().unstack("condition")

    #help to create those as the yerr is pretty sensitive to changes, had to create a 0 columns for std.
    errors = df['std']
    errors['obs'] = 0
    means = df['value']
    # means['obs'] = observed.__dict__[graph_variable][domain].ix[begin : ]

    #plotting
    ax = means.plot(kind='bar',yerr= errors.values.T, error_kw={'ecolor':'grey','linewidth':0.2},legend=False, color = [scenario1.color ], title=plot_title,  grid=False )

    #Create label for axis
    plt.ylabel( ylabel )
    plt.xlabel( 'Years' )
    plt.ylim(ymin=0 ) 

    ax = ticks(ax)
    
    plt.legend(handles = [ scenario1.patch],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

    output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_indiv_bar', domain,graph_variable, model , str(begin), str(end)]) + '.png' )

    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_metric(scenario1 , observed , output_path , pdf, model , graph_variable, year_range , domain , cumsum=True , *args):
    #This plot compare the cumulative area burn for managed, unmanaged and historical period


    begin, end = year_range

    #Set some Style and settings for the plots
    if cumsum == True :
        plot_title = 'ALFRESCO Cumulative Sum of %s %d-%d \n %s - %s \n %s' % (upcase(graph_variable) , begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )
    else :
        plot_title = 'ALFRESCO Annual %s %d-%d \n %s - %s \n %s' % (upcase(graph_variable), begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )

    #Handling the historical, oserved data
    if cumsum == True :
        obs_domain = np.cumsum( observed.__dict__[graph_variable][domain].ix[begin : ] )
    else : 
        obs_domain = observed.__dict__[graph_variable][domain].ix[begin : ]

    data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain].ix[begin : end] for scen_arg in [scenario1]}

    df = df_processing2(data , cumsum_arg = cumsum)

    #checking if colors_list list work
    ax = df.plot( legend=False, color=['0.75'],  grid=False )
    df.mean(axis=1).plot(ax=ax , legend=False, color = scenario1.color, title=plot_title,lw = 1,  grid=False)
    obs_domain.plot( ax=ax,legend=False, color=observed.color, grid=False, label= "observed" ,lw = 1)

    #Create label for axis
    plt.xlabel( 'Years' )
    if graph_variable == 'avg_fire_size' :
        ylabel ='Average Fire Size ('+'$\mathregular{km^2}$' + ')' 

    elif graph_variable == 'number_of_fires' :
        ylabel = 'Number of Fires'

    elif graph_variable == 'total_area_burned' :
        ylabel = 'Area Burned in ('+'$\mathregular{km^2}$' + ')'

    else : 'Error with Title'
    plt.ylabel( ylabel )
    ax = ticks(ax , decade=True)

    #have to pass the scenario object so they are avalaible for color definition
    replicate = mlines.Line2D([], [], linewidth=1.2, color='0.75', label= 'Replicates' )
    plt.legend(handles = [ scenario1.line , observed.line, replicate],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

    if cumsum == True :
        output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_lines_cumsum',domain,graph_variable, model , str(begin), str(end)]) + '.png' )
    else : 
        output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_lines_annual',domain,graph_variable, model , str(begin), str(end)]) + '.png' )

    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_cab_vs_fs(scenario1 , observed , output_path , pdf, model , graph_variable, year_range , domain , *args):
    #This graph shows the cumulative area burnt by fire size, managed and unmanaged scenario are compared on the same plot
    #Mainly based on Michael's code https://github.com/ua-snap/alfresco-calibration/blob/cavm/alfresco_postprocessing_plotting.py#L252
    
    begin, end = year_range

    fig, ax = plt.subplots() 

    def wrangling(df , color , scenario):
        if scenario!= 'observed' :

            df_list = []
            for col in df.columns[1:]:
                mod_sorted = sorted( [ j for i in df[ col ].astype(str) for j in ast.literal_eval(i) ] )
                mod_cumsum = np.cumsum( mod_sorted )
                replicate = [ col for i in range( len( mod_sorted ) ) ]
                df_list.append( pd.DataFrame( {'mod_sorted':mod_sorted, 'mod_cumsum':mod_cumsum, 'replicate':replicate} ) )
            mod_melted = pd.concat( df_list )   
            mod_melted.groupby( 'replicate' ).apply( lambda x: plt.plot( x['mod_sorted'], x['mod_cumsum'], color=color, alpha=0.5, lw=1) )

        else :
            mod_sorted = sorted( [ j for i in df[df.columns[0]].astype(str) for j in ast.literal_eval(i) ] )
            mod_cumsum = np.cumsum( mod_sorted )
            plt.plot( mod_sorted, mod_cumsum, color=color, alpha=0.5, lw=1)


    wrangling(scenario1.__dict__[graph_variable][domain].ix[begin : end] , scenario1.color ,'scenario1')
    wrangling(observed.__dict__[graph_variable][domain].ix[begin : ], observed.color , 'observed')

    #Create label for axis
    plt.ylabel( 'Area burned in ('+'$\mathregular{km^2}$' + ')' )
    plt.xlabel( 'Fire size ('+'$\mathregular{km^2}$' + ')' )

    fig.suptitle('Cumulative Area Burned vs. Fire Sizes %d-%d \n ALFRESCO, %s, %s, %s' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) ))

    plt.legend(handles = [ scenario1.line , observed.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

    output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_cab_vs_fs',domain, model , str(begin), str(end)]) + '.png' )

    plt.savefig( output_filename )
    pdf.savefig()
    plt.close()

def compare_vegcounts(scenario1  , observed , output_path , pdf, model , graph_variable,year_range , domain , *args):


    begin, end = year_range #subset the dataframes to the years of interest

    for veg_name in scenario1.veg_counts[domain].keys():
        try :
            plot_title = "ALFRESCO Vegetation Annual %s Cover area %s-%s \n %s - %s \n %s" \
                % ( veg_name, str(begin), str(end),scenario1.model,scenario1.mscenario,underscore_fix(domain) )

            data = {scen_arg.scenario :scen_arg.__dict__[graph_variable][domain][veg_name].ix[begin : end] for scen_arg in [scenario1]}

            df =df_processing2(data)
            # Plot the average value by condition and date
            ax = df.mean(axis=1).plot(legend=False, color = scenario1.color, title=plot_title,lw = 0.7,  grid=False )

            ax = ticks(ax , decade=True)
            
            #Create label for axis
            plt.xlabel( 'Year' )
            plt.ylabel( 'Area Covered ('+'$\mathregular{km^2}$' + ')' )

            fill_in(ax , df , scenario1.color ,low_percentile = 5 , high_percentile = 95)

            plt.legend(handles = [ scenario1.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)

            output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_annual_areaveg_line',model, domain, veg_name.replace(' ', '' ), str(begin), str(end) ]) + '.png' ) 

            plt.savefig( output_filename )
            pdf.savefig()
            plt.close()
        except :
            pass


def CD_ratio(scenario1 , observed , output_path , pdf, model , graph_variable, year_range , domain , *args):

    begin, end = year_range


    try :
        plot_title = 'ALFRESCO Conifer:Deciduous Ratios %d-%d \n %s - %s \n %s ' % ( begin, end, scenario1.model,scenario1.mscenario,underscore_fix(domain) )


        data = {scen_arg.scenario : get_veg_ratios( scen_arg.__dict__[graph_variable], domain ) for scen_arg in [scenario1] }


        df =df_processing2( data )  
        ax = df.mean(axis=1).plot(legend=False, color = scenario1.color, title=plot_title,lw = 0.7,  grid=False )

        ax = ticks(ax, decade=True)
        
        #Create label for axis
        plt.xlabel( 'Year' )
        plt.ylabel( 'C:D Ratio' )

        fill_in(ax , df , scenario1.color ,low_percentile = 5 , high_percentile = 95)

        plt.legend(handles = [ scenario1.line ],fontsize='medium',loc='best',borderaxespad=0.,ncol=1,frameon=False)


        output_filename = os.path.join( output_path, domain , '_'.join([ 'alfresco_CD_ratio',domain, model, str(begin), str(end) ]) + '.png' ) 

        plt.savefig( output_filename )
        pdf.savefig()
        plt.close()
    except :
        pass


def launcher_SERDP(obs_json_fn,src_path, model , out ) :

    from collections import defaultdict


    json = os.path.join(src_path , 'JSON' , model + '.json' )
    json_obs = os.path.join( obs_json_fn )

    mod_obj= Scenario( json, model, 'scenario1', model , '#000000')
    hist_obj = Scenario( json_obs, model, 'Observed', "Historical", '#B22222' )

    visual = os.path.join( out , 'Plots_SERDP' )
    if not os.path.exists( visual ):
        os.mkdir( visual )


    output_path = os.path.join( visual , model )
    if not os.path.exists( output_path ) :
        os.makedirs( output_path )

    for i in mod_obj.domains :
        _tmp = os.path.join(output_path,i)
        if not os.path.exists( _tmp ) :
            os.makedirs( _tmp )
    for domain in mod_obj.domains:
        pdf = os.path.join( output_path, '_'.join([ model, domain ,'plots']) + '.pdf' )

        with PdfPages(pdf) as pdf:

            _ = [decade_plot(mod_obj , hist_obj , output_path , pdf, model , metric, year_range ,domain) for metric in mod_obj.metrics if metric not in [ 'veg_counts' , 'all_fire_sizes', 'severity_counts']]
            _ = [compare_metric(mod_obj , hist_obj , output_path , pdf, model , metric, year_range ,domain , cumsum=False) for metric in mod_obj.metrics if metric not in [ 'veg_counts' , 'all_fire_sizes' , 'severity_counts']]
            compare_metric(mod_obj , hist_obj , output_path , pdf, model , 'total_area_burned', year_range ,domain, cumsum=True)
            _ = [bar_plot(mod_obj , hist_obj , output_path , pdf, model , metric, year_range, domain) for metric in mod_obj.metrics if metric not in [ 'veg_counts' , 'all_fire_sizes' , 'severity_counts']]
            CD_ratio(mod_obj , hist_obj , output_path , pdf, model , 'veg_counts', year_range, domain)
            compare_vegcounts(mod_obj  , hist_obj , output_path , pdf, model , 'veg_counts', year_range, domain)
            compare_cab_vs_fs(mod_obj , hist_obj , output_path , pdf, model , 'all_fire_sizes', year_range , domain)
            _ = [Alec_boxplot(mod_obj , hist_obj , output_path , pdf, model , metric, year_range, domain) for metric in mod_obj.metrics if metric not in [ 'veg_counts' , 'all_fire_sizes', 'severity_counts']]


