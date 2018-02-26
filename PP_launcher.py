#Official ALFRESCO PP launcher, needs to be updated to include csv TRUE/FALSE and historical TRUE/FALSE
import alfresco_postprocessing as ap
from tinydb import TinyDB, Query
import os, argparse, json

parser = argparse.ArgumentParser( description='' )
parser.add_argument( "-p", "--base_path", action='store', dest='base_path', type=str, help="path to output directory" )
parser.add_argument( "-shp", "--shapefile", action='store', dest='shp', type=str, help="full path to the subdomains shapefile used in subsetting" )
parser.add_argument( "-field", "--field_name", action='store', dest='id_field', type=str, help="field name in shp that defines subdomains" )
parser.add_argument( "-name", "--name", action='store', dest='name', type=str, help="field name in shp that defines subdomains name" )
parser.add_argument( "-o", "--output", action='store', dest='out', type=str, help="output path" )
parser.add_argument( "-hist_path", "--hist_path", action='store', dest='hist_path', type=str, help="path to historical directory" )
parser.add_argument( "-cores", "--ncores", action='store', dest='ncores', type=int, help="number of cores" )
parser.add_argument( "-csv", "--csv", action='store', dest='csv', type=bool, help="Compute CSV" )
parser.add_argument( "-hist", "--hist", action='store', dest='hist', type=bool, help="Compute historical" )
parser.add_argument( "-met", "--metrics", action='store', dest='metrics', type=str, help="metric list" ,nargs="+")
parser.add_argument( "-plot", "--plot", action='store', dest='plot', type=bool, help="Compute Plots only" )
args = parser.parse_args()
ncores=args.ncores
base_path = args.base_path


if not os.path.exists( args.out ):
    os.makedirs( args.out )

json_path = os.path.join(args.out,'JSON')
if not os.path.exists( json_path ):
    os.makedirs( json_path )
csv_path = os.path.join(args.out,'CSV')
if not os.path.exists( csv_path ):
    os.makedirs( csv_path )

historical_maps_path=args.hist_path
obs_json_fn = os.path.join( json_path, 'Observed.json' )
if args.plot != True :
    #run historical
    if args.hist==True:
        if not os.path.isfile(obs_json_fn) :
    	    pp_hist = ap.run_postprocessing_historical( historical_maps_path, obs_json_fn, ncores, ap.veg_name_dict, args.shp, args.id_field, args.name)
    	    pp_hist.close()

    metrics = args.metrics
    suffix = os.path.split(base_path)[1]
    mod_json_fn = os.path.join( json_path,'_'.join([ suffix + '.json'  ]))
    maps_path = os.path.join(base_path,  'Maps')

    pp = ap.run_postprocessing( maps_path, mod_json_fn, ncores , ap.veg_name_dict ,args.shp, args.id_field, args.name )

    if args.csv==True:
    	_ = ap.to_csvs( pp, metrics, csv_path, suffix )

    pp.close()

_plot = ap.launcher_SERDP( obs_json_fn , args.out, suffix , args.out)
