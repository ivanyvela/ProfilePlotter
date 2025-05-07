#-------------------------------------------------INMEDIATE TODOS-----------------------------------------------
#TODO: PC_01 only has profiler data in it, and therefore gives an error. see #TODO: Model.createProfiles uses ttem_model_idx. How about if no tTEM in it?
#TODO: do not plot borehole legend if no boreholes in the profile
#TODO: profiler and borehole elevation problems were fixed but weird data structure left behind
#TODO: Plot tTEM models with xm distance check---> check against actual tTEM map
#TODO: make profiler class specially thinking about the profileMap method
#TODO: when plotting sTEM or profiler data, the DOI is not very clear
#TODO: all profiles get a certain axis fontseize no matter their length, but then because the long ones do not fit into an A4 page, they need to 
# get squeezed and the fontsize looks small, which looks amateur on reports
#TODO: ask Andy the issue he wanted to resolve regarding the aspect ratio of the outputtet 
#TODO: when creating maps ask whether one wants it to be squared or not a part from whether one wants start, end and ticks (from Denys) 
#-------------------------------------------------INMEDIATE TODOS-----------------------------------------------


#-------------------------------------------------MIDTERM TODOS-----------------------------------------------
#TODO: embed the getAarhusColormap function properly within the matplotlib cmap integration and put the logarithmic ticks in the colorbar legend
#TODO: profiler.py:344: RuntimeWarning: something is dividing by NaN and consuming time in the Model.createProfiles, most likely the interpolation
#TODO: the interpotaling functions is too slow, and it consumes most of the time. see LLM Profiler_Inefficiency
#TODO: review the interpolation method. It can be both done faster and there needs to be an option for just grabbing the nearest neighbour to the profile
#TODO: The dynamic calculation of square, text size and spacing of Plot.addBoreholeLegend uses a single formula for all elements. Works well for 2 to 8 lithologies
#TODO: Fit the legend into a TEMprofile figure. Figure out the coordinate system to plot things on in a sandbox 
#TODO: allow to choose which DOI is to be plotted
#TODO: add a margin signature for the different sounding types and legend possibility
#TODO: give the possibility of choosing which legends/infos to output, like sounding type (tTEM, sTEM, profiler, borehole), projection, colorscale, DOI....
#TODO: else statement in model createProfiles regarding filetype doesnt make sense
#TODO: Model-CreateProfiles is called so in plural, but it seems you can only pass it one profile at a time? 
#TODO: in Plot.ProfileMap, add an option for choosing the color of the profile line with its ticks and the start and end of profile. Black is not always visible
#-------------------------------------------------MIDTERM TODOS-----------------------------------------------


#-------------------------------------------------DREAMY TODOS-----------------------------------------------
#Be able to load all of the data and draw the profiles yourself and order/store/manage them intuitively
#output an interactive window where you can follow the profile on the map
#Do this most of this web-based
#-------------------------------------------------DREAMY TODOS-----------------------------------------------