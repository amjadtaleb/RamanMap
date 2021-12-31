import RamanMap
from bokeh.io import curdoc
from bokeh import events
from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.models import Slider, ColumnDataSource, RadioGroup, CustomJS, Button, Plot, LinearAxis, Grid, Span, ColorBar, LinearColorMapper, Column
from bokeh.models.tools import BoxZoomTool, ResetTool, SaveTool, WheelZoomTool
from bokeh.models.glyphs import MultiLine
from bokeh.models.widgets import Div, Tabs, Panel
from bokeh.palettes import brewer
# =============================================================================
# ~ Initial Values
# =============================================================================
# instanciate RamanMap
# rm = RamanMap.RamanMap('Graphene_Map_RamanSemrock_50x50points_20um2_1s_2.asc')
rm = RamanMap.RamanMap('TianzeHu.txt')
# =============================================================================
# prepare ColumnDataSources
dim = rm.dim
deviation_source = ColumnDataSource(
    data=dict(waves=rm.waves, deviations=rm.mean))
# rm.selectPeaks()
deviation_sel_source = ColumnDataSource(data=dict(waves=rm.selected_waves,
                                                  proms=rm.selected_peaks['peak_heights'].values))
spectra_src = ColumnDataSource(data=rm.sliced_spectra)
maps_src = ColumnDataSource(data=rm.sliced_maps)
# run the in following as ordered since prepare_training_data() which returns
# training_data runs inside predictorOptimizer() and not predictor()
rm.predictorOptimizer(5)
# elbow_plot = ColumnDataSource(data=
# )
labeled_materials_data, pca_variance_data = rm.predictor()
labeled_materials = ColumnDataSource(data=labeled_materials_data)
pca_variance = ColumnDataSource(data=pca_variance_data)

user_input = ColumnDataSource(data=dict(x=[0.5], y=[0.5]))
spectra_available_multi = ColumnDataSource(
    data={'ys': [i for i in rm.array], 'labels': labeled_materials_data['image'][0].flatten()})
#labels is redundant in spectra_available_multi
spectra_visible_multi = ColumnDataSource(data=dict(waves=[rm.waves], ys=[
                                         rm.array[0]], label=[brewer['RdYlBu'][rm.n_mat][0]], index=[0]))
# ColumnDataSource allows updating colors of user-slected spectra
palette = brewer['RdYlBu']
palette[2] = [brewer['RdYlBu'][3][0], brewer['RdYlBu'][3][2]]
palette_sr = ColumnDataSource(data={'colors': palette[rm.n_mat]})
# =============================================================================
# ~ plotting functions
# =============================================================================


def make_mapper(source, image_name):
    '''
    Prepare color mapper for basicMap()
    returns mapper
    '''
    mapper_low, mapper_high = source.data[image_name][0].min(
    ), source.data[image_name][0].max()
    mapper = LinearColorMapper(
        palette=brewer['RdYlBu'][11], low=mapper_low, high=mapper_high)
    return mapper
# =============================================================================


def basicMap(source, size=400, image_name=''):
    '''
    Prepare bokeh.image for map plotting
    Input: source is ColumnDataSource
           size: dimention of ouput image in screen pixels
           image_name: dictionary key name of the image in CDS.data
    Example: basicMap(maps, 400, 'intensities')
    returns image and color mapper
    '''
    image = figure(plot_width=size,  plot_height=size, tooltips=[
                   ('index:', '@index')], tools=['reset', 'save', 'hover', 'wheel_zoom'], toolbar_location='below')
    image.axis.visible = False
    image.image(image=image_name, source=source, x=0, y=0,
                dw=dim, dh=dim, palette=brewer['RdYlBu'][11])
    mapper = make_mapper(source, image_name)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
    image.add_layout(color_bar, 'right')
    image.x_range.range_padding = image.y_range.range_padding = 0
    image.js_on_event(events.Tap, update_input_multi(
        user_input, spectra_visible_multi, spectra_available_multi, labeled_materials, palette_sr))
    image.rect('x', 'y', 1, 1, source=user_input,  fill_color=None,
               line_width=2, line_color='black',)  # line_color = 'green',
    return image, mapper
# =============================================================================


def basicPlot(source, t):
    '''
    Prepare bokeh plot for multiple lines in a the rm.Spectra datasource
    returns the plot object
    '''
    plot = figure(x_axis_label='Wavenumver (nm)',
                  y_axis_label='Intensity (CPS)', plot_width=400,  plot_height=400)
    plot.line('x', 'y_'+t+'_mx', color='red',
              legend_label='max', source=source)
    plot.line('x', 'y_'+t+'_md', color='black',
              legend_label='med', source=source)
    plot.line('x', 'y_'+t+'_mn', color='blue',
              legend_label='min', source=source)
    return plot
# =============================================================================
# ~ on_change functions
# =============================================================================


def peakSelector(attr, old, new):
    '''
    Change the shown maps, their correspoding spectra and color mappers
    by changing the ColumnDataSources and high-low values of mappers
    also changes position of the red vertical line Span on spectra figues
    '''
    n = slider.value
    rm.sliceMap(n)
    spectra_src.data = rm.sliced_spectra
    maps_src.data = rm.sliced_maps
    span.location = spectra_src.data['x'].mean()
    g_mapper.low, g_mapper.high = maps_src.data['g_max'][0].min(
    ), maps_src.data['g_max'][0].max()
    int_mapper.low, int_mapper.high = maps_src.data['intensities'][0].min(
    ), maps_src.data['intensities'][0].max()
# =============================================================================


def repredict(attr, old, new):
    '''
    Rerun Machine Learning methods and change CDS for the predicted image and
    the corresponding color mapper high-low values

    Future objective: update colors of selected spectra in spectra_visible_multi
    '''
    n = n_materials.active
    n = int(n_materials.labels[n])
    palette_sr.data = {'colors': palette[n]}
    n_pca = n_pca_input.active
    n_pca = int(n_pca_input.labels[n_pca])
    labeled_materials.data, pca_variance.data = rm.predictor(
        n_mat=n, n_pca=n_pca)
    label_mapper.low, label_mapper.high = labeled_materials.data['image'][0].min(
    ), labeled_materials.data['image'][0].max()
#    still testing to update multiline colors in this callback function
#    vis_ys = spectra_visible_multi.data['ys']
#    print(user_input.data)
#    vis_waves = spectra_visible_multi.data['waves']
#    vis_index = spectra_visible_multi.data['index']
#    vis_labels = [palette[n][i] for i in labeled_materials.data['image'][0].flatten()[vis_index]]
#    print(vis_labels)
#    spectra_visible_multi.data = dict(waves = vis_waves, ys = vis_ys, label= vis_labels, index= vis_index)

# =============================================================================


# def featureCounter(attr, old, new):
#     '''
#     Change number of features to use in Machine Learning
#     reproduce learning inertia and update predicted image with colormapper
#     '''
#     n_features = n_features_input.active
#     if rm.x == True:  # no effect for now, it is here to switch g_max on/off in training data
#         # currently this scales or not training data PCA
#         n_features = int(n_features_input.labels[n_features])*2
#     else:
#         n_features = int(n_features_input.labels[n_features])*2
#     elbow_plot.data = rm.predictorOptimizer(n_features)
#     labeled_materials.data, pca_variance.data = rm.predictor(
#         n_mat=rm.n_mat, n_pca=rm.n_pca)
#     label_mapper.low, label_mapper.high = labeled_materials.data['image'][0].min(
#     ), labeled_materials.data['image'][0].max()

# =============================================================================


def update_input_multi(user_source, visible, available, lab, col):
    '''
    Reads tapped index from basicMap objects, update drawn rectangels
    update selected visible spectra
    color visible spectra according to labels if predicted image
    '''
    return CustomJS(args=dict(user_source=user_source, visible=visible, available=available, lab=lab, col=col, dim=dim), code='''
            var data = user_source.data;
            var spectra = visible.data;
            var all_spectra = available.data;
            var labels = lab.data['image'][0];
            var cols = col.data['colors']
            
            //const n_mat = n_mat;
            //console.log(cols);
            const x = parseInt(cb_obj.x);
            const y = parseInt(cb_obj.y);
            
            if((x >= 0) && (y >= 0) && (x < dim) && (y < dim)){
            const z = (x+1)+(y*dim);
            const label = labels[z-1];
            
            // check if the only available data is (0,0) then replace it with the first data point the user clicks
                if((data['x'].length == 1) && (data['x'] == 0.5) && (data['y'] == 0.5)){
                    data['x'] = [x+0.5];
                    data['y'] = [y+0.5];
                    spectra['ys'] = [all_spectra['ys'][z-1]];
                    spectra['label'] = [cols[label]];
                    spectra['index'] = [z]

                    visible.change.emit();                    //data values are shifted 0.5 to center rectangles on index
                    user_source.change.emit();
                    
                    }   else   {
                    data['x'].push(x+0.5);
                    data['y'].push(y+0.5);
                    
                    var wave = spectra['waves']['0'];
                    spectra['ys'].push(all_spectra['ys'][z-1]);
                    
                    spectra['waves'].push(wave);
                    spectra['label'].push(cols[label]);
                    spectra['index'].push(z);
                    
                    // console.log('label:', label);
                    
                    visible.change.emit();
                    user_source.change.emit();
                }
            }
            ''')
# =============================================================================


def selectSpectra_reset():
    '''
    Reset user selections: remove selected rectangles on maps and visible spectra
    '''
    user_input.data = dict(x=[0.5], y=[0.5])
    spectra_visible_multi.data = dict(waves=[rm.waves], ys=[rm.array[0]], label=[
                                      brewer['RdYlBu'][rm.n_mat][0]])


# =============================================================================
# ~ Generate figures
# =============================================================================
r, g_mapper = basicMap(maps_src, 400, 'g_max')
s, int_mapper = basicMap(maps_src, 400, 'intensities')
p = basicPlot(spectra_src, 'd')
q = basicPlot(spectra_src, 'i')
# =============================================================================
# ~ Full spectral view
# ~ span highleights position of shown map with a vertical line
# ~ mean shows the mean of all spectra with highleighted selected peaks as red dots
# ~ multi_plot shows user-selected spectra
# =============================================================================
span = Span(location=spectra_src.data['x'].mean(
), dimension='height', line_color='red', line_width=1)
# =============================================================================
mean = figure(x_axis_label='Wavenumver (nm)', y_axis_label='Intensity (CPS/pixels)', plot_width=800,
              plot_height=400, tools=['reset', 'save', 'hover', 'wheel_zoom', ], y_axis_type="log")   # 'tap'
mean.line('waves', 'deviations', source=deviation_source)
mean.circle('waves', 'proms', color='red', source=deviation_sel_source)
mean.add_layout(span)
# =============================================================================
multi_plot = Plot(plot_width=800, plot_height=400,
                  background_fill_color='silver')
multi_glyph = MultiLine(xs='waves', ys='ys',  line_width=2, line_color='label')
multi_plot.add_glyph(spectra_visible_multi, multi_glyph)
xaxis, yaxis = LinearAxis(), LinearAxis()
multi_plot.add_layout(xaxis, 'below')
multi_plot.xaxis.axis_label = 'Wavelength (nm)'
multi_plot.add_layout(yaxis, 'left')
multi_plot.yaxis.axis_label = 'Intensity (CPS)'
multi_plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
multi_plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
multi_plot.add_tools(BoxZoomTool())
multi_plot.add_tools(WheelZoomTool())
multi_plot.add_tools(ResetTool())
multi_plot.add_tools(SaveTool())
multi_plot.add_layout(span)
# =============================================================================
# ~~  Clustering figures
# ~  labeled_materials_image: map image of predicted material labels
# ~  elbow inertia vs n_materials plot
# ~  varBar bar plot of pca explained variances
# =============================================================================
labeled_materials_image, label_mapper = basicMap(
    labeled_materials, 400, 'image')
# elbow = figure(x_axis_label='Number of Materials', y_axis_label='Learning Inertia',
#                plot_width=400, plot_height=200, toolbar_location=None)
# elbow.line('num_clusters', 'inertia', source=elbow_plot)
varBar = figure(plot_width=400, plot_height=200, toolbar_location=None)
varBar.vbar(x='x', top='top', source=pca_variance, width=0.9)
# =============================================================================
# ~ Interaction controls
# ~ slider: choose slice of maps and spectra
# ~ n_features: limit total number of features in ML analysis
# ~ n_pca: limit number of Principal Components in ML analysis
# ~ n_materials: choose number of predicted materisl
# ~ btn_rst: reset user selection of visible spectra
# =============================================================================
slider = Slider(start=0, end=rm.n_peaks-1, value=0,
                step=1, title='Select a peak')
slider.on_change('value', peakSelector)

# n_features_input = RadioGroup(
#     labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], active=2, inline=True)
# n_features_input.on_change('active', featureCounter)
# n_features_input_title = Div(text="""# features """)

n_pca_input = RadioGroup(
    labels=['1', '2', '3', '4', '5'], active=1, inline=True)
n_pca_input.on_change('active', repredict)
n_pca_input_title = Div(text="""# pca:  """)

n_materials = RadioGroup(
    labels=['2', '3', '4', '5', '6', '7', '8', '9'], active=1, inline=True)
n_materials.on_change('active', repredict)
n_materials_title = Div(text="""# Materials """)

btn_rst = Button(label='reset')
btn_rst.on_click(selectSpectra_reset)
# =============================================================================
# ~ Layout setup
# =============================================================================
div1 = Div(text="""<h2>dw</h2>""", )
div2 = Div(text="""<h2>I</h2>""", )
#blank = Div(text="""  """, width=400)
# row_feature = row([n_features_input_title,  Column(n_features_input)])
row_pca = row([n_pca_input_title,  Column(n_pca_input)])
row_materials = row(n_materials_title,  Column(n_materials))

grid_raw = gridplot([[div1, r, p],  [div2, s, q]])

grid_cluster = row(column([varBar, row_pca]),
                   column([row_materials, labeled_materials_image, btn_rst]))

a = Panel(child=mean, title='Average Spectrum')
b = Panel(child=multi_plot, title='Spectra per pixel')

right_panel = column([Column(slider), grid_raw])
left_panel = column([grid_cluster, Tabs(tabs=[a, b])])
layout = row(left_panel, right_panel)


curdoc().title = 'Analysis of file ' + rm.file.split('\\\\')[-1]
curdoc().add_root(layout)
