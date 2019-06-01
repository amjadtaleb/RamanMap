import RamanMap
from bokeh.io import curdoc
from bokeh import events
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, column, row, gridplot
from bokeh.models import Slider, ColumnDataSource, RadioGroup, CustomJS, Button, Plot, LinearAxis, Grid, Span, ColorBar, LinearColorMapper
from bokeh.models.tools import BoxZoomTool, ResetTool, SaveTool
from bokeh.models.glyphs import MultiLine
from bokeh.models.widgets import Div, Tabs, Panel
from bokeh.palettes import brewer
# =============================================================================
# ~ Initial Values
# =============================================================================
rm = RamanMap.RamanMap('info.txt') #instanciate and run RamanMap
dim = rm.dim
deviation_source = ColumnDataSource(data=dict(waves = rm.waves, deviations = rm.var))
deviation_sel_source = ColumnDataSource(data=dict(waves = rm.selected_peaks['waves'].values,
                      proms = rm.selected_peaks['prominences'].values,))

spectra_src = ColumnDataSource(data = rm.sliced_spectra)
maps_src = ColumnDataSource(data = rm.sliced_maps)
rm.prepare_training_data()
labeled_materials_data, pca_variance_data = rm.predictor()
labeled_materials = ColumnDataSource(data = labeled_materials_data)
pca_variance = ColumnDataSource(data = pca_variance_data)
user_input = ColumnDataSource(data=dict(x=[0.5], y=[0.5]))
spectra_available_multi = ColumnDataSource(data = {'ys':[i for i in rm.array]})
spectra_visible_multi = ColumnDataSource(data = dict(waves = [rm.waves], ys = [rm.array[0]]))
elbow_plot = ColumnDataSource(data = rm.predictorOptimizer(5))
# =============================================================================
# ~ plotting functions
# =============================================================================
def make_mapper(source, image_name):
    mapper_low, mapper_high = source.data[image_name][0].min(), source.data[image_name][0].max()
    mapper = LinearColorMapper(palette=brewer['RdYlBu'][11], low=mapper_low, high=mapper_high)
    return mapper

def basicMap(source, size = 400, image_name = ''):
    image = figure(plot_width=size,  plot_height = size, tooltips = [('index:', '@index')], tools=['reset', 'save','hover', 'wheel_zoom'], toolbar_location = 'below')
    image.axis.visible = False
    image.image(image=image_name ,source = source,x=0, y=0, dw=dim, dh=dim, palette=brewer['RdYlBu'][11])
    mapper = make_mapper(source, image_name)
    color_bar = ColorBar(color_mapper=mapper, location=(0,0))
    image.add_layout(color_bar, 'right')
    image.x_range.range_padding = image.y_range.range_padding = 0
    image.js_on_event(events.Tap, update_input_multi(user_input, spectra_visible_multi, spectra_available_multi, labeled_materials))
    image.rect('x', 'y',1,1, source = user_input,  fill_color = None, line_width = 2 , line_color = 'black',) #line_color = 'green',
    return image, mapper


def basicPlot(source, t):
    plot = figure(x_axis_label='Wavenumver (cm)', y_axis_label='Intensity (CPS)', plot_width=400,  plot_height = 400)
    plot.line('x', 'y_'+t+'_mx', color= 'red', legend = 'max', source = source)
    plot.line('x', 'y_'+t+'_md', color='black', legend = 'med', source = source)
    plot.line('x', 'y_'+t+'_mn', color='blue', legend = 'min', source = source)
    return plot
# =============================================================================
# ~ on_change functions
# =============================================================================
def peakSelector(attr, old, new):
    n = slider.value
    rm.sliceMap(n)
    spectra_src.data = rm.sliced_spectra
    maps_src.data = rm.sliced_maps
    span.location = spectra_src.data['x'].mean()
    g_mapper.low, g_mapper.high = maps_src.data['g_max'][0].min(), maps_src.data['g_max'][0].max()
    int_mapper.low, int_mapper.high = maps_src.data['intensities'][0].min(), maps_src.data['intensities'][0].max()

def repredict(attr, old, new):
    n = n_materials.active
    n = int(n_materials.labels[n])
    n_pca = n_pca_input.active
    n_pca = int(n_pca_input.labels[n_pca])
    labeled_materials.data, pca_variance.data = rm.predictor(n_mat = n,n_pca = n_pca)
    label_mapper.low, label_mapper.high =  labeled_materials.data['image'][0].min(), labeled_materials.data['image'][0].max()
    
def featureCounter(attr, old, new): #
    n_features = n_features_input.active
    n_features = int(n_features_input.labels[n_features])
    elbow_plot.data = rm.predictorOptimizer(n_features)

def update_input_multi(user_source, visible, available, lab):
    return CustomJS(args = dict(user_source = user_source, visible=visible, available = available, lab = lab), code ='''
            var data = user_source.data;
            var spectra = visible.data;
            var all_spectra = available.data;
            var labels = lab.data['image'][0];
            var x = parseInt(cb_obj['x']);
            var y = parseInt(cb_obj['y']);
            
            if((x >= 0) && (y >= 0) && (x < 50) && (y < 50)){
            var z = (x+1)+(y*50);
            console.log('x:' ,x, ' y:', y, ' z:', z);
            // check if the only available data is (0,0) then replace it with the first data point the user clicks
                if((data['x'].length == 1) && (data['x'] == 0.5) && (data['y'] == 0.5)){
                    data['x'] = [x+0.5];
                    data['y'] = [y+0.5];
                    spectra['ys'] = [all_spectra['ys'][z-1]];

                    visible.change.emit();                    //data values are shifted 0.5 to center rectangles on index
                    user_source.change.emit();
                    
                    }   else   {
                    data['x'].push(x+0.5);
                    data['y'].push(y+0.5);
                    
                    var wave = spectra['waves']['0'];
                    spectra['ys'].push(all_spectra['ys'][z-1]);
                    spectra['waves'].push(wave);
                    
                    visible.change.emit();
                    user_source.change.emit();
                }
            }
            ''')

def selectSpectra_reset():
    user_input.data = dict(x=[0.5], y=[0.5])
    spectra_visible_multi.data = dict(waves = [rm.waves], ys = [rm.array[0]])

# =============================================================================
# ~ Generate figures
# =============================================================================
r,g_mapper = basicMap(maps_src, 400, 'g_max')
s,int_mapper = basicMap(maps_src, 400, 'intensities')
p = basicPlot(spectra_src, 'd')
q = basicPlot(spectra_src, 'i')
# =============================================================================
# ~ Full spectral view
# =============================================================================
span = Span(location=spectra_src.data['x'].mean(), dimension='height', line_color='red',line_width=1)

dev = figure(x_axis_label='Wavenumver (cm)', y_axis_label='SD/avg (%)', plot_width=800, plot_height = 400,tools=['reset', 'save','hover', 'wheel_zoom', ] , y_axis_type="log")   # 'tap'
dev.line('waves', 'deviations', source = deviation_source)
dev.circle('waves', 'proms', color= 'red', source = deviation_sel_source)
dev.add_layout(span)

multi_plot = Plot(plot_width=800, plot_height=400)
multi_glyph = MultiLine(xs='waves', ys='ys', line_width=2)
multi_plot.add_glyph(spectra_visible_multi, multi_glyph)
xaxis , yaxis = LinearAxis(), LinearAxis()
multi_plot.add_layout(xaxis, 'below')
multi_plot.add_layout(yaxis, 'left')
multi_plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
multi_plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
multi_plot.add_tools(BoxZoomTool())
multi_plot.add_tools(ResetTool())
multi_plot.add_tools(SaveTool())
multi_plot.add_layout(span)
# =============================================================================
# ~~  Clustering figures
# =============================================================================
labeled_materials_image, label_mapper = basicMap(labeled_materials, 400, 'image')
elbow = figure(x_axis_label='Number of Materials', y_axis_label='Learning Inertia', plot_width=400, plot_height = 200, toolbar_location = None)
elbow.line('num_clusters', 'inertia', source = elbow_plot)
varBar = figure(plot_width=400, plot_height = 200, toolbar_location = None)
varBar.vbar(x = 'x', top = 'top', source = pca_variance, width = 0.9)
# =============================================================================
# ~ Interaction controls
# =============================================================================
slider = Slider(start=0, end=rm.n_peaks-1, value=0, step=1, title='Select a peak')
slider.on_change('value', peakSelector)

n_features_input = RadioGroup(labels=['2', '3', '5', '8', '13', '21'], active = 2, inline = True)
n_features_input.on_change('active', featureCounter)
n_features_input_title = Div(text="""# features """)
                             
n_pca_input = RadioGroup(labels=['1', '2', '3', '4', '5'], active = 1, inline = True)
n_pca_input.on_change('active', repredict)
n_pca_input_title = Div(text="""# pca:  """)

n_materials = RadioGroup(labels=['2', '3', '4', '5', '6', '7', '8', '9'], active = 1, inline = True)
n_materials.on_change('active', repredict)
n_materials_title = Div(text="""# Materials """)

btn_rst = Button(label = 'reset')
btn_rst.on_click(selectSpectra_reset)
# =============================================================================
# ~ Layout setup
# =============================================================================
div1 = Div(text="""<h2>dw</h2>""", )
div2 = Div(text="""<h2>I</h2>""", )
#blank = Div(text="""  """, width=400)
row_feature = row([n_features_input_title,  widgetbox(n_features_input)])
row_pca = row([n_pca_input_title,  widgetbox(n_pca_input)])
row_materials =  row(n_materials_title,  widgetbox(n_materials))

grid_raw = gridplot([ [div1, r, p],  [div2, s, q]  ])

grid_cluster = row(  column([elbow, row_feature , varBar, row_pca]), 
                 column([row_materials, labeled_materials_image, btn_rst])  )

a = Panel(child=dev, title='Variance per wavelength')
b = Panel(child=multi_plot, title='Spectra per pixel')

right_panel = column([widgetbox(slider),grid_raw])
left_panel = column([grid_cluster, Tabs(tabs = [a,b])])
layout = row(left_panel, right_panel)


curdoc().title ='Analysis of file '+ rm.file.split('\\\\')[-1]
curdoc().add_root(layout)