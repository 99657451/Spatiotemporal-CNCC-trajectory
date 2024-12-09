# author    : zhangzhao (zhangzhao3@genomics.cn)
# date      : 2024/01/10
# version   : 0.03 now we have logger
# last change log: rewrite some functions
# single & spatial transcriptomics data analysis toolkit (s&sat)

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Sequence, Callable, Literal
from functools import partial
from itertools import chain, combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colorbar import Colorbar
from matplotlib.patheffects import Stroke, Normal
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize

from plotnine import * # deprecated

import marsilea as ma
import marsilea.plotter as mp
import seaborn as sns

import numpy as np
import pandas as pd
import geopandas as gpd # geo support
from geopandas import GeoDataFrame, GeoSeries

from adjustText import adjust_text

from anndata import AnnData

# TODO verbose(log) numba annotation debug
# TODO global set group_by like seurat Idents
# TODO future feature: Preprocessing recipe (preprocessor class, pipeline)


# ------------------------------logging-------------------------------------
from loguru import logger
from rich.logging import RichHandler
import sys

class LoggerHelper:
    def __init__(self, min_level='INFO'):

        self.logger = logger
        self.min_level = min_level
        self.set_level()
        self.config()
        
    def _filter(self, record):
        return record["level"].no >= logger.level(self.min_level).no
    
    @staticmethod
    def formatter(record):
        if record["level"].no <= 20:
            return "📃  {message}"
        elif record["level"].no <= 30:
            return "❕  {message}"
        else:
            return "{message}"

    def set_level(self):
        self.logger.remove()
        self.logger.add(sys.stderr, filter=self._filter)
        
    def config(self, **kwargs):
        kwargs.setdefault(
            'handlers', 
            [{"sink": RichHandler(
                markup=True, log_time_format='[%X]', 
                show_level=False, show_path=True),
              "format": self.formatter,
             }]
            )
        
        self.logger.configure(**kwargs)

Logger = LoggerHelper()
logger = Logger.logger  


# ------------------------------ style -------------------------------------
# TODO rich text
# TODO add colorbar artist some time
# TODO add some preprocessing recipe (this require scanpy)
#? support input multi adata?
MPL_RC = {
    'font.family':'Arial',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.linewidth': 1.2,
    'axes.spines.right' : False,
    'axes.spines.top' : False,
    "axes.axisbelow": False,
    "patch.force_edgecolor": False,
    'lines.linewidth': 1.5,
    'legend.labelspacing': 0.58, # 0.5
    'legend.columnspacing': 1.5, # 2
    'legend.title_fontsize': 'medium',
    'axes.labelsize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'axes.titlesize' : 'medium',
    'legend.fontsize': 'medium',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0,
    'ytick.minor.size': 2.0,
    'xtick.major.width': 1.125,
    'ytick.major.width': 1.125,
    'xtick.major.pad': 2,
    'xtick.minor.pad': 1.9,
    'ytick.major.pad': 2,
    'ytick.minor.pad': 1.9,
    "axes.labelcolor": 'k',
    "xtick.color": 'k',
    "ytick.color": 'k',
    "text.color": 'k',
    'figure.dpi': 80,
}

sns.set_theme(
    context='notebook', 
    style='ticks', 
    palette='muted', 
    font='sans-serif', 
    font_scale=1., 
    color_codes=True, 
    rc=MPL_RC
)

# next version
# change pram 
# * split_by to splitby
# * dim_plot to dimplot
# * vln_plot to vlnplot
# * dot_plot to dotplot
# * volcano_plot to volcanoplot
# * ridge_plot to ridgeplot

# ------------------------------ color -------------------------------------

# TODO 
_PALETTES = dict(
    google_4     = ['#3D79F3FF', '#E6352FFF', '#F9B90AFF', '#34A74BFF'],
    Darjeeling_5 = ['#FF0000FF', '#00A08AFF', '#F2AD00FF', '#F98400FF', '#5BBCD6FF'],
    Moonrise_5   = ['#85D4E3FF', '#F4B5BDFF', '#9C964AFF', '#CDC08CFF', '#FAD77BFF'],
    pony9        = ['#EB5291FF', '#FBBB68FF', '#F5BACFFF', '#9DDAF5FF', '#6351A0FF', 
                    '#ECF1F4FF', '#FEF79EFF', '#1794CEFF', '#972C8DFF'],
    
    # https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
    tab10        = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948',
                    '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'],
    nord_10      = ['#8fbcbb', '#88c0d0', '#81a1c1', '#5e81ac', '#bf616a', '#d08770', 
                    '#ebcb8b', '#a3be8c', '#b48ead', '#2e3440'],
    zhang15      = ['#EFE2AA', '#83B4EF', '#DBC9B3', '#8ECFF8', '#7EC0C3', '#EED0E0', 
                    '#EBAEA9', '#AAD0AC', '#95A6DA', '#BFA6C9', '#F5E0BA', '#AED0DF',
                    '#89B780', '#F5D8D0', '#CB95BB'],
    qu_16        = ['#A17CAF', '#D14827', '#CE6463', '#5C4595', '#F6E7AD', '#E9A6C6', 
                    '#4D8A9C', '#ACB566', '#C47DA1', '#2E60A9', '#6A8C3B', '#F9D232',
                    '#5A77AE', '#C9B9D5', '#DE9B53', 'lightgrey', 'grey'],
    Echart_16    = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#9a60b4", "#73c0de", 
                    "#3ba272", "#fc8452", "#27727b", "#ea7ccc", "#d7504b", "#e87c25", 
                    "#b5c334", "#fe8463", "#26c0c0", "#f4e001"],
    Tailwind_20  = ["#1d4ed8", "#60a5fa", "#c2410c", "#fb923c", "#15803d", "#86efac", 
                    "#b91c1c", "#f87171", "#7e22ce", "#c084fc", "#a16207", "#facc15", 
                    "#be185d", "#f472b6", "#374151", "#9ca3af", "#4d7c0f", "#a3e635", 
                    "#0369a1", "#38bdf8"],
    zeileis_28   = ["#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b",
                    "#4a6fe3", "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a",
                    "#11c638", "#8dd593", "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708",
                    "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4",
                     # these last ones were added:
                    '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600"],
    ting_36     =  ['#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', 
                    '#476D87', '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', 
                    '#8C549C', '#585658', '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', 
                    '#58A4C3', '#E4C755', '#F7F398', '#AA9A59', '#E63863', '#E39A35', 
                    '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B', '#712820', '#DCC1DD', 
                    '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963','#968175']
)

# set value when value is not None else set default
def set_value(value, default):
    """set value when value is not None else set default"""
    return default if value is None else value

def _is_run_from_ipython():
    """Determines whether we're currently in IPython."""
    import builtins
    return getattr(builtins, "__IPYTHON__", False)

if _is_run_from_ipython():
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')

def register_colormap(name, cmap):
    """Handle changes to matplotlib colormap interface in 3.6."""
    try:
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    except AttributeError:
        mpl.cm.register_cmap(name, cmap)
        
# register some color 
COLORMAPS = dict(
    roma = 'roma.txt',
    parula = 'parula.txt',
    # TODO add more
)

for name, filename in COLORMAPS.items():
    v = np.loadtxt(filename)
    cmap = mpl.colors.ListedColormap(v, name=name)
    register_colormap(name, cmap)
    register_colormap(name + '_r', cmap.reversed())
    
register_colormap('lightr', sns.color_palette("light:r", as_cmap=True))
register_colormap('lightb', sns.color_palette("light:b", as_cmap=True))

# show palette
def show_palette(size=0.5):
    for name, palette in _PALETTES.items():
        sns.palplot(palette, size=size)
        ax = plt.gca()
        ax.axis('off')
        plt.title(name)

        
# ------------------------------ param -------------------------------------
# set default param
# next vesion will 
# * kwargs: 
#   * groupby: str indentify the groupby column
#   * reduction: str indentify the reduction method
# (key_name, default_value)

GROUPBY = 'groupby'
SPLITBY = 'splitby'
REDUCTION = 'reduction'

SSAT_PARAMS = dict(
    tool_param = {},
    plot_param = {
        # GROUPBY: None,
        # SPLITBY: None,
        # REDUCTION: None
    },
)

def register(adata, **params):
    """ santilize adata """
    if 'ssat' not in adata.uns:
        logger.warning('ssat is not registered, register now')
        adata.uns['ssat'] = SSAT_PARAMS
        
    if params:
        adata.uns['ssat'].update(params)
        
# register decorator
# TODO context manager
def register_decorator(func):
    """ register decorator """
    
    def wrapper(*args, **kwargs):
        adata = args[0]
        register(adata)
        return func(*args, **kwargs)
    
    return wrapper


def get_ident(adata, groupby=None):
    if groupby is None:
        try:
            groupby = adata.uns['ssat']['plot_param'][GROUPBY]
        except IndexError:
            return None, None
        
    return groupby, adata.obs[groupby]
        
        
@register_decorator
def set_indent(adata, ident: Union[str, np.ndarray, pd.Series]):
    """ set indent """
    register(adata)
    
    logger.info(f'set indent {ident}')
    if isinstance(ident, str):
        if is_cell_anno(adata, ident):
            ident = ident
        else:
            raise ValueError(
                f'{ident} is not a valid cell annotation')
    else:
        assert len(adata) == len(ident), 'indent length must be equal to adata length'
        logger.info('Update `adata.obs`')
            
        if isinstance(ident, np.ndarray):
            adata.obs['_Ident'] = ident
            ident = '_Ident'
        # series column name
        elif isinstance(ident, pd.Series):
            add_metadata(adata, ident)
            ident = ident.name
        else:
            raise ValueError(
                f'{ident} is not a valid indent')
    
    adata.uns['ssat']['plot_param'][GROUPBY] = ident
    
    
@register_decorator   
def reset_indent(adata):
    """ reset indent """
    adata.uns['ssat']['plot_param'][GROUPBY] = None

# TODO
# def rename_indent(adata, mapping: Dict[str, str], groupby=None):
    
    
# TODO
def set_reduction(adata, reduction: Union[str, np.ndarray, pd.DataFrame]):
    """ set reduction """
    register(adata)
    
    if isinstance(reduction, str):
        pass

# --------------------------------------------------------------------------
#                           plotting utilities
# --------------------------------------------------------------------------

def new_axis(ax: Axes, position: str = 'top', 
             size: str = '3%', pad: float = 0, 
             sharex=None, sharey=None) -> Axes: 
    """ create a new axis """
    
    # function takes an existing axes, adds it to a new AxesDivider
    divider = make_axes_locatable(ax)
    # Add an axes above the main axes. 
    # Note: Higher the value for the pad 
    # the colorbar is away from the x-axis
    cax = divider.append_axes(position, size=size, pad=pad, 
                              sharex=sharex, sharey=sharey)
    return cax


# class LegendTitle(object):
#     def __init__(self, text_props=None):
#         self.text_props = text_props or {}
#         super(LegendTitle, self).__init__()

#     def legend_artist(self, legend, orig_handle, fontsize, handlebox):
#         x0, y0 = handlebox.xdescent, handlebox.ydescent
#         width, height = handlebox.width, handlebox.height
#         title = Text(x0, y0, orig_handle, **self.text_props)
#         handlebox.add_artist(title)
#         return title

class TitleStr(str):
    def __new__(cls, t, bbox_ec=None, bbox_fc=None):
        obj = str.__new__(cls, t)
        obj.bbox_ec = bbox_ec
        obj.bbox_fc = bbox_fc
        return obj

# TODO: beta version
class LegendSubtitleHandle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendSubtitleHandle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        
        prop = FontProperties(size=fontsize)
        fontsize = prop.get_size_in_points()
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        facecolor=set_value(orig_handle.bbox_fc, '.9')
        edgecolor=set_value(orig_handle.bbox_ec, 'none')
  
        dpi = mpl.rcParams['figure.dpi']
        spacing = mpl.rcParams['legend.labelspacing'] * fontsize
        borderpad = mpl.rcParams['legend.borderpad'] * fontsize
        
        width = fontsize * 6.5
        height = fontsize * 1.1 

        # add boxr
        box = Rectangle(
            xy=(x0 + borderpad, y0 - spacing / 2), 
            facecolor=facecolor, edgecolor=edgecolor,
            width=width, height=height, 
            transform=handlebox.get_transform(),
            capstyle='projecting', antialiased=False)
        box.set_alpha(0.8)
        handlebox.add_artist(box)
        
        # add title
        title = Text(
            width / 2 + borderpad, (height - spacing) / 2, 
            orig_handle, va='center', ha='center' , **self.text_props)
        
        handlebox.add_artist(title)
        
        return title


class HandlerRect(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        x = width//2
        y = 0
        w = h = 10

        # create
        p = Rectangle(xy=(x, y), width=w, height=h)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]
    
    
class BOX:
    def __init__(self, 
                 facecolor='none', 
                 edgecolor='black'):
        
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        

class BoxplotHandler:
    # adapted from plotnine
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        # add box
        box = Rectangle([width * .125 - x0, height * .25 - y0], 
                        width=width *.75, height=height*.5, 
                        facecolor=orig_handle.facecolor, 
                        edgecolor=orig_handle.edgecolor,
                        capstyle='projecting', antialiased=False,
                        lw=0.5, transform=handlebox.get_transform())
        handlebox.add_artist(box)
        
        
        lw=0.75
        # add strike line
        strike = Line2D([width*.125 - x0, width*.875 - x0], 
                        [height*.5 - y0,  height*.5 - y0],
                        color=orig_handle.edgecolor,
                        lw=lw, solid_capstyle='butt')
        handlebox.add_artist(strike)
        
        # add top line
        top = Line2D([width*.5 - x0, width*.5 - x0], 
                     [height*.75 - y0, height*.9 - y0],
                     color=orig_handle.edgecolor,
                     lw=lw, solid_capstyle='butt')
        handlebox.add_artist(top)
        
        # add bottom line
        bottom = Line2D([width*.5 - x0, width*.5- x0],
                        [height*.25- y0, height*.1 - y0],
                        color=orig_handle.edgecolor,
                        lw=lw, solid_capstyle='butt')
        
        handlebox.add_artist(bottom)
        return handlebox
    

# TODO maybe merge with colorbar
# TODO markerscale adapt to plot size
def plot_legend(
    ax:Axes=None, fig:Figure=None, subtitle=None, 
    subtitle_fcs=None, subtitle_ec=None,
    handles=None, labels=None, title=None, 
    num_in_legend=False, figsize=(5, 5), **kwarg
): 
    """plot legend"""
    
    #TODO location

    scale_ratio = min(figsize) / 5
    size = (200 * scale_ratio 
            if num_in_legend 
            else 120 * scale_ratio)
    
    kw = dict(bbox_to_anchor=(1, 0.5), 
              loc='center left', 
              markerscale=size,
              fontsize='medium', 
              frameon=False,
            #   labelspacing=0.55,
              title=None,
              number_size=8.328 ** 2 * scale_ratio,
              number_color='white',
              handler_map={},
              subtitle_fontsize='medium',
              title_align=None)
    
    kw.update(kwarg)
    number_size       = kw.pop('number_size')
    number_color      = kw.pop('number_color')
    handler_map       = kw.pop('handler_map')
    markerscale       = kw.pop('markerscale')
    subtitle_fontsize = kw.pop('subtitle_fontsize')
    title_align       = kw.pop('title_align')
    
    if fig is None and ax is None:
        axs = plt.gcf().axes
    elif fig is not None:
        axs = fig.axes
    elif ax is not None:
        axs = [ax]
        
    if len(axs) == 1:
        obj = axs[0]
    else:
        obj = fig

    def fig_legend(axs):
        # ref: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
        # Generate a sequence of tuples, each contains
        #  - a list of handles (lohand) and
        #  - a list of labels (lolbl)
        tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in axs)
        # E.g., a figure with two axes, ax0 with two curves, ax1 with one curve
        # yields:   ([ax0h0, ax0h1], [ax0l0, ax0l1]) and ([ax1h0], [ax1l0])

        # The legend needs a list of handles and a list of labels,
        # so our first step is to transpose our data,
        # generating two tuples of lists of homogeneous stuff(tolohs), i.e.,
        # we yield ([ax0h0, ax0h1], [ax1h0]) and ([ax0l0, ax0l1], [ax1l0])
        tolohs = zip(*tuples_lohand_lolbl)

        # Finally, we need to concatenate the individual lists in the two
        # lists of lists: [ax0h0, ax0h1, ax1h0] and [ax0l0, ax0l1, ax1l0]
        # a possible solution is to sum the sublists - we use unpacking
        handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
        # remvoe duplicate labels
        handles_labels_dict = dict(zip(labels, handles))
        handles, labels = handles_labels_dict.values(), handles_labels_dict.keys()
        return handles, labels
        
    handles_labels = fig_legend(axs)
    handles = handles_labels[0] if handles is None else handles 
    labels  = handles_labels[1] if labels  is None else labels
    handler_map.update(
        {BOX: BoxplotHandler()})
    
    # set fixed size for legend
    # generate fake numbers for legend
    if num_in_legend:
        handles_num = [plt.scatter([], [], marker=r"$\mathcal{%s}$" %i , 
                                   s=len(str(i)) * number_size, 
                                   c=number_color, lw=0)
                       for i in range(len(handles))]
        handles = list(zip(handles, handles_num))
        
    
    if subtitle is not None:
        handles_dict = {}
        
        if isinstance(subtitle, list):
            if len(subtitle) != len(labels):
                raise ValueError(
                    'subtitle length must be equal to labels length')
            
            # input [1, 2, 3, 4, 5]  ['2', '2', '1', '1', '3']
            # output {'2': ['1', '2'], '1': ['3', '4'], '3': ['5']}
            subtitle_dict = {}
            for i, j in zip(subtitle, zip(handles, labels)):
                handles_dict.setdefault(str(i), []).append(j[0]) 
                subtitle_dict.setdefault(str(i), []).append(j[1])
                
        elif isinstance(subtitle, dict):  # {'sub_title': 'labels'}
            handles_labels_dict = dict(zip(labels, handles)) # {'labels': 'handles'}
            subtitle_dict = {
                i: [j] if isinstance(j, str) else j 
                for i, j in subtitle.items()
                }  # {'subtitle': ['label1']}
            handles_dict = {
                i: [handles_labels_dict[k] for k in j] 
                for i, j in subtitle_dict.items()
                }  # {'sub_title': [handles1, handles2]}
        else:
            raise ValueError('subtitle must be list or dict')
        
        subtitle_fcs = set_value(subtitle_fcs, [None] * len(subtitle_dict))
        handles = list(chain(*[list(chain([TitleStr(i, subtitle_ec, fc)], *[j])) 
                               for (i, j), fc in zip(handles_dict.items(), subtitle_fcs)]))
        labels  = list(chain(*[list(chain([''], *[j])) for _, j in subtitle_dict.items()]))
        handler_map.update({str: LegendSubtitleHandle({'fontsize': subtitle_fontsize})})
        
    kw.setdefault('ncol', (1 if len(labels) <= 14 else 2 if len(labels) <= 30 else 3))
    
    lgd = obj.legend(
        handles=handles,
        labels=labels,
        # markerscale=3, 
        handler_map=handler_map,
        **kw
        ) # borderpad=1.5, labelspacing=1.5 handletextpad=0.1
    
    if title is not None:
        lgd.set_title(title, prop={'size': 'medium'})
    
    # legend title align
    if title_align in ['left', 'right']:
        lgd._legend_box.align = title_align
    
    for legend_handle in lgd.legend_handles:
        try:
            legend_handle.set_sizes([markerscale]) 

        except AttributeError:
            pass
        
    return lgd


# TODO: ticks labels 
def set_colorbar(cb: Colorbar, **kw):  # you cannot hope one function to do everything 
    """ set colorbar """
    
    theme       = kw.get('theme', 'ggplot')
    orientation = kw.get('orientation', 'vertical')
    position    = kw.get('position', None)
    # location = kw.get('location', None)
    ticks       = kw.get('ticks', None)
    labels      = kw.get('labels', None)
    rasterized  = kw.get('rasterized', True)
    # fontsize    = kw.get('fontsize', 'medium')
    title       = kw.get('title', '')
    title_style = kw.get('title_style', 'normal')
    title_align = kw.get('title_align', 'center')
    edgecolor   = kw.get('edgecolor', 'none')
    tickscolor  = kw.get('tickscolor', 'white')
    
    cb.outline.set_edgecolor(edgecolor)
    cb.outline.set_linewidth(0.8)
    window_extent = cb.ax.get_window_extent()
    length = 4.5 * window_extent.width / 35
    
    ggplot_params = dict(
        which='both', 
        direction='in',
        length=length, 
        width=.8,
        color=tickscolor, 
        labelsize='small',
        left=True, 
        pad=2.5
    )
    
    ticks_params = dict(
        ticks=ticks,
        labels=labels,
        fontsize='small'
    )
    
    min_max_params = dict(
        ticks=[cb.vmin, cb.vmax], 
        labels=['min', 'max'],
        fontsize='small'
    )
    
    if theme == 'ggplot':
        if orientation == 'horizontal':
            ggplot_params.update(
                {'left': False, 'top': True})
            
        cb.ax.tick_params(**ggplot_params)
        cb.ax.locator_params(nbins=4) 
        
    else:
        cb.ax.tick_params(size=0, pad=2.5)
        
        if theme == 'min_max':
            ticks_params.update(min_max_params)
            
            if orientation == 'vertical':
                cb.ax.set_yticks(**ticks_params)
                labels = cb.ax.get_yticklabels()
                labels[0].set_verticalalignment('bottom')
                labels[1].set_verticalalignment('top') 
            elif orientation == 'horizontal':
                cb.ax.set_xticks(**ticks_params)
                labels = cb.ax.get_xticklabels()
                labels[0].set_horizontalalignment('left')
                labels[1].set_horizontalalignment('right')
            else:
                pass
        
    if position is not None:
        cb.ax.set_position(position)
    
    if rasterized:
        cb.ax.set_rasterization_zorder(0)
        
    cb.ax.set_title(
        title, loc=title_align,
        fontsize='medium', 
        fontstyle=title_style
    ) # 
    
    return cb


# TODO: no need to create axis
def plot_subtitle(ax:Axes, text, position='top', **kw):
    """ plot subtitle """
    
    fontsize    = kw.get('fontsize', 'medium')
    theme       = kw.get('theme', None)
    fontweight  = kw.get('fontweight', 'normal')
    style       = kw.get('style', 'normal')
    text_loc    = kw.get('text_loc', (0.5, 0.5))
    facecolor   = kw.get('facecolor', '0.9')
    text_color  = kw.get('text_color', 'black')
    
    prop = FontProperties(size=fontsize)
    fontsize = prop.get_size_in_points()
    height = kw.get('height', fontsize * 1.08)
    
    rotation = {'top': 0, 'bottom': 180, 'left': 90, 'right': 270}
        
    if theme == 'ggplot':
        
        # function takes an existing axes, adds it to a new AxesDivider
        divider = make_axes_locatable(ax)
        # Add an axes above the main axes.
        # Note: Higher the value for the pad 
        # the colorbar is away from the x-axis
        cax = divider.append_axes(position, size=f"{height}%", pad=0, sharex=ax)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.spines[:].set_visible(False) # TODO maybe some color spines
        cax.set_facecolor(facecolor) # TODO maybe other facecolor
        cax.text(
            text_loc[0],
            text_loc[1],
            text,
            style=style,
            size=fontsize,
            color=text_color,
            transform=cax.transAxes,
            rotation=rotation[position],
            ha="center",
            va="center",
            fontweight=fontweight,
        )
    else:
        ax.set_title(text, 
                     fontsize=fontsize, 
                     fontweight=fontweight, 
                     style=style)
        ax.set_yticklabels
        
# TODO: change the position of the title 
def plot_figtitle(fig, text, ncols, nrows, **kw): # need useful kwargs
    """ plot figtitle """
    
    fontsize    = kw.get('fontsize', 16)
    theme       = kw.get('theme', 'classic')
    fontweight  = kw.get('fontweight', 'normal')
    style       = kw.get('style', 'normal')
    text_color  = kw.get('text_color', 'black')
    text_loc    = kw.get('text_loc', (0.5, 0.5))
    facecolor   = kw.get('facecolor', '0.9')
    
    if theme == 'classic':
        fig.suptitle(
            text, fontsize=fontsize, fontweight=fontweight, style=style)
        
    elif theme == 'ggplot': # TODO: potion vertical or horizontal
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import FancyBboxPatch
        
        ax = fig.add_axes([0.12, 0.95, 0.77, 0.12 / nrows]) 
        bb = mtransforms.Bbox([[0.1, -0.1], [0.9, 0.8]])
        
        def add_fancy_patch_around(ax, bb, **kwargs):
            fancy = FancyBboxPatch((bb.xmin, bb.ymin), 
                                   bb.width, bb.height,
                                   fc=facecolor, ec=None,
                                   **kwargs)
            ax.add_patch(fancy)
            
        add_fancy_patch_around(
            ax, bb, boxstyle="round,pad=0.1,rounding_size=0.2")

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines[:].set_visible(False)

        ax.text(
            text_loc[0],
            text_loc[1],
            text,
            size=fontsize,
            style=style,
            color=text_color,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontweight='bold'
        )
    

def val_or_rc(val, rc_name):
    return val if val is not None else mpl.rcParams[rc_name]


# TODO arrow on 3d plot
# TODO let label on the end of arrow ?
# the ratio associated with the aspect and the data ratio
def draw_arrow(ax: Axes, invet_axis=False, **arrow_prop):
    """ draw arrow """
    
    arrow_weight = arrow_prop.get('arrow_weight', 1.05)
    arrow_style  = arrow_prop.get(
        'arrow_style', '<|-|>, head_length=0.5, head_width=0.25')
    arrow_label  = arrow_prop.get('arrow_label', 'COORD')
    fontsize     = arrow_prop.get('fontsize', 'medium')
    # offset       = arrow_prop.get('offset', 0.015)
    color        = arrow_prop.get('color', 'black')
    components   = arrow_prop.get('components', [0, 1])
    components   = map(str, components) 
    
    labels = ['_'.join([arrow_label, c]) for c in components]
    prop = FontProperties(size=fontsize)
    fontsize = prop.get_size_in_points()
    arrow_length = len(labels[0]) * .9 * fontsize
    arrow_length = arrow_prop.get('arrow_length', arrow_length)
    # 1 point = how much data units?
        
    if invet_axis:
        bb = ax.get_tightbbox() # must do this, will change window_extend
        height = bb.height / ax.figure.dpi # inch 
        height = height * 72
        x1, y0 = (arrow_length, height)
        x0, y1 = (0, height - arrow_length)
    else:
        x1, y0 = (arrow_length, 0)
        x0, y1 = (0, arrow_length)
    
    ax.annotate(
        '', 
        xy=(x1, y0), # start point
        xytext=(x0, y1), # end point
        xycoords='axes points', # coordinate system
        weight='bold', # font weight
        # arrow style
        arrowprops=dict(
            arrowstyle=arrow_style, 
            connectionstyle='angle,angleA=90,angleB=180,rad=0', 
            lw=arrow_weight, color=color)
    )
    
    if invet_axis: 
        align_x = {'va': 'bottom', 'loc': 'left'}
        align_y = {'va': 'bottom', 'loc': 'top'}
        ax.set_xlabel(f'{labels[0]}', fontsize=fontsize, **align_x)
        ax.set_ylabel(f'{labels[1]}', fontsize=fontsize, **align_y)
        ax.xaxis.set_label_position('top') 
    else:
        align_x = {'va': 'top', 'loc': 'left'}
        align_y = {'va': 'bottom', 'loc': 'bottom'}
        ax.set_xlabel(f'{arrow_label}_1', fontsize=fontsize, **align_x)
        ax.set_ylabel(f'{arrow_label}_2', fontsize=fontsize, **align_y)
        
        
def _generate_layout(corlor_list:list, 
                     nrows:Optional[int]=None, 
                     ncols:Optional[int]=None) -> tuple:
    """generate row and col numbers"""
    
    num = len(corlor_list)
    
    if nrows is not None:
        ncols = num // nrows + int(bool(num % nrows))
    else:
        if ncols is None:
            raise ValueError(
                'you must give the value to one of `ncols` and `nrows`')
        else:
            nrows = (num // ncols + int(bool(num % ncols)))
    
    if nrows == 1:
        ncols = num 
    elif ncols == 1:
        nrows = num
        
    return nrows, ncols


def _check_annotation(row_anno: Union[int, dict],
                      col_anno: Union[int, dict]):
    
    anno = {'left': 0, 'right': 0,
            'top':  0, 'bottom': 0}
    
    if isinstance(row_anno, int):
        anno.update({'left': row_anno})
    elif isinstance(row_anno, dict):
        anno.update(row_anno)
    
    if isinstance(col_anno, int):
        anno.update({'top': col_anno})
    elif isinstance(col_anno, dict):
        anno.update(col_anno)
    
    return anno

def mosaic_layout(row_annotations: Union[int, dict] = 0, 
                  col_annotations: Union[int, dict] = 0, 
                  parent: Optional[Union[List, str]] = None,
                  transpose: bool = False):
    """ generte the mosaic layout of subplot """
    
    if parent is None:
        mosaic = [['parent']]
    else:
        if isinstance(parent, str):
            mosaic = [[parent]]
        elif isinstance(parent, list):
            mosaic = [
                [item] if isinstance(item, str) 
                else item for item in parent]
    
    if len(mosaic) > 1:
        assert(
            all([len(row) == len(mosaic[0]) for row in mosaic[1:]])
        ), """checkout your input"""
    
    anno = _check_annotation(row_annotations, col_annotations)
    mosaic = anno['top'] * [['.'] * len(mosaic[0])] + mosaic
    mosaic = mosaic + anno['bottom'] * [['.'] * len(mosaic[0])]
    
    for _ in range(anno['left']):
        mosaic = [['.'] + row for row in mosaic]
        
    for _ in range(anno['right']): 
        mosaic = [row + ['.'] for row in mosaic]
        
    if transpose:
        mosaic = np.transpose(mosaic).tolist()

    return mosaic 

def remove_row_blank(mosaic):
    
    mosaic = [
        row for row in mosaic 
        if not all([item == '.' for item in row])
    ]
    return mosaic

def remove_col_blank(mosaic):
    
    mosaic = list(zip(*mosaic))
    mosaic = remove_row_blank(mosaic)
    return list(map(list, zip(*mosaic)))

def remove_blank(mosaic):
    
    mosaic = remove_row_blank(mosaic)
    mosaic = remove_col_blank(mosaic)
    return mosaic

def ajust_figsize(fig, cax, position):
    
    size_before = fig.get_size_inches()
    height = cax.get_window_extent().height / fig.dpi
    width = cax.get_window_extent().width / fig.dpi
    
    if position in ['left', 'right']:
        width += size_before[0]
    elif position in ['top', 'bottom']:
        height += size_before[1]
    
    return fig.set_size_inches(width, height)


VBound = Union[str, float, Callable[[Sequence[float]], float]]
def _get_vboundnorm(
    vmin: Sequence[VBound],
    vmax: Sequence[VBound],
    vcenter: Sequence[VBound],
    index: int,
    color_vector: Sequence[float],
) -> Tuple[Union[float, None], Union[float, None]]:
    # reference: scanpy
    """
    Evaluates the value of vmin, vmax and vcenter, which could be a
    str in which case is interpreted as a percentile and should
    be specified in the form 'pN' where N is the percentile.
    Eg. for a percentile of 85 the format would be 'p85'.
    Floats are accepted as p99.9

    Alternatively, vmin/vmax could be a function that is applied to
    the list of color values (`color_vector`).  E.g.

    def my_vmax(color_vector): np.percentile(color_vector, p=80)


    Parameters
    ----------
    index
        This index of the plot
    color_vector
        List or values for the plot

    Returns
    -------

    (vmin, vmax, vcenter, norm) containing None or float values for
    vmin, vmax, vcenter and matplotlib.colors.Normalize  or None for norm.

    """
    
    out = []
    for v_name, v in [('vmin', vmin), ('vmax', vmax), ('vcenter', vcenter)]:
        if len(v) == 1:
            # this case usually happens when the user sets eg vmax=0.9, which
            # is internally converted into list of len=1, but is expected that this
            # value applies to all plots.
            v_value = v[0]
        else:
            try:
                v_value = v[index]
            except IndexError:
                warnings.warn(
                    f"The parameter {v_name} is not valid. If setting multiple {v_name} values,"
                    f"check that the length of the {v_name} list is equal to the number "
                    "of plots. "
                )
                v_value = None

        if v_value is not None:
            if isinstance(v_value, str) and v_value.startswith('p'):
                try:
                    float(v_value[1:])
                except ValueError:
                    warnings.warn(
                        f"The parameter {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check the correct format for percentiles."
                    )
                # interpret value of vmin/vmax as quantile with the following syntax 'p99.9'
                v_value = np.nanpercentile(color_vector, q=float(v_value[1:]))
            elif callable(v_value):
                # interpret vmin/vmax as function
                v_value = v_value(color_vector)
                if not isinstance(v_value, float):
                    warnings.warn(
                        f"The return of the function given for {v_name} is not valid. "
                        "Please check that the function returns a number."
                    )
                    v_value = None
            else:
                try:
                    float(v_value)
                except ValueError:
                    warnings.warn(
                        f"The given {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check that the value given is a valid number, a string "
                        f"starting with 'p' for percentiles or a valid function."
                    )
                    v_value = None
        out.append(v_value)
    return tuple(out)


def check_colornorm(vmin=None, vmax=None, vcenter=None): # scanpy
    
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        # matplotlib<3.2
        from matplotlib.colors import DivergingNorm as DivNorm

    if vcenter is not None:
        norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def linewidth_from_data_units(linewidth, ax, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    
    fig = ax.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * ax.get_position().width
        value_range = np.diff(ax.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * ax.get_position().height
        value_range = np.diff(ax.get_ylim())
        
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return np.fabs(linewidth * (length / value_range))


def axis_ticklabels_overlap(labels): # seaborn
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False
    
    
def _set_labelsalign(rotation, loc='bottom'): 
    center_rotation = [0, 90, -90, None]
    in_center = rotation in center_rotation

    align_map = {
        'bottom': {'center': {'ha': 'center', 'va': 'top'},
                   'other': {'ha': 'right', 'va': 'top'}},
        'top': {'center': {'ha': 'center', 'va': 'bottom'}, 
                'other': {'ha': 'left', 'va': 'bottom'}},
        'left': {'center': {'ha': 'right', 'va': 'center'}, 
                 'other': {'ha': 'right', 'va': 'top'}},
        'right': {'center': {'ha': 'left', 'va': 'center'}, 
                  'other': {'ha': 'left', 'va': 'bottom'}}
    }
    align = align_map[loc]['center'] if in_center else align_map[loc]['other']
    
    return align if in_center else {'rotation_mode': 'anchor', **align}

           
def set_rotation(ax, axis='x', angle=None): # rotate xticklabels [45, 90]
    assert axis in ['x', 'y']

    ANGLES = [45, 90] if angle is None else [angle]
    
    if axis == 'x':
        ticklabels = ax.get_xticklabels() 
    else:
        ticklabels = ax.get_yticklabels()
    
    # TODO wrapper in a function named calculate_angle
    if ticklabels:
        for rotation in ANGLES:
            is_overlap = axis_ticklabels_overlap(ticklabels)
                          
            if is_overlap:
                angle = rotation
                ax.tick_params(
                    axis=axis,
                    rotation=rotation)
            else:
                break
            
        ax.set_xticklabels(
            ticklabels, 
            rotation=angle,
            **_set_labelsalign(angle)
        )


def categorical_order(
    vector: Union[pd.Series, np.ndarray], 
    order: Optional[List]=None
):
    # reference: seaborn
    """
    Return the order of levels in this categorical variable.

    Parameters
    ----------
    vector : Union[pd.Series, np.ndarray]
        vector of categorical values
    order : Optional[List], optional
        order of the categories, by default None

    Returns
    -------
    List
    
    """
    if order is None:
        if hasattr(vector, "categories"):
            order = vector.categories
        else:
            try:
                order = vector.cat.remove_unused_categories()
                order = order.cat.categories
            except (TypeError, AttributeError):
                order = np.unique(vector)
                
        order = filter(pd.notnull, order)

    
    return list(order)

# TODO colormap palette
def get_palette(
    palette: Optional[Union[str, Sequence]] = None, 
    as_hex: bool = True, 
    n_colors: int = 1,
    n_labels: Optional[int] = None,
    desat: Optional[float] = None,
    verbose: bool = False
):
    """
    return a list of colors for a given label
    
    this is a wrapper of seaborn.color_palette
    
    Parameters
    ----------
    label : Sequence
        a list of labels
    palette : Optional[Union[str, Sequence]], optional
        palette name or a list of colors, by default None
    as_hex : bool, optional
        Return a color palette with hex codes instead of RGB value
    n_colors : Optional[int], optional
        Number of colors in the palette. If ``None``, the default will depend
        on how ``palette`` is specified. but grabbing the current palette or 
        passing in a list of colors will not change the number of colors 
        unless this is specified. Asking for more colors than exist in the 
        palette will cause it to cycle. 

    Returns
    -------
    seborn color palette or list of colors
    """

    # TODO auto assign depending on the number of labels
    if palette is None:
        if n_colors <= 10:
            palette = _PALETTES['tab10']
            n_colors = 10
            
        elif n_colors <= 20:
            palette = 'tab20'
            n_colors = 20
            
        else:
            palette = _PALETTES['zeileis_28']
            n_colors = 28 
    else:
        if isinstance(palette, str):
            if palette in _PALETTES:
                palette = _PALETTES[palette]
            else:
                try:
                    palette = sns.mpl_palette(palette, n_colors)
                    n_colors = len(palette)
                except ValueError:
                    raise ValueError(
                        f"{palette} is not a valid palette name")
                    
        n_colors = len(np.unique(palette))
                
    n_labels = set_value(n_labels, n_colors)
    
    if (n_colors < n_labels) & verbose:
        logger.warning(
            "The number of label is larger than "
            "the number of colors in the palette. ")

    palette = sns.color_palette(palette, n_labels, desat=desat)
    return palette.as_hex() if as_hex else palette


def _check_palette(adata: AnnData, key, n_labels):
    color_key = f'{key}_colors'
    if color_key not in adata.uns:
        return False
    else:
        return len(adata.uns[color_key]) >= n_labels

def assign_palette(
    adata: AnnData,
    groupby: str, 
    order: Optional[Sequence] = None, 
    palette: Optional[Union[str, Sequence]] = None,
    as_dict: bool = True
):
    """
    Assign palette for the given group 
    
    in `adata.obs` and save it to `adata.uns`.

    Parameters
    ----------
    adata : AnnData
        an AnnData object
    groupby : str
        value key in adata.obs
    order : Optional[Sequence], optional
        categories order, by default None
    palette : Optional[Union[str, Sequence]], optional
        palette name or a list of colors, by default None

    Returns
    -------
    pd.Series
        a color dictionary that maps each category to a color
    """
   
    if is_cell_anno(adata, groupby):
        vector = adata.obs[groupby]
    else:
        g, v_ = get_ident(adata)
        if g == groupby:
            vector = v_
        else:
            raise ValueError(
                f'{groupby} is not a valid key')
    
    groups = categorical_order(vector)   
    n_groups = len(groups) 
    _valid = _check_palette(adata, groupby, n_groups)
    color_key = f'{groupby}_colors'
    save = False
    
    if order is not None:
        if set(order) == set(groups):
            groups = order
        else:
            order = [o for o in order if o in groups]
            n_groups = len(order)
    else:
        order = groups
    
    if palette is None:
        if _valid:
            palette = pd.Series(
                dict(zip(groups, adata.uns[color_key]))
                )
            palette = palette[order].values
        else:
            save = True
            palette = get_palette(n_colors=n_groups)

    else:
        if isinstance(palette, dict):
            palette = pd.Series(palette)[order].values
        else:
            palette = get_palette(
                palette, 
                n_colors=n_groups, n_labels=n_groups,
                verbose=True)
        save = True
            
    n_colors = len(np.unique(palette))
    if n_colors < n_groups:
        save = False
        n = n_groups - n_colors
        logger.warning(
            f'{n} labels will have the same color. ')
        
    # save palette to adata, so that we can use it in other plots
    if save:
        logger.info(
            f'add palette for `{groupby}` to adata.uns')
        adata.uns[color_key] = list(palette) 
    
    return pd.Series(dict(zip(order, palette))) if as_dict else palette


def _auoto_color(style, color=None):
    if color is None:
        color = 'black'
    
        if style == 'dark_background':
            if color == 'black':
                color = 'white'
        elif style == 'fast':
            pass
    
    return color

# set cax limit to make cax share x or y axis to ax
def set_cax_limit(ax, cax, sharex=False, sharey=False):
    if sharex:
        cax.set_xlim(ax.get_xlim())
        
    if sharey:
        cax.set_ylim(ax.get_ylim())
        

def inset_zoom(ax:Axes, bbox_to_anchor, 
               xlim=None, ylim=None, edgecolor='k',
               indicate=True, lw=0.5, **kwarg):
    try:
        from matplotview import inset_zoom_axes
    except ImportError:
        raise ImportError(
            "Please install matplotview to use this function."
        )
    
    axins = inset_zoom_axes(ax, bbox_to_anchor, **kwarg)
    if xlim is not None:
        axins.set_xlim(xlim)
    else:
        axins.set_xlim(ax.get_xlim())
        
    if ylim is not None:
        axins.set_ylim(ylim)
    else:
        axins.set_ylim(ax.get_ylim()) 
        
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.axis('off')
        
    if indicate:
        ax.indicate_inset_zoom(axins, edgecolor=edgecolor, lw=lw) 

    return axins
        

# --------------------------------------------------------------------------
#                               save utilities
# --------------------------------------------------------------------------        

def save_fig(
    path=None,
    prefix=None,
    dpi=None,
    ext="pdf",
    transparent=False,
    close=True,
    verbose=True,
    **kw
):
    """
    Save a figure from pyplot.
    
    code adapated from:
    http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/
    
    Parameters
    ----------
    path: `string`
        The path (and filename, without the extension) to save_fig the figure to.
    prefix: `str` or `None`
        The prefix added to the figure name. This will be automatically set
        accordingly to the plotting function used.
    dpi: [ None | scalar > 0 | 'figure' ]
        The resolution in dots per inch. If None, defaults to rcParams["savefig.dpi"].
        If 'figure', uses the figure's dpi value.
    ext: `string` (default='pdf')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close: `boolean` (default=True)
        Whether to close the figure after saving.  If you want to save_fig
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose: boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """

    if path is None:
        path = os.getcwd() + "/"

    # Extract the directory and filename from the given path
    directory   = os.path.split(path)[0]
    filename    = os.path.split(path)[1]
    if directory == "":
        directory = "."
    if filename == "":
        filename = "savefig"

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save_fig to
    savepath = (
        os.path.join(directory, filename + "." + ext)
        if prefix is None
        else os.path.join(directory, 
                          prefix + "_" + filename + "." + ext)
    )

    if verbose:
        logger.info(f"Saving figure to {savepath}...")

    # Actually save the figure
    plt.savefig(
        savepath,
        dpi=set_value(dpi, 300),
        transparent=transparent,
        format=ext,
        bbox_inches="tight",
        **kw
    )

    # Close it
    if close:
        plt.close()

    if verbose:
        logger.info("Done")
        

# --------------------------------------------------------------------------
#                   variable checking utilities
# --------------------------------------------------------------------------

def is_gene_name(adata: AnnData, var: str, use_raw: bool = False) -> bool:
    """ Check if a variable is a gene name. """
    use_raw = _check_use_raw(adata, use_raw)
    
    if type(var) in [str, np.str_]:
        return var in adata.raw.var_names if use_raw else var in adata.var_names
    else:
        return False
    
def is_cell_anno(adata: AnnData, var: str) -> bool:
    """ Check if a variable is a cell annotation column. """
    
    if type(var) in [str, np.str_]:
        return var in adata.obs.columns
    else:
        return False
    
def is_anno(adata: AnnData, var: str, use_raw: bool = False) -> bool:
    """ Check if a variable is an annotation column. """
    
    if type(var) in [str, np.str_]:
        return is_gene_name(adata, var, use_raw) or is_cell_anno(adata, var)
    else:
        return False

def _check_use_raw(adata: AnnData, use_raw: Union[None, bool]) -> bool:
    """
    Normalize checking `use_raw`.

    """
    if use_raw is not None:
        return use_raw
    else:
        if adata.raw is not None:
            return True
        else:
            return False


# TODO: base on this function to implement subset method in seurat
# TODO: keep replicates or not
# TODO: keep sparse matrix or not
# TODO: rename to subset

def fetch_data(
    adata: AnnData, 
    vars: Union[List[str], str], 
    layer: Optional[str] = None,
    as_frame: bool = True, 
    as_sparse: bool = False,
    use_raw: Optional[bool] = None,
    return_col: bool = False
): 
    """ Subset the meta value of a list of keys. """
    assert int(as_frame) + int(as_sparse) <= 1, (
        'as_frame and as_sparse cannot be True at the same time.')
    
    if layer is None:
        use_raw = _check_use_raw(adata, use_raw)
    else:
        use_raw = False
    
    if isinstance(vars, str):
        vars = [vars]
    
    # remove duplicated keys and keep the order
    vars = np.asarray(list(dict.fromkeys(vars)))
    
    genes   = [is_gene_name(adata, gene, use_raw=use_raw) for gene in vars]
    obs_col = [is_cell_anno(adata, obs) for obs in vars]
    
    duplicate_mask = np.logical_and(genes, obs_col)
    duplicates = vars[duplicate_mask]
    
    found_mask = np.logical_and(np.invert(genes), np.invert(obs_col))
    not_found = vars[found_mask]
    
    if len(duplicates) > 0:
        logger.warning(
             "The following keys are both in the .obs and .var, "
            f"we will use the obs value: {duplicates}")
        genes = np.logical_and(
            genes, np.invert(duplicate_mask))
    
    if len(not_found) > 0:
        logger.warning(
            f'{not_found} is neither gene nor cell annotation.')
    
    genes = vars[genes]
    obs_col = vars[obs_col]
    vars = vars[~found_mask]
    
    if use_raw:
        var_names = adata.raw.var_names
        X = adata.raw.X
    else:
        var_names = adata.var_names
        X = adata.X

    if len(genes) > 0:
        from scipy.sparse import issparse
        if len(obs_col) & as_sparse:
            as_sparse = False
            logger.warning(
                'as_sparse is not supported when obs_col is True')

        gene_mask = var_names.isin(genes)
        genes = var_names[gene_mask]
        values = (X[:, gene_mask] 
                  if layer is None else 
                  adata.layers[layer][:, gene_mask])
        
        values = values.A if issparse(values) else values
        genes_df = pd.DataFrame(
            data=values, columns=genes, index=adata.obs.index)
    else:
        genes_df = pd.DataFrame()

    obs_df = adata.obs.loc[:, obs_col]
    data = pd.concat([genes_df, obs_df], axis=1)
    data = data[vars]
        
    data = data if as_frame else data.values
    
    if return_col:
        return data, (genes, obs_col)
    else:
        return data

# subset method
# obsm, varm, layers, uns, obs, var, obsp, varp, uns
# obs X obsm -> merge
# var varm -> merge

def downsample(adata: AnnData, groupby:str, max_cell_per_group:int):
    """ Downsample the data to a given number of cells per group. """
    
    if max_cell_per_group < 1:
        raise ValueError('max_cell_per_group must be larger than 0')
    
    groups = adata.obs[groupby].unique()
    n_groups = len(groups)
    n_cells = adata.obs[groupby].value_counts()
    n_cells[n_cells > max_cell_per_group] = max_cell_per_group
    n_cells = n_cells.to_dict()
    
    subset_idx = adata.obs.groupby(groupby).apply(lambda x: x.sample(n=n_cells[x.name]).index)
    subset_idx = np.concatenate(subset_idx.values)
    
    return adata[subset_idx]
    

def subset(
    adata:AnnData,
    subset:str, 
    features, 
    cells, 
    inplace: bool=False):
    mask = adata.obs.eval(subset)
    
    if inplace:
        adata._inplace_subset_obs(mask)
    else:
        return adata[mask].copy()


# TODO rename to melt
def generate_var_value_df(
    adata: AnnData, 
    var_key: Union[str, list]=None, 
    value_key: Union[str, list]=None, 
    restrict_to: Optional[Union[dict, List]]=None,
    use_raw: Optional[bool]=None,
    layer: Optional[str]=None
):
    """ Generate a dataframe of variable values. """
    
    if value_key is not None:
        value_key = ([value_key] 
                     if isinstance(value_key, str) 
                     else list(dict.fromkeys(value_key)))
    else:
        value_key = []
        
    if var_key is not None:
        var_key   = ([var_key] 
                     if isinstance(var_key, str) 
                     else list(dict.fromkeys(var_key)))
    else:
        var_key = []
        
    if set(var_key).intersection(set(value_key)):
        raise ValueError('var_key and value_key must be different')

    df = fetch_data(adata, value_key + var_key, 
                           use_raw=use_raw, layer=layer)   
    value_key = [key for key in value_key if key in df.columns]
    var_key   = [key for key in var_key   if key in df.columns]
    
    if len(value_key) == 0:
        raise ValueError(
            'No valid value_key found in Anndata object')
    
    df = pd.melt(df, value_vars=value_key, id_vars=var_key)
    
    if restrict_to is not None:
        try:
            if isinstance(restrict_to, list):
                if len(var_key) == 1:
                    df = df[df[var_key[0]].isin(restrict_to)]
                    df[var_key[0]] = pd.Categorical(
                        df[var_key[0]], categories=restrict_to)
                else:
                    raise ValueError(
                        'multiple var_key found, '
                        '`restrict_to` must be a dict')
            
            elif isinstance(restrict_to, dict):
                if len(var_key) == 1:
                        df = df[
                            df[var_key[0]].isin(restrict_to[var_key[0]])
                            ]
                        df[var_key[0]] = pd.Categorical(
                            df[var_key[0]], categories=restrict_to[var_key[0]]
                            )
                else:
                    for key in restrict_to.keys():
                        df = df[df[key].isin(restrict_to[key])]
                        df[key] = pd.Categorical(
                            df[key], categories=restrict_to[key])
            else:
                raise ValueError('restrict_to must be a list or dict')
            
        except KeyError:
            pass
    
    return df

def pseudobulk_expression(
    adata: AnnData,
    groupby: Union[str, List],
    # rep: Union[str, List] = None,
    features: Optional[list] = None,
    return_adata: bool = False,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    with_ratio: bool = False,
    min_value: float = 0.,
    min_cells: float = 1,
    method: Literal['mean', 'sum'] = 'mean',
    summarized_layers: Optional[Union[str, List]] = None,
    normalization_method: Optional[str] = None,
    scale: bool = False,
):
    """Returns averaged expression values for each identity class

    Parameters
    ----------
    adata : AnnData
        an AnnData object
    groub_by : Union[str, List]
        Categories for grouping (e.g, ident, replicate, celltype)
    use_raw : bool, optional
        whether use raw data, by default False
    features : Optional[list], optional
        Features to analyze, by default None
    return_adata : bool, optional
        whether return the data as AnnData object, by default False
    layer : Optional[str], optional
        layer to use , by default None
    with_ratio : bool, optional
        whether with ratio, by default False
    min_value : float, optional
        the min expression to count, by default 0.

    Returns
    -------
    DataFrame | AnnData object

    """
    
    # TODO
    use_raw = _check_use_raw(adata, use_raw)

    if use_raw or (not return_adata):
        summarized_layers = None
    
    if summarized_layers is not None:
        if isinstance(summarized_layers, str):
            summarized_layers = [summarized_layers]
        
        _exist_layers = [i for i in summarized_layers 
                         if i in adata.layers.keys()]
        if len(summarized_layers) == 0:
            raise ValueError('No valid layers found') 
        elif len(_exist_layers) < len(summarized_layers):
            summarized_layers = _exist_layers
            logger.warning(
                f'Only {summarized_layers} layers found in .layers')
        else:
            pass
        
    if isinstance(groupby, str):
        groupby = [groupby]
        
    groupby = [i for i in groupby if i in adata.obs_keys()] 
    cell_mask = adata.obs[groupby].isna().any(axis=1)
    if cell_mask.sum() > 0: 
        logger.warning(
            "Removing cells with NA for 1 or more grouping variables")
        adata = adata[~cell_mask, :]
        
    # remove columns with 1 unique value
    group_mask = adata.obs[groupby].nunique() == 1
    if group_mask.sum() > 0:
        logger.warning(
            f"Removing columns with 1 unique value")
        groupby = group_mask.index[~group_mask].tolist()
    
    if len(groupby)>1:
        cat_vec = (
            adata.obs[groupby[0]].str.cat(
            adata.obs[groupby[1:]], sep='_').astype('category'))

        cat2group = (
            adata.obs[groupby]
            .groupby(cat_vec)
            .apply(lambda x: x.drop_duplicates())
        )
    else:
        cat_vec = pd.Categorical(adata.obs[groupby[0]])
    
    order = categorical_order(cat_vec)
    # one-hot encoding of categorical variables
    _onehot = pd.get_dummies(
        cat_vec, sparse=True, dtype=np.int16)[order] # cells x groups
    _onehot = _onehot.values.T
    
    if features is not None:
        if isinstance(features, str):
            features = [features]
        
        data = fetch_data(
            adata, features, use_raw=use_raw, layer=layer)
        features, data = data.columns, data.values
        
        if summarized_layers:
            layers_data = {
                key: fetch_data(
                    adata, features, layer=key, as_frame=True)
                for key in summarized_layers}
    else:
        data = adata.X if layer is None else adata.layers[layer]
        features = adata.var_names
        if summarized_layers:
            layers_data = {key: adata.layers[key] for key in summarized_layers}
    
    if with_ratio:
        ratio = _onehot @ (data > min_value)  
        ratio = ratio / _onehot.sum(axis=1, keepdims=True) * 100
        
    if scale & ('log1p' in adata.uns_keys()) & (not use_raw):
        data = np.expm1(
            data.astype(np.float64))
    
    layers_X = {}
    if method == 'mean':
        cat_counts = _onehot.sum(axis=1, keepdims=True)
        X = _onehot @ data / cat_counts
        if summarized_layers:
            layers_X = {
                key: _onehot @ value / cat_counts
                for key, value in layers_data.items()}
    elif method == 'sum':
        X = _onehot @ data
        if summarized_layers:
            layers_X = {
                key: _onehot @ value
                for key, value in layers_data.items()}

    if return_adata:
        obs = pd.DataFrame(index=order)
        if len(groupby) > 1:
            obs[groupby] = cat2group.loc[order, groupby].values
        else:
            obs[groupby[0]] = order
            
        var = pd.DataFrame(index=features)
        if with_ratio:
            layers_X['ratio'] = ratio
        return AnnData(X=X, obs=obs, var=var, layers=layers_X)
    else:
        mean_df = pd.DataFrame(X, index=order, columns=features)
        if with_ratio:
            ratio_df = pd.DataFrame(ratio, index=order, columns=features)
        
        return mean_df if not with_ratio else (mean_df, ratio_df)


def add_metadata(
    adata: AnnData, 
    metadata: Union[pd.DataFrame, pd.Series, np.ndarray],
    colname: Optional[Union[str, List]] = None,
):
    """
    Add metadata to an AnnData object.

    Parameters
    ----------
    adata : AnnData
        an AnnData object
    metadata : Union[pd.DataFrame, pd.Series, np.array, dict]
        metadata to add
    colname : Optional[str], optional
        column name of the metadata, by default None

    Returns
    -------
    AnnData
        an AnnData object with metadata added
    """
    
    if isinstance(metadata, np.ndarray):
        metadata = pd.DataFrame(metadata, index=adata.obs.index)
    elif isinstance(metadata, pd.Series):
        colname = metadata.name if colname is None else colname
        metadata = metadata.to_frame()
    elif isinstance(metadata, pd.DataFrame):
        colname = metadata.columns if colname is None else colname
    
    if colname is None:
        raise ValueError('colname must be provided')
    else:
        if isinstance(colname, str):
            colname = [colname]
        
        adata.obs[colname] = metadata
    

#TODO automate detect the embedding method
def get_cell_embeddings(
    adata: AnnData, 
    reduction: Optional[str] = 'auto', 
    contour_col: str = 'contour',  # as_geo
    n_components: Optional[Union[int, Sequence]] = None,
    as_frame: bool = True, 
):  
    """
    get cell embeddings df from adata

    Parameters
    ----------
    adata : AnnData
        anndata object
    reduction : Optional[str], optional
        cell reduction saved in .obsm, usually start with 'X_', by default 'pca'
    as_frame : bool, optional
        whether return pandas dataframe, by default True
    contour_col : str, optional
        geo coutour, often save cell contour point, by default 'contour'

    Returns
    -------
    pd.DataFrame | np.ndarray
    
    """
    
    if reduction not in adata.obsm_keys():
        if reduction == 'contour':  
            data, _ = _parse_coord(adata.obs[contour_col], True)
        elif reduction == 'auto':
            raise NotImplementedError(
                'auto detect reduction method is not implemented')
        else:
            try:
                data = adata.obsm[f'X_{reduction}']
            except KeyError:
                raise ValueError(
                    f'{reduction} or X_{reduction} is not in adata.obsm_keys()')
    else:   
        data = adata.obsm[reduction].astype(np.float32).copy()
    
    if reduction == 'contour':
        data = GeoSeries(data, adata.obs_names)
    else:
        data = pd.DataFrame(data, index=adata.obs.index)
        N_COMP = data.shape[1]
        
        if n_components is not None:
            if isinstance(n_components, int):
                data = data.iloc[:, :min(n_components, N_COMP)]
            elif isinstance(n_components, Sequence):
                if N_COMP < len(n_components):
                    n_components = slice(None, N_COMP)
                data = data.iloc[:, list(n_components)]
            else:
                raise ValueError(
                    'n_component must be int or Sequence[int]')
        else:
            data = data.iloc[:, :min(2, N_COMP)]
        
        if data.shape[1] == 2:
            data.columns = ['x', 'y']   
        elif data.shape[1] == 3:
            data.columns = ['x', 'y', 'z']
        else:
            pass
    
    return data if as_frame else data.values 


def rotate2d(coord: np.ndarray, angle: float):
    """ Rotate the coordinate by an angle. """
        
    angle   = np.deg2rad(angle)
    rot_mat = np.array([
        [ np.cos(angle),  np.sin(angle)],
        [-np.sin(angle),  np.cos(angle)]
    ])
    
    return np.dot(coord, rot_mat)

def rotate(adata, angle, reduction='spatial', inplace=False):
    coords = get_cell_embeddings(adata, reduction, as_frame=True)
    logger.info(f'rotating {reduction} by {angle} degree')
    
    if reduction == 'contour':
        coords = np.asarray(
            coords.rotate(angle, origin=(0,0)),
            dtype=str
        )
        if inplace:
            adata.obs[reduction] = coords
        else:
            return coords
    else:
        coords = rotate2d(coords, angle)
        if inplace:
            adata.obsm[reduction] = coords
        else:
            return coords
        
        
def flip2d(coord: np.ndarray, axis: str): 
    """ Flip the coordinate along an axis. """
    
    if axis == 'x':
        flip_mat = np.array([
            [-1, 0], 
            [ 0, 1]
        ])
    elif axis == 'y':
        flip_mat = np.array([
            [1,  0],
            [0, -1]
        ])
    else:
        raise ValueError(
            'axis must be x or y')
    
    return np.dot(coord, flip_mat)

def flip(adata, reduction='spatial', inplace=False, axis='y'):
    coords = get_cell_embeddings(adata, reduction, as_frame=True)
    logger.info(f'flipping {reduction} along {axis} axis')
    
    if reduction == 'contour':
        flip_mat = np.array([-1, 0, 0, 1, 0, 0])
        flip_mat = flip_mat if axis=='x' else -flip_mat
        coords = np.asarray(
            coords.affine_transform(flip_mat),
            dtype=str
        )
        
        if inplace:
            adata.obs[reduction] = coords
        else:
            return coords
    else:
        coords = flip2d(coords, axis)
        if inplace:
            adata.obsm[reduction] = coords
        else:
            return coords
        
        
# TODO more flexible position
def move_point(coord_1, coord_2, gap=0, position='right', is_contour=False):
    """ move coord 1 of bject 1 to the right or bottom of object 2 """
    
    if is_contour:
        coord_1 = GeoSeries(coord_1)
        coord_2 = GeoSeries(coord_2)
        bounds_1 = coord_1.bounds
        bounds_2 = coord_2.bounds
        
        if position == 'right':
            offset_center = bounds_1[['minx', 'miny']].min()
            move_to = (
                bounds_2[['maxx', 'miny']]
                .agg({'maxx': np.max, 'miny': np.min})
            )
            gap = (gap, 0)
        else:
            center_x1 = (
                bounds_1[['minx', 'maxx']]
                .agg({'minx': np.min, 'maxx': np.max})
                .mean()
            )
            center_x2 = (
                bounds_2[['minx', 'maxx']]
                .agg({'minx': np.min, 'maxx': np.max})
                .mean()
            )
            offset_center = (center_x1, bounds_1['maxy'].max())
            move_to = (center_x2, bounds_2['miny'].min())
            gap = (0, -gap)

        coord_moved = coord_1.translate(
            xoff=move_to[0]-offset_center[0]+gap[0],
            yoff=move_to[1]-offset_center[1]+gap[1],
        )
    else:
        coord_1 = np.asarray(coord_1)
        coord_2 = np.asarray(coord_2)
        x1, y1 = coord_1.T
        x2, y2 = coord_2.T 
    
        if position == 'right':
            # center of object 1
            offset_center  = (x1.min(), y1.min())
            # move to the right of object 2
            move_to = (x2.max(), y2.min())
            gap = (gap, 0)
        elif position == 'bottom': 
            offset_center  = (np.median(x1), y1.max())
            move_to = (np.median(x2), y2.min())
            gap = (0, -gap)
        else:
            raise ValueError(
                '`position` now only supports right or bottom')
    
        coord_center = coord_1 - offset_center
        coord_moved = move_to + coord_center
        coord_moved = coord_moved + gap
    
    return coord_moved

def arange_spatial(
    adata: AnnData, 
    groupby: str = 'batch', 
    reduction: str = 'spatial',
    wspace:float = 0,
    hspace:float = 0,
    ncols:Optional[int] = None,
    nrows:Optional[int] = None,
    order: Optional[Sequence] = None, # ordered by size ?
    verbose: bool = True,
    inplace: bool = False 
):

    """
    Arrange spatial coordinates in a grid.

    output like below

    coord_0 - wspace - coord_1 - ...

    |

    hspace

    |

    ...

    """
    
    if ncols is None and nrows is None:
        nrows = 1

    coords = get_cell_embeddings(adata, reduction, as_frame=False)
    order = categorical_order(adata.obs[groupby], order=order)
    nrows, ncols = _generate_layout(order, nrows=nrows, ncols=ncols)
    masks = [adata.obs[groupby] == group for group in order]
    is_contour = reduction == 'contour'

    if verbose:
        logger.info('moving coordinate')

    for i in range(len(masks)):
        if i > 0:
            to_move = coords[masks[i]]

            if i % ncols != 0: # i // ncols == 0
                coord_pre = coords[masks[i-1]]
                kw = dict(position='right', gap=wspace)
            else:
                coord_pre = coords[masks[i-ncols]]
                kw = dict(position='bottom', gap=hspace)

            coords[masks[i]] = move_point(
                to_move, coord_pre, 
                is_contour=is_contour, **kw)
        
    if inplace:
        if is_contour:
            adata.obs[reduction] = coords.astype(str)
        else:
            adata.obsm[reduction] = coords
    else:
        return coords
    

def within2d(coord: np.ndarray, bbox: np.ndarray):
    """ Filter the coordinate within the bounding box. """
    
    x, y = coord.T
    x_min, y_min, x_max, y_max = bbox
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    return mask
    
def within(adata, bbox, reduction='spatial', inplace=False):
    coords = get_cell_embeddings(adata, reduction, as_frame=False)
    
    if reduction == 'contour':
        from shapely.geometry import box
        mask = coords.within(box(*bbox))
    else:
        mask = within2d(coords, bbox)

    return adata[mask] if inplace else mask
        
        
# TODO
def top_markers(adata: AnnData, groups:Optional[str]=None, 
                key:str='rank_genes_groups', top_n:int=10, **kwargs): 
    
    pass

# TODO align multiple bintype stereo-seq data
# how to transfer the label of small bin to large bin
# how to transfer the label of large bin to small bin


# ==================================================================================
#                               scatter plot
# ==================================================================================

# when change the limit of the axis, the size of the dot will change
def _unit_scatter_size(ax, xy, size=1):
    """
    fake scatter plot to get the
    size of the dot in data unit
    """
    ax.set_aspect('equal')
    ax.scatter(*zip(*xy), 
                s=0, alpha=0, 
                rasterized=True)
    dot_size = linewidth_from_data_units(size, ax=ax)
    return dot_size


def _parse_coord(
    coords: Union[np.ndarray, pd.Series, pd.DataFrame]
    , is_contour=False
):
    """ Parse the coordinates. """
    if isinstance(coords, (pd.Series, pd.DataFrame)):
        coords = coords.values
    else:
        if not hasattr(coords, 'shape'):
            raise ValueError(
                'Coordinates must be a numpy array '
                'or pandas Series/DataFrame')
        
    if is_contour:
        coords = coords.flat if coords.ndim == 2 else coords
        try:
            coords = GeoSeries(coords).values
        except TypeError:
            coords = GeoSeries.from_wkt(coords).values
        
        colnames = 'contour'
    else:
        if coords.shape[1] == 1:
            raise ValueError(
                'Coordinates must be 2D or 3D')
            
        colnames = ['x', 'y', 'z'][: coords.shape[1]]
        
    return coords, colnames


def draw_scalebar(
    ax, dx=0.5, unit='um', location='lower right',
    length_fraction=0.2, width_fraction=0.02,
    box_alpha=0.1, **kwargs
):
    """ 
    Draw scalebar on the figure.
    
    for stereo-seq, 1 pixel = 0.5 um (500nm)
    
    """
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
    except ImportError:
        raise ImportError(
            "try `pip install matplotlib-scalebar`")  
    
    scalebar = ScaleBar(
        dx, 
        unit, 
        box_alpha=box_alpha, 
        location=location, 
        length_fraction=length_fraction, 
        width_fraction=width_fraction, 
        **kwargs
    )
    
    ax.add_artist(scalebar)
    
    
def set_force_equal_aspect(ax: Axes = None):
    if ax is None:
        ax = plt.gca()
        
    ax.set_aspect(
        1.0 / ax.get_data_ratio(), adjustable='box')

# border
def scatter( 
    # TODO trajectory stream plot
    # TODO projection on 3d
    coords: Union[np.ndarray, pd.Series],
    colors: Optional[np.ndarray],
    shapes: Optional[np.ndarray] = None, 
    order: Optional[Union[Tuple, List]] = None,
    markers: Optional[Union[str, List]] = None,
    labels: bool = False,
    is_contour: bool = False,
    dot_size: Optional[float] = None,
    label_size: Union[float, str] = 'small',
    stroke: Optional[float] = None,
    alpha: float = .9,
    text: bool = False,
    repel: bool = False,
    raster: bool = True,
    legend: bool = True,
    invert_y: bool = False, 
    legend_kw: dict = {},
    aspect: str = 'equal', 
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Union[Tuple, List] = (6, 6),
    palette: Optional[Union[str, Sequence]] = None,
    save_path: Optional[str] = None,
    **kwargs
    # legend_loc: str = 'right'
    # projection: Iterable
) -> Axes:

    """
    scatter used for clustering plot or spatial plot

    Parameters
    ----------
    coords : Union[np.ndarray, List]
        two dimension data
    colors : Union[np.ndarray, List]
        point labels
    order : Union[Tuple, List]
        label order
    labels : Optional[Dict], optional
        the label annotion, by default None
    dot_size : `float`, optional
        point size, by default 1.2
    alpha : `float`, optional
        alpha, by default 1.
    stroke : float, optional
        linewidth, by default 0.25
    text : `bool`, optional
        whether draw label text, by default False
    figsize : Union[Tuple, List], optional
        figure size, by default (6, 6)
    repel : `bool`, optional
        whether adjust label text only work when `text=True`, by default False
    ax : Optional[Axes], optional
        The matplotlib axes object where new plots will be added to, by default None
    save_path : Optional[str], optional
        save path, by default None
    invert_y : `bool`, optional
        whether invert y axis, by default False
    legend_kw: `dict`, optional
        legend kwargs, by default {}
        - available kwargs:
            - bbox_to_anchor
            - loc by default 'center left'
            - markerscale by default 250
            - font_size by default medium
    """
 
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    elif ax is None:
        ax = fig.add_subplot(111)
        
    # get palette
    colors = np.array(colors).flatten()
    order = categorical_order(colors, order=order)   
    palette = get_palette(palette, 
                          n_colors=len(order),
                          n_labels=len(order))
    
    coords, _ = _parse_coord(coords, is_contour)
    
    # plot
    txts = []
    kw = dict(alpha=alpha, rasterized=raster)
    
    kwargs.setdefault('edgecolor', 'none')
    kw.update(kwargs)
    
    # TODO shape.by
    # if isinstance(markers, str):
    #     markers = np.repeat(markers, coords.shape[0])
    # elif markers is None:
    #     markers = np.repeat('.', coords.shape[0])
    # else:
    #     # do like seaborn convert categorical to kind of shape
    #     pass
    
    for idx, color in enumerate(order):
        # masks
        mask = colors == color
        coord = coords[mask]
        
        kw['color'] = palette[idx]
        
        if mask.sum() == 0:
            logger.warning(f'No {color} in data')
            continue
        
        # spatial cell polygon
        if is_contour:
            stroke = set_value(stroke, 0.001)
            kw.setdefault('linewidth', stroke)
            
            GeoSeries(coord).plot(ax=ax, **kw)
            
        # single or binsize cell circle
        else:
            kw.setdefault('s', dot_size)
            kw.setdefault('linewidth', stroke)
            
            ax.scatter(*coord.T, **kw)

        ax.scatter([], [], color=kw['color'], label=color)
        # text  
        if text: 
            x_t, y_t = coord.mean(axis=0)
            t = str(idx) if labels is None else color

            txt = ax.text(
                    x_t, 
                    y_t, 
                    t, 
                    c='k', 
                    fontsize=label_size
                ) 

            txt.set_path_effects(
                [Stroke(linewidth=1.25, 
                        foreground="w"),
                 Normal()]
                )
            txts.append(txt)

    # adjust text
    if repel:
        adjust_text(
            txts,
            arrowprops=dict(
                arrowstyle="->", 
                color="grey", 
                lw=0.5)
        )
    
    # invert y axis
    if invert_y:
        ax.invert_yaxis()
        
    #set aspect
    if aspect == 'force_equal':
        set_force_equal_aspect(ax)
    else:
        ax.set_aspect(aspect)
        
    # draw legend  
    # TODO: have to adapt to different figure size
    if legend:
        plot_legend(ax=ax, figsize=figsize, **legend_kw)
        
    # save figure
    if save_path:
        save_fig(save_path)
            
    return ax


def plot_edges(
    axs, adata, reduction, edges_width, edges_color, neighbors_key=None):
    
    import networkx as nx
    if not isinstance(axs, Sequence):
        axs = [axs]

    if neighbors_key is None:
        neighbors_key = 'neighbors'
    if neighbors_key not in adata.uns:
        raise ValueError(
            '`edges=True` requires `pp.neighbors` to be run before.')
    
    neighbors = adata.uns[neighbors_key]
    g = nx.Graph(neighbors['connectivities'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax in axs:
            edge_collection = nx.draw_networkx_edges(
                g,
                adata.obsm[reduction],
                ax=ax,
                width=edges_width,
                edge_color=edges_color,
            )
            edge_collection.set_zorder(-2)
            edge_collection.set_rasterized(True)


# def _get_reduction(adata):
#     return adata.uns['ssat']['plot_param']['reduction']

# refer: scanpy
def _basis2name(basis: str):
    """
    converts the 'basis' into the proper name.
    """
    basis = basis.replace('X_', '')

    component_name = (
        'DC'
        if basis == 'diffmap'
        else 'tSNE'
        if basis == 'tsne'
        else 'UMAP'
        if basis == 'umap'
        else 'PC'
        if basis == 'pca'
        else basis.replace('draw_graph_', '').upper()
        if 'draw_graph' in basis
        else basis.upper()
    )
    return component_name


# TODO high-level to create an inset axs (maplotview)
# TODO build a layout manager

def get_groupby_vector(adata, groupby, order=None):
    
    if groupby not in adata.obs_keys():
        g, v = get_ident(adata)
        
        if g is not None:
            groupby = g
        else:
            if groupby is not None:
                logger.warning(
                    f'`{groupby}` not in adata.obs_keys() '
                    'ignore groupby')
                
            return None, None, []
    else:
        v = adata.obs[groupby]
    
    order = categorical_order(v, order)
    return groupby, v, order
    
# set default reduction to auto   

# check if a list have repeated elements and return the repeated elements
def _check_repeated_elements(lst):
    
    lst = np.asarray(lst)
    unique, counts = np.unique(lst, return_counts=True)
    return unique[counts > 1]

def _parse_category(adata, key, order):
    
    key_in_obs = key in adata.obs_keys()
    
    if (key is not None) & key_in_obs:
        vec = adata.obs[key]
        order = categorical_order(vec, order)
    else:
        key, vec, order = None, None, []
    
    return key, vec, order 

# TODO support shape.by
# TODO highlight include cells.highlight, cols.highlight, sizes.highlight
# TODO support 3d scatter plot and 3d spatial plot

@register_decorator
def dimplot(
    adata: AnnData, 
    groupby: Union[str, Sequence] = None,  # multiple categories ?
    reduction: str = 'umap', 
    splitby: str = None, 
    aspect: int = 1,
    height: float = 4, 
    ncols: int = 4, # TODO
    legend: bool = True,
    legend_kw: dict = {}, 
    arrow_kw: dict = {}, 
    invert_y: bool = False,
    style: str = 'fast',
    text: bool = False,
    arrow: bool = False,
    edges: bool = False,
    sharex: bool = True,
    sharey: bool = True,
    wspace: float = 0.05,
    outline: bool = False, # TODO chull or outline? 
    density: bool = False, # TODO
    edges_width: float = 0.1,
    edges_color: str = 'grey',
    # dot_size: Optional[float] = None,
    labels: Optional[Dict] = None,
    components: Optional[Sequence[int]] = None,
    contour_col: Optional[str] = 'contour',
    order: Optional[Sequence[str]] = None,
    split_order: Optional[Sequence[str]] = None,
    legend_subtitle: Optional[Union[List, Dict, str]] = None,
    palette: Optional[Union[str, Sequence[str], dict]] = None,
    background: bool = False,
    mask: Optional[np.ndarray] = None, # rename to highlight next version
    ax: Optional[Axes] = None,
    **kwargs
):
    """
    Graphs the output of a dimensional reduction technique on a 2D scatter plot
    where each point is a cell and it's positioned based on the cell embeddings 
    determined by the reduction technique or spatial coordinate By default, cells are 
    colored by their identity class (can be changed with the groupby parameter).
    
    Parameters
    ----------
    adata : AnnData
        anndata
    groupby : `str`, optional
        Name of one or more metadata columns to group (color) cells by 
        (for example, orig.ident); pass 'ident' to group by identity class
    splitby : `str`, optional
        A discret variable vector to split the plot by, 
        pass 'ident' to split by cell identity'
    components : `tuple`, optional
        Dimensions to plot, must be a two-length numeric 
        vector specifying x- and y-dimensions
    order: `list`, optional
        Specify the order of plotting for the idents. 
        This can be useful for crowded plots if points of interest are being buried. 
        Provide either a full list of valid idents or a subset to be plotted last (on top)
        
    """
    
    components = set_value(components, [0, 1])
    if len(components) not in [2, 3]:
        raise ValueError(
            'components must be a list of length 2 or 3')
    
    if isinstance(order, str):
        order = [order]

    if isinstance(groupby, List):
        if len(groupby) == 1:
            groupby = groupby[0]
    
    # transfer str to categorical
    # adata._sanitize()  # sometimes the data is not sanitized
    
    # santitize reduction
    
    # get data
    data = get_cell_embeddings(
        adata, 
        reduction=reduction, 
        contour_col=contour_col,
        n_components=components)
    
    # column names of data when plot spatial plot,
    # the column name is 'contour'  
    coords = data.values
            
    # map color to categorical data
    groupby, v, order = get_groupby_vector(
        adata, groupby, order)
    
    splitby, col_v, split_order = _parse_category(
        adata, splitby, split_order)
    
    # assign color to each cell
    if groupby:
        color_dict = assign_palette(adata, 
                                    groupby=groupby,
                                    order=order, 
                                    palette=palette)
    else:
        background = True
        legend = False
        legend_subtitle = None
        color_dict = pd.Series({np.nan: '.8'}, name='nan')
        
    nrows = 1
    ncols = 1 if splitby is None else len(split_order)
    figsize = (ncols * height * aspect * (1 + wspace), nrows * height)
    
    if ncols == 1:   
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
        fig = ax.figure
        axs = [ax]
    else:        
        fig = plt.figure(figsize=figsize)
        # subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        # gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        
        axs = fig.subplots(
            nrows,
            ncols, 
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            # subplot_kw=subplot_kws,
            # gridspec_kw=gridspec_kws
        )
        axs = axs.flatten()
        
    # def set size
    if reduction == 'spatial':
        kwargs.setdefault('marker', 'o')
        dot_size = (kwargs.pop('dot_size') 
                    if 'dot_size' in kwargs else 0.5)
        stroke = (kwargs.pop('stroke') 
                  if 'stroke' in kwargs else 0.1)
        
        SIZE_FACTOR = np.sqrt(np.pi)
        unit = _unit_scatter_size(axs[0], coords)
        size = dot_size * unit * SIZE_FACTOR
        stroke = stroke * unit * SIZE_FACTOR
        kwargs['dot_size'] = size
        kwargs['stroke'] = stroke
    elif reduction == 'contour':
        kwargs['is_contour'] = True
    else:
        kwargs.setdefault('marker', '.')
        kwargs.setdefault('dot_size', 
                          120000 / adata.shape[0])
    
    def _background(points, ax):
        background_param = dict(
            alpha=0.5,
            zorder=-1,
            color='.8',
            rasterized=True)
        
        if reduction == 'contour':
            GeoSeries(points).plot(
                ax=ax, 
                **background_param
            )
        else:
            ax.scatter(
                *points.T,
                s=kwargs['dot_size'] * 1.15,
                marker=kwargs['marker'],
                **background_param
            )
        
        ax.scatter([], [], color=background_param['color'], label='Others')

    # TODO subtitle facecolor
    subtitle_fc = None
    if legend_subtitle is not None:
        logger.info('set subtitle to the groups')
        if isinstance(legend_subtitle, str):
            subtitle_fc = assign_palette(adata, legend_subtitle)
            legend_subtitle = adata.obs[legend_subtitle]
            
        if isinstance(legend_subtitle, (list, pd.Series)):
            df_to_count = pd.DataFrame(
                {'GROUPBY': legend_subtitle, 'COUNT': v})
            legend_subtitle = dict(
                df_to_count.groupby('GROUPBY')['COUNT']
                .apply(lambda x: categorical_order(x)))
            
        if isinstance(legend_subtitle, dict):
            legend_order = list(chain(*list(legend_subtitle.values())))
            repeated = _check_repeated_elements(legend_order)
            if repeated.size > 0:
                raise ValueError(
                    f'subtype {repeated} appear in different groups')
            order = [od for od in legend_order if od in order]
    
    def facet_data(v, col_v, mask, order, split_order):   
        
        if order:
            # data[groupby] = v
            idx = data.loc[mask].index if mask is not None else data.index
            isin = data.index.isin(idx)   
        else:
            split_order = []
            v = np.full(adata.shape[0], np.nan)  
            isin = np.repeat(False, adata.shape[0])     
        
        if split_order:
            # data[splitby] = col_v
            col_masks = [col_v == n for n in split_order]
        else:
            col_masks = [np.repeat(True, data.shape[0])]

        for i, col in enumerate(col_masks):
            col = col & isin    
            notin = ~col    
            yield i, coords[col], v[col], notin   
    
    iters = facet_data(v, col_v, mask, order, split_order)

    # facet by the splitby   
    size = None
    for (i, coord, color, bg) in iters:
        ax = axs[i]

        # the color and category may not be the same
        unique = categorical_order(color)
        re_order = [i for i in order if i in unique]
            
        # keep the color right
        pal = color_dict[re_order].values
        
        # draw on facet axis
        scatter(
            coord, 
            color,
            text=text, 
            palette=pal,
            legend=False,
            order=re_order, 
            invert_y=invert_y,
            labels=labels,
            ax=ax,
            **kwargs
        )
        
        # background points
        if background:
            if len(re_order) < len(unique):
                mask_group = color.isin(re_order)
                _background(coord[~mask_group], 
                            axs[i])
                
            if bg.sum() > 0:
                _background(coords[bg], 
                            axs[i])
        
        # set title 
        if split_order:
            ax.set_title(split_order[i])
            
        # draw connectivities
        if edges:
            plot_edges(
                axs=ax,
                adata=adata,
                reduction=reduction,
                edges_width=edges_width,
                edges_color=edges_color)
        
        if density:
            pass
        
        ax.axis('off')

    if legend:
        number_color = legend_kw.pop('number_color', 
                                     _auoto_color(style))
        plot_legend(
            fig=fig, ax=ax, 
            subtitle=legend_subtitle,
            subtitle_fcs=subtitle_fc,
            num_in_legend=((labels is None) & text),
            number_color=number_color,
            # figsize=plt.gcf().get_size_inches(), # TODO
            **legend_kw)
    
    if arrow:
        draw_arrow(
            ax=ax if fig is None else fig.get_axes()[0],  
            invet_axis=invert_y,
            arrow_label=_basis2name(reduction),
            components=components,
            color=_auoto_color(style),
            **arrow_kw)
            
    return axs[0] if len(axs) == 1 else axs


# ==================================================================================
#                               feature plot
# ==================================================================================

# distribution plot (umap, spatial, ...)

# TODO: draw on fig ?
# TODO need component variable
# TODO: put picture in the background
def distribution_plot(
    coords: Union[np.ndarray, pd.Series],
    value_df: pd.DataFrame,
    splitby: Optional[np.array] = None,
    dot_size: Optional[str] = None,
    nrows: Optional[int] = None,
    ncols: int = 4,
    height: int = 3.,
    invert_y: bool = False,
    labels: bool = False,
    stroke: float = 0.,
    kind: str = 'cell', 
    hexbin: bool = False,
    gridsize: float = 60,
    wspace: float = 0.15,
    hspace: float = 0.15,
    frameon: bool = True,
    share_cbar: bool = False,
    raster: bool = True,
    legend: bool = True,
    plot_arrow: bool = False,
    cbar_kw: dict = {},
    subtitle_kw: dict = {},
    arrow_kw: dict = {},
    style: str = 'fast',
    pl_cls: str = 'coord',
    cbar_orientation: str = 'vertical',
    aspect: Union[float, str] = 'equal', # to figure_aspect, axis_aspect
    transpose_axes: bool = False,
    tight_layout : bool =  True,
    axs: Optional[Union[Axes, List[Axes]]] = None, # TODO rename to ax  List
    fig: Optional[Figure] = None,
    save_path: Optional[str] = None,
    is_genes: Optional[Sequence[bool]] = None,
    cmap: Optional[Union[str, Sequence]] = None,
    vmin: Optional[Union[str, List[str]]] = None,
    vmax: Optional[Union[str, List[str]]] = None,
    vcenter: Optional[Union[str, List[str]]] = None,
    **kwargs
): 
    
    from palettable.colorbrewer.sequential import Purples_7
    
    avaliable_kind = ['spatial', 'cell', 'contour']
    if kind not in avaliable_kind:
        raise ValueError(
            f'kind must be one of the {avaliable_kind}')
    
    # default dot size and cmap
    is_contour = False
    if kind in ['spatial', 'contour']:
        cmap = set_value(cmap, 'Spectral_r')
        
        if kind == 'spatial':
            size = 1.2 if dot_size is None else dot_size
            # TODO
        else:
            is_contour = True
    else:
        size = None if dot_size is None else dot_size
        cmap = set_value(cmap, Purples_7.mpl_colormap)    
        
    # coord and value_df 
    coords, colnames = _parse_coord(coords, is_contour)
    colors = value_df.columns
    assert len(colors) > 0, 'No colors found in value_df'
    value_df[colnames] = coords 

    # generate row and col numbers
    if splitby is None:
        cols = None
        nrows, ncols = _generate_layout(colors, nrows, ncols)
    else:
        cols = categorical_order(splitby)
        nrows, ncols = (len(colors), len(cols))
        groups = len(colors) * cols
    
    # genrator for facet data
    colnames = [colnames] if isinstance(colnames, str) else colnames
    def facet_data(df, row, col=None, is_genes=None):
        is_genes = set_value(
            is_genes, np.array([False] * len(colors)))
        
        for is_gene, var in zip(is_genes, row):
            data = df[[*colnames, var]]
            
            if col is not None:
                for group in col:
                    split_mask = splitby == group
                    
                    yield data[split_mask], is_gene, var
            else:
                yield data, is_gene, var
                
    facet_df = facet_data(value_df, colors, cols, is_genes)

    # plot
    with mpl.style.context(style):    
        if axs is None:
            if transpose_axes:
                nrows, ncols = ncols, nrows
                
            figsize = (ncols * 1.05 * height, nrows * height)
            fig, axs = plt.subplots(
                nrows=nrows, 
                ncols=ncols,
                figsize=figsize, 
                frameon=frameon,
                sharex=True, 
                sharey=True,
                # constrained_layout=True
            ) 
            axs = np.array(axs)
            axs = axs.T if transpose_axes else axs
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
        else:
            fig = plt.gcf() 
            
        axs = axs.flatten() if hasattr(axs, 'shape') else [axs]
        
        # scatter with colormap
        if isinstance(vmin, str) or not isinstance(vmin, Sequence):
            vmin = [vmin]
        if isinstance(vmax, str) or not isinstance(vmax, Sequence):
            vmax = [vmax]
        if isinstance(vcenter, str) or not isinstance(vcenter, Sequence):
            vcenter = [vcenter]
        
        for (i, (_ax, (data, is_gene, color))) in enumerate(zip(axs, facet_df)):

            coords     = data[colnames].values
            color_data = data[color].values
            
            vmin_float, vmax_float, vcenter_float = _get_vboundnorm(
                vmin, vmax, vcenter, i, color_data)
            if kind == 'spatial':
                dot_size = _unit_scatter_size(_ax, coords) * size
            elif kind == 'contour':
                vmin_float = set_value(vmin_float, color_data.min())
                vmax_float = set_value(vmax_float, color_data.max())
            else:
                dot_size = (size if size is not None 
                            else 120000 / coords.shape[0])
            norm = check_colornorm(vmin_float, vmax_float, vcenter_float)
            
            # scatter or hexbin
            if kind != 'contour':
                if not hexbin:
                    sc = _ax.scatter(
                        x=coords[:, 0], 
                        y=coords[:, 1],
                        c=color_data, 
                        marker='.', 
                        s=dot_size, 
                        lw=stroke,
                        cmap=cmap, 
                        norm=norm,
                        rasterized=raster,
                        **kwargs
                    )            
                else:
                    sc = _ax.hexbin(
                        x=coords[:, 0],
                        y=coords[:, 1], 
                        C=color_data, 
                        gridsize=gridsize,
                        lw=0, 
                        cmap=cmap, 
                        norm=norm,
                        rasterized=raster,
                        **kwargs
                    )
            else:
                gdf = GeoDataFrame(geometry=coords.flatten())
                # colorbar mapping
                sc = cm.ScalarMappable(norm=norm, cmap=cmap)
                _ax = gdf.plot(
                    column=color_data, 
                    vmin=vmin_float, 
                    vmax=vmax_float,
                    cmap=cmap, 
                    lw=0.01, 
                    legend=False, 
                    ax=_ax,
                    rasterized=raster,
                    **kwargs
                )
                
            # italicize gene names
            title_style = 'italic' if is_gene else 'normal'
            cb_kw = dict(
                title = color if splitby is not None else '',
                title_style = title_style
            )
            sub_kw = dict(
                text = color if splitby is None else groups[i],
                style = title_style if splitby is None else 'normal'
            )
            
            cbar_kw.update({'orientation': cbar_orientation})
            cb_kw.update(cbar_kw)
            sub_kw.update(subtitle_kw)
            
            plot_subtitle(_ax, **sub_kw) 
            
            # draw color bar 
            if not share_cbar and legend:
                cb = plt.colorbar(
                        sc,
                        ax=_ax,
                        aspect=4.5,  # length / width
                        shrink=0.28, # shrink
                        orientation=cbar_orientation,
                        # ticks=[0, 1],
                    )
                
                set_colorbar(cb, **cb_kw)
            
            # coord labels
            if labels:
                _ax.set(xlabel=f'{pl_cls}_1', ylabel=f'{pl_cls}_2')
            else:
                _ax.spines[:].set_visible(False)
                
            # TODO 
            _ax.set(xticks=[], yticks=[])
            if aspect == 'force_equal':
                set_force_equal_aspect(_ax)
            else:
                _ax.set_aspect(aspect)
                
            # add arrow on the left bottom
            if (i == (nrows - 1) * ncols) & plot_arrow:
                arrow_kw.update({'arrow_label': pl_cls, 
                                 'fontsize': 'large'})
                draw_arrow(
                    ax=_ax, 
                    invet_axis=invert_y,
                    color=_auoto_color(style),
                    **arrow_kw)
            
        # invert y axis
        if invert_y:
            _ax.invert_yaxis()
        
        # remove superfluous axis    
        if (len(colors) < len(axs)) & (splitby is None):
            for _ax in axs[len(colors):]:
                _ax.remove()
        
        # share colorbar
        # TODO now we have better ways to do this 
        # new in 3.7 let position=right
        if share_cbar and legend: 
            
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes(
                [0.93, 0.4, 0.05 / ncols, 0.2 / nrows])  # (1, 0.04) (4,)
            cb = fig.colorbar(sc, cax=cbar_ax)  
            # cb = fig.colorbar(sc, ax=np.array(axs).reshape(nrows, ncols)[:, :], 
            #                   shrink=0.28, aspect=4, location='right', pad=0.1)
            set_colorbar(cb, **cb_kw)
            
        # save plot
        if save_path:
            save_fig(save_path, transparent=False)

        if tight_layout:
            plt.tight_layout()

    return axs[0] if len(axs) == 1 else axs 
    
# support mask for feature plot
def featureplot(
    adata: AnnData, 
    features: Union[Sequence[str], str], 
    reduction: str = 'umap', 
    splitby: Optional[str] = None, 
    use_raw: bool = None,
    contour_col: Optional[str] = 'contour',
    components: Sequence[int] = (0, 1),
    layer: Optional[str] = None,
    ax: Optional[Union[Axes, List[Axes]]] = None,
    **kwargs
):
    """spatial feature plot

    Parameters
    ----------
    adata : AnnData
        anndata
    dot_size : `float`, optional
        point size, by default 1.2
    features : `list`, optional
        gene or column in `adata.obs`
    ncols : `int`, optional
        subplots cols, by default 4
    nrows : Optional[int], optional
        subplots rows, by default None
    scale : `int`, optional
        subplot scale, by default 5
    invert_y : `bool`, optional
        whether invert y axis, by default False
    labels : `bool`, optional
        axis labels, by default False
    stroke : `float`, optional
        line width, by default 0.25
    cmap : `str`, optional
        color map, by default 'viridis'
    hexbin : `bool`, optional
        draw hexbin, by default False
    share_cbar: `bool` optional
        whether share colorbar, by default False
    set_cbar_kw: `dict` optional
        set colorbar kwargs, by default {}
        - available kwargs:
            - theme: by default 'min_max'. 'ggplot', 'classic' also supported
            - orientation: 'vertical' or 'horizontal'
            - position `cb.ax.set_position(position)`
            
    subtitle_kw: `dict` optional
        set subplot title kwargs, by default {}
        - available kwargs:
            - theme by default 'classic', 'ggplot' also supported
            - text_loc text location eg. (0.5, 0.5) as center
            - facecolor facecolor of title by default '0.9' eg.'#4DBBD57F'
            - text_color color of title text by default 'black'
        
    save_path : Optional[str], optional
        save path, by default None

        
    Examples:
    -------- 
    >>> feature_plot(
            adata, features='Myh6', splitby='batch')
    
    hexbin plot
    >>> feature_plot(adata, hexbin=True)
    
    advanced usage
    >>> feature_plot(
            adata, dot_size=1.2, ncols=3, 
            features=['Prrx1', 'Foxf1', 'Foxf1', 'Myh6', 'Myh6'], 
            cmp='Spectral_r', wspace=0.15, hspace=0.15, 
            sub_title_kw={'theme': 'ggplot', 'text_loc': (0.1, 0.5), 
                          'facecolor': '#4DBBD57F', 'text_color': 'white'},
            set_cbar_kw={'theme': 'ggplot'},
            invert_y=True, frameon=False, hexbin=True, labels=False)
    """
    
    
    if adata.raw is None and use_raw:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check.")
    
    if reduction == 'spatial':
        kwargs['kind'] = 'spatial'
        
    if reduction == 'contour':
        kwargs['kind'] = 'contour'
        
    if isinstance(features, str):
        features = [features]  
        
    value_df = fetch_data(
        adata, features, layer=layer, use_raw=use_raw)
    
    coords  = get_cell_embeddings(
        adata, reduction=reduction, 
        contour_col=contour_col, as_frame=False)
    
    is_genes = [
        is_gene_name(adata, feature, use_raw=use_raw) 
        for feature in value_df.columns
    ]
    
    if splitby is not None:
        if splitby not in adata.obs.columns:
            raise ValueError(
                f'{splitby} is not in `adata.obs.columns`')
        else:
            splitby = adata.obs[splitby]
    
    axs = distribution_plot(
        coords=coords, value_df=value_df, 
        splitby=splitby, is_genes=is_genes,
        pl_cls=_basis2name(reduction), axs=ax, **kwargs)

    return axs

    
# ==================================================================================
#                                   dot plot
# ==================================================================================
from marsilea.base import RenderPlan
import marsilea.plotter as mp

def merge_and_count(color_list):
    # input: ['3', '3', '2', '1', '3'] 
    # output: length -> [2, 1, 1, 1], color -> ['3', '2', '1', '3']
    from itertools import groupby
    Groupby = groupby(color_list)
    count_label = [(i, len(list(j))) for i,j in Groupby]
    
    return count_label

def generate_cb_df(count_label, start=0):  
    cb_df = pd.DataFrame(count_label, columns=['colors', 'num'])
    cb_df['x'] = cb_df.num.cumsum() - cb_df.num * 0.5 - start
 
    return cb_df

class CircleAnno(RenderPlan):
    
    def __init__(self, labels, radius=0.2, edgecolor=None, edgewidth=None, text=False,
                 label=None, label_loc=None, props=None, palette=None, **kwargs):
        
        self.set_label(label, label_loc, props)
        labels = self.data_validator(labels, target="1d")
        self.radius = radius
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.kwargs = kwargs
        
        self.palette = palette
        
        # self._legend_kws = dict(title=self.label, size=1)

        self.set_size(.1)
        self.set_data(labels)
        
    
    def render_ax(self, spec):
        import matplotlib.patches as mpatches
        
        data = spec.data
        ax = spec.ax
        text_props = spec.params

        lim = (0, len(data))
        
        if text_props is None:
            text_props = [{}] * len(data)
            
        coords = np.arange(0.5, len(data) + 0.5)
        params = {'ha': 'center', 'va': 'center'}
        
        if self.is_flank:
            ax.set_ylim(lim)
        else:
            ax.set_xlim(lim)
        
        for s, c, p in zip(data, coords, text_props):
            x, y = (0.5, c) if self.is_flank else (c, 0.5)
            options = {**params, **p}
            circle = mpatches.Circle((x, y), self.radius, 
                                     ec=self.edgecolor)
            ax.add_artist(circle)
            ax.text(x, y, s=int(c), **options)
            
        # ax.set_axis_off()
        ax.set_aspect('equal')
        
        
class RecAnno(RenderPlan):

    def __init__(self, data, width=1, edgecolor=None, edgewidth=None, text=None,
                 height=.1, side='top', label=None, label_loc=None, props=None, text_kw={},
                 palette=None, kind='rectangle', **kwargs):
        
        self.set_label(label, label_loc, props)
        self.width = width
        
        self.edgecolor = edgecolor
        self.palette = palette
        
        self.edgewidth = edgewidth
        self.kwargs = kwargs
        self.kind = kind
        self.side = side
        
        self.add_text = set_value(text, kind=='bracket')
        self.text_kw = text_kw
        
        self._legend_kws = dict(title=self.label, size=1)
        data = np.asarray(data)

        self.set_size(height)
        self.set_side(side)
        self.set_data(data)

    def render_ax(self, spec):
        ax = spec.ax
        data = spec.data
        ax.axis('off')

        lim = len(data)
        
        count_label = merge_and_count(data)
        data = generate_cb_df(count_label)
        
        color = self.edgecolor
        fill = self.palette if self.kind == 'rectangle' else 'none'
        
        order = list(dict.fromkeys(data.colors))
        if isinstance(color, dict):
            if len(color) != len(order):
                raise ValueError(
                    'The length of color dict must be '
                    'equal to the number of unique values in data'
                )
            color = data.colors.map(color)
        else:
            if self.kind == 'bracket':
                color = get_palette(
                    ['black'], n_colors=len(order), n_labels=len(order))
                color = data.colors.map(dict(zip(order, color)))
        
        if isinstance(fill, dict):
            if len(fill) != len(order):
                raise ValueError(
                    'The length of fill dict must be '
                    'equal to the number of unique values in data'
                )
            fill = data.colors.map(fill)
        else:
            if self.kind == 'rectangle':
                fill = get_palette(
                    fill, n_colors=len(order), n_labels=len(order))
                fill = data.colors.map(dict(zip(order, fill)))
            
        geo_dict = {
            'rectangle': PolyCollection,
            'bracket': LineCollection}
            
        data = data.assign(
            x_min = data['x'] - 0.5 * data.num * self.width,
            x_max = data['x'] + 0.5 * data.num * self.width,
            y_min = 0,
            y_max = 1)
        limits = zip(data["x_min"], data["x_max"], 
                     data["y_min"], data["y_max"])
        
        if self.is_flank:
            verts = [[(b, l), (t, l), (t, r), (b, r)] 
                     for (l, r, b, t) in limits]
            xs, ys = [1.01]*len(order), data.x
            ax.set_ylim(lim, 0)
            ax.set_xlim(0, 1.1)
        else:
            verts = [[(l, b), (l, t), (r, t), (r, b)] 
                     for (l, r, b, t) in limits]
            xs, ys = data.x, [1.01]*len(order)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, 1.1)
        
        col = geo_dict[self.kind](
            verts,
            facecolors=fill,
            edgecolors=color,
            linewidth=self.edgewidth,
            rasterized=False,
            **self.kwargs
        )
        ax.add_collection(col)
        
        if self.add_text:
            rotation = self.text_kw.pop('rotation', 0)
            for x, y, t in zip(xs, ys, order):
                ax.text(
                    x, y, t, rotation=rotation, 
                    **_set_labelsalign(rotation, self.side)
                )


def calc_dendrogram(adata, groupby, features=None, 
                    reduction=None, n_pcs=50, key_added=None,
                    cor_method='pearson', method='complete', 
                    use_raw=None, layer=None, inplace=True):
    
    from pandas.api.types import CategoricalDtype
    
    if isinstance(groupby, str):
        groupby = [groupby]
        
    for group in groupby:
        if group not in adata.obs_keys():
            raise ValueError(
                'groupby has to be a valid observation. '
                f'Given value: {group}, valid observations: {adata.obs_keys()}'
            )
        if not isinstance(adata.obs[group].dtype, CategoricalDtype):
            raise ValueError(
                'groupby has to be a categorical observation. '
                f'Given value: {group}, Column type: {adata.obs[group].dtype}'
            )
        
    if reduction is None:
        if 'X_pca' not in adata.obsm_keys():
            reduction = 'X'
        else:
            reduction = 'pca'
    
    if reduction == 'X':
        if adata.X.shape[1] > 50:
            raise ValueError(
                f" {adata.X.shape} genes to plot. "
                    " Consider using `reduction='pca'` ") # TODO to compute pca
        else:
            data = pseudobulk_expression(
                adata, groupby, features, 
                use_raw=use_raw, layer=layer)
    else:
    # merge multiple groups
        categorical = adata.obs[groupby[0]]
        if len(groupby) > 1:
            for group in groupby[1:]:
                # create new category by merging the given groupby categories
                categorical = (
                    categorical.astype(str) + "_" 
                    + adata.obs[group].astype(str)
                ).astype('category')
                
        join_name = "_".join(groupby)
    
        data = get_cell_embeddings(
            adata, reduction=reduction, n_components=n_pcs)
        
        data[join_name] = categorical
        categories = data[join_name].cat.categories
        data = data.groupby(join_name).mean()
            
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    corr_matrix = data.T.corr(method=cor_method)
    corr_condensed = distance.squareform(1 - corr_matrix)
    linkage = hierarchy.linkage(
        corr_condensed, method=method, optimal_ordering=False)
    
    dendro_info = hierarchy.dendrogram(
        linkage, labels=list(categories), no_plot=True)
    
    dat = dict(
        linkage=linkage,
        groupby=groupby,
        use_rep=reduction,
        cor_method=cor_method,
        linkage_method=method,
        categories_ordered=dendro_info['ivl'],
        categories_idx_ordered=dendro_info['leaves'],
        dendrogram_info=dendro_info,
        correlation_matrix=corr_matrix.values,
    )
    
    if inplace:
        if key_added is None:
            key_added = f'dendrogram_{join_name}'
        logger.info(f'Storing dendrogram info using `.uns[{key_added!r}]`')
        adata.uns[key_added] = dat
    else:
        return dat
    
# def dendrogram()
    
def _get_dendrogram_key(adata, dendrogram_key, groupby):
    # the `dendrogram_key` can be a bool an NoneType or the name of the
    # dendrogram key. By default the name of the dendrogram key is 'dendrogram'
    if not isinstance(dendrogram_key, str):
        if isinstance(groupby, str):
            dendrogram_key = f'dendrogram_{groupby}'
        elif isinstance(groupby, list):
            dendrogram_key = f'dendrogram_{"_".join(groupby)}'

    if dendrogram_key not in adata.uns:

        logger.warning(
            f"dendrogram data not found (using key={dendrogram_key}). "
            "Running with default parameters. For fine "
            " to run `sc.tl.dendrogram` independently."
        )
        calc_dendrogram(adata, groupby, key_added=dendrogram_key)

    if 'dendrogram_info' not in adata.uns[dendrogram_key]:
        raise ValueError(
            f"The given dendrogram key ({dendrogram_key!r}) does not contain "
            "valid dendrogram information."
        )

    return dendrogram_key
    
def _matrix_normalize(mat, axis=0, method='minmax', **kwargs):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    
    mat = np.asarray(mat)
    mat = mat.T if axis == 1 else mat
    
    if method == 'minmax':
        scaler = MinMaxScaler(**kwargs)
    elif method == 'zscore':
        scaler = StandardScaler(**kwargs)
    else:
        # callables
        scaler = method
    
    mat = scaler.fit_transform(mat)
    return mat.T if axis == 1 else mat

# TODO: bug fix
# 1. replicate the same gene in different groups

def dotplot(
    adata: AnnData, 
    features: Union[str, list, dict],
    groupby: Union[str, list], 
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    feature_colors: bool = True, 
    group_colors: bool = False,  # TODO
    feature_cluster: bool = False, # right or left
    group_cluster: bool = False, # top or bottom
    group_order: Optional[List] = None,
    cell_trunk: Optional[List] = None, # TODO
    fontsize: Optional[int] = 'medium',
    rotation: Optional[float] = None,
    # fontstyle: str = 'italic',
    kind: Literal['matrix', 'dot'] = 'dot',
    flavor: str = 'scanpy', # minmax or zscore
    add_text: bool = False,
    col_min: float = -2.5,
    col_max: float = 2.5,
    dot_min: Optional[float] = None,
    dot_max: Optional[float] = None,
    scale_range: Tuple[float, float] = (0, 150),
    scale: str = 'gene', 
    tree_kw: dict = {},
    color_kw: dict = {},
    dendro_key: Optional[str] = None,
    cmap: str = "Purples", # "RdBu_r"
    palette: Optional[Union[Sequence, str]] = None,
    swap_axes: bool = False,
    show: bool = True,
    size_legend_kws = {},
    colorbar_kws = {},
    marker: str = 'o',
    **kwargs
):
    """
    clustered dot plot
    
    Parameters
    ----------
    adata : AnnData
        anndata object
    group_by : Union[str, list]
        group key in `adata.obs.columns`
    features : list
        list of genes
    row_colors : bool
        whether to draw row colors, by default True
        not implemented yet
    col_colors : bool, 
        whether to draw col colors(gene annotation), by default False
    row_cluster : bool
        clustered by cells, by default None
    col_cluster : bool
        clustered by genes, by default None
    scale : int, optional
        scale axis, by default 1
    add_text : bool, False, optional
        whether to add text on cluster bar, by default False, 
        when col_cluster is True it will draw inner text
    col_min : float
        min value of color bar
    col_max : float
        max value of color bar
    dot_min : float
        min value of dot size
    cb_width : float, optional
        color bar width [0, 1], by default 1 
    fontsize : int 
        fontsize 
    rotation : Union[int, float]
        rotation of xticklabels
    fontstyle : str
        fontstyle of xticklabels
    dpt_kw : dict
        keyword arguments passed to `_dpt_plot`
    tree_kw : dict
        keyword arguments passed to `denrogram`
    legend_position : str, optional
        legend position, by default 'right'
    dendrogram_ratio : float, optional
        dendrogram height, by default 0.12 of figure height/widht
    colors_ratio : float, optional
        annotation rectangle height, by default 0.025 of figure height/widht
    cmap : str
        color map, by default "Purples"
    palette : Optional[Sequence], optional
        palette for annotation, by default None  
        
    Examples
    --------
    >>> dot_plot(adata, group_by=['celltype'], features=marker_gene,
                 row_colors=True, col_colors=False, cmap='Purples', scale=0.8, 
                 rotation=90, text_size=10, cb_width=0.9, add_text=True, cb_height=3)
        
    """
    
    prop = FontProperties(size=fontsize, weight='normal')
    
    if len(groupby) == 2:
        if isinstance(features, str):
            features = [features]
        if len(features) != 1:
            raise ValueError('only one feature is allowed')
    
    if isinstance(groupby, str):
        groupby = [groupby]
    
    group_vec = adata.obs[groupby[0]]
    group_order = categorical_order(group_vec, group_order)
    
    # likage order muset consistent with cell order
    if group_cluster:
        if len(group_order) < group_vec.nunique():
            dendro = calc_dendrogram(
                adata[group_vec.isin(group_order)],
                groupby[0], inplace=False)
        else:
            dendro_key = _get_dendrogram_key(adata, dendro_key, groupby)
            dendro = adata.uns[dendro_key]
            
        group_order = dendro['categories_ordered']
    
    anno_to_genes = None
    notin = []
    def _filter_features(x):
        if isinstance(x, str):
            x = [x]
        x = np.asarray(x)
        mask = np.array([is_anno(adata, i, use_raw) for i in x])
        notin.extend(list(x[~mask]))
        return list(x[mask])
    
    if isinstance(features, dict):
        # input annotation: list of genes
        if set(features.keys()) == set(group_order):
            features = {k: features[k] for k in group_order}
            palette = assign_palette(adata, groupby[0]).to_dict()
                
        anno_to_genes = {
            k: _filter_features(v) 
            for k, v in features.items()
            if len(v) > 0}

        features = list(chain(*list(anno_to_genes.values())))
    else:
        features = _filter_features(features)

    if notin:
        logger.warning(
            f'{notin} not in .obs_names or .var_names')
        
    # get average expression and pct expressed
    data = pseudobulk_expression(
        adata, 
        groupby=groupby, 
        features=features, 
        with_ratio=kind=='dot', 
        scale=True if flavor == 'seurat' else False,
        use_raw=use_raw,
        layer=layer
    )
    
    # TODO whether to support duplicate genes
    exp, ratio = data if kind == 'dot' else (data, None)
    row_names = exp.index # save row names
    
    # scale data
    if flavor == 'seurat':
        if scale is not None:
            exp = _matrix_normalize(exp, method='zscore')
            exp = exp.clip(col_min, col_max)
        else:
            # log1p exp values
            exp = np.log1p(exp)
    elif flavor == 'scanpy':
        axis = 0 if scale=='gene' else 1
        exp = _matrix_normalize(
            exp, axis=axis, method='minmax')
    else:
        pass
    
    # --------
    # dot plot
    # --------
    prop = dict(
        size=fontsize, weight='normal')
    
    # ---size legend---
    if kind == 'dot':
        array = np.asarray(ratio).flatten()
        smin = np.amin(array)
        smax = np.amax(array)
        
        if dot_max is None:
            dot_max = smax // 5 * 5 # < smax
        else:
            if dot_max < 0 or dot_max > 100:
                raise ValueError(
                    "`dot_max` value has to be between 0 and 1")
            
        if dot_min is None:
            dot_min = 0
        else:
            if dot_min < 0 or dot_min > 100:
                raise ValueError(
                    "`dot_min` value has to be between 0 and 1")
            
        diff = dot_max - dot_min
        if 30 < diff <= 50:
            step = 10
        elif diff <= 30:
            step = 5
        else:
            step = 25
        
        size_norm = Normalize(vmin=dot_min, vmax=dot_max)
        size_range = np.arange(dot_max, dot_min, step * -1)[::-1] 
        show_at = np.interp(size_range, [smin, smax], [0, 1])
        func = lambda x: np.round(x, decimals=0).astype(int)
        
        _size_legend_kws = dict(
            func = np.vectorize(func),
            show_at = show_at,
            title = 'Fraction of cells(%)',
            labelspacing = 0.8,
            prop = prop,
            title_fontproperties=prop
        )
        _size_legend_kws.update(size_legend_kws)
    
    # ———colorbar———
    _colorbar_kws = dict(
        width = 1.5, height = 5.7,
        title = 'Mean expression',
        title_fontproperties = prop
    )
    _colorbar_kws.update(colorbar_kws)
    
    # ---main plot---
    if rotation is None:
        rotation = 0 if swap_axes else 90
    
    if swap_axes:
        genes_labels_side = 'left'
    else:
        genes_labels_side = 'bottom'
    
    feature_style = [
        'italic' 
        if is_gene_name(adata, i, use_raw=use_raw) 
        else 'normal' for i in features]
    text_props = {'style': feature_style}
    genes_labels = mp.Labels(
        labels=features, 
        rotation=rotation, 
        text_props=text_props,
        **_set_labelsalign(
            rotation, genes_labels_side),
        **prop, 
    )
    cells_labels = mp.Labels(labels=row_names, **prop)
    
    if swap_axes:
        xlabels, ylabels = cells_labels, genes_labels
    else:
        xlabels, ylabels = genes_labels, cells_labels

    if kind == 'dot':
        if swap_axes:
            exp, ratio = exp.T, ratio.T
            
        h = ma.SizedHeatmap(
            size=ratio, 
            color=exp, 
            cmap=cmap, 
            marker=marker,
            size_norm=size_norm,
            sizes=scale_range, 
            size_legend_kws=_size_legend_kws,
            color_legend_kws=_colorbar_kws,
            name='dpt',
            **kwargs
        )
    else:
        if swap_axes:
            exp = exp.T
            
        h = ma.Heatmap(
            data=exp, 
            cmap=cmap, 
            name='dpt',
            cbar_kws=_colorbar_kws,
            **kwargs
        )
        
    h.add_left(ylabels, pad=.05)
    h.add_bottom(xlabels, pad=.05)
    
    # ----------
    # dendrogram
    # ----------
    if group_cluster: 
        if swap_axes:
            side = 'top'
        else:
            side = 'right'
            
        method, linkage = dendro['linkage_method'], dendro['linkage']

        _tree_kw = dict(
            method = method, linkage = linkage,
            pad = .05, size = .8, side = side
        )
        _tree_kw.update(tree_kw)
        h.add_dendrogram(
            name='dendrogram', **_tree_kw)

    if feature_cluster: # TODO circle anno
        pass
        
    if feature_colors:
        if swap_axes:
            side = 'right'
        else:
            side = 'top'
            
        _color_kw = dict(
            pad = .05, width = .9, side = side,
        )
        _color_kw.update(color_kw)
        pad = _color_kw.pop('pad')
        
        if anno_to_genes is not None:
            anno_label = [k for k, v in anno_to_genes.items() for _ in v]
            anno = RecAnno(anno_label, palette=palette, **_color_kw)
            h.add_plot(anno.side, anno, name='colors', pad=pad)
            
    if show:
        h.add_legends()
        h.render()
    else:
        return h
        
# ==================================================================================
#                               violin plot
# ==================================================================================
        
# deprecated
def plot_qc_vln(adatas,
                qc_key=None,
                groupby=None,
                figsize=(14, 4), 
                jitter=0.4,
                inner='box'
                ):  # TODO: count_key gene_key, other qc alternative to choose

    # TODO: wrapper in qc?
    # if not all([set(qc_key).isubset(adata.obs.columns) for adata in adatas]):
    #     raise ValueError("there is no qc value in your data run `sc.pp.calculate_qc_metrics` first")

    if qc_key is None:
        qc_key = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
        
    fig, ax = plt.subplots(1, len(qc_key), figsize=figsize)

    cmp = ['#FF5A5F', '#FFB400', '#007A87']
    pl = qc_key

    if groupby is not None:
        pl.append(groupby)
        data = pd.concat([adata.obs[pl] for adata in adatas])
    else:
        data = adatas.obs[pl]

    for k, (c, v) in enumerate(zip(cmp, pl)):
        sns.violinplot(
            x=groupby, y=v, 
            data=data, ax=ax[k],
            inner=inner, color=c,
            scale='width'
            )
        sns.stripplot(
            x=groupby, y=v, 
            data=data, ax=ax[k],
            color=".3", orient='v', 
            jitter=jitter, size=1,
            )
        
        ax[k].set(xlabel=qc_key[k].capitalize(), ylabel='')
        ax[k].set_rasterization_zorder(0)
        
    ax[0].set_ylabel('Counts')
    fig.tight_layout()
    return ax


def plot_qc_hist(adata, qc_key=None, figsize=None, **kwargs):
    
    if qc_key is None:
        qc_key = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    
    if figsize is None:
        figsize = (len(qc_key) * 4, 4)
        
    fig, axs = plt.subplots(1, len(qc_key), figsize=figsize)
    
    for i, key in enumerate(qc_key):
        ax = sns.histplot(adata.obs[key], kde=False, bins=40, 
                          ax=axs[i], **kwargs)
        ax.set_ylabel('')
    fig.tight_layout()
    
    return ax


def vlnplot(
    adata: AnnData, 
    features: List[str], 
    groupby: Optional[str] = None, 
    splitby: Optional[str] = None,
    kind: str = 'violin',
    ncols: Optional[int] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    height: Optional[int] = 2.5, 
    aspect: Optional[float] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    scale: Literal['area', 'count', 'width'] = 'width', # deprecated in seaborn 0.13
    # density_norm: Literal['area', 'count', 'width'] = 'width',
    rotation: Optional[int] = None, 
    ylabel: str = '',
    sig: bool = False, 
    log: bool = False,
    sharex: bool = True,
    sharey: bool = False,
    split: bool = False,
    # wspace: float = 0.2,
    # hspace: float = 0.2,
    comp_pairs: Optional[List[Tuple]] = None,
    stata_configure: dict = {},
    inner: Optional[Literal['box', 'quartile', 'point']] = 'quart',
    jitter: Union[float, bool] = False,
    edgecolor: str = '.5',
    style: str = 'fast',
    legend: bool = True,
    tight_layout : bool = False,
    palette: Optional[str] = None, 
    fill: bool = True, 
    jitter_kws: dict = {},
    **kwargs
    ): 
    """
    gen feature violin plot

    Parameters
    ----------
    adata : AnnData
        Anndata
    features : List
        a list of features eg. genes
    groupby : str
        group key eg. `batch` or `celltype`
    use_raw : bool, optional
        whether to use raw expression value, by default False
    kind : str, optional
        plot kind, by default 'violin', also support 'box'
    ncols : Optional[int], optional
        ncols of subplot, by default None
    rotation : Optional[int], optional
        x label rotation, by default 45
    y_label : str, optional
        y label, by default 'Expression Level'
    splitby : Optional[str], optional
        split key, by default None, eg. the key of `case|control`
    size : Optional[int], optional
        plot size, by default 3
    aspect : Optional[float], optional
        aspect, by default None
    scale : Literal['area', 'count', 'width'], optional
        The method used to scale the width of each violin. 
        If area, each violin will have the same area. 
        If count, the width of the violins will be scaled 
        by the number of observations in that bin. 
        If width, each violin will have the same width.
    restrict_to : Optional[list], optional
        resctrict to the group of the element of `groupby`, by default None
    log : Optional[bool]
        whether to log the expression value, by default False
    stata_configure : dict, optional
        stata configure, by default {}
    comp_pairs : Optional[List[Tuple]], optional
        comparison pairs, by default None
    inner : str, optional
        inner, by default 'quart', 
        available options are 'box', 'quart', 'point', 'stick', None
    jitter : Union[float, bool], optional
        add jitter, by default False
    kwargs : dict, optional
        kwgs of `sns.catplot`
        
    Examples
    --------
    >>> vln_plot(adata, ['Cwc22', 'Slc26a6'], 'celltype', 
                 ncols=3, aspect=None, kind='violin', hue='batch', restrict_to=['PTS1-S2'])
    """
    
    variable_key = []
    n_groups = 1
    n_splits = 0
    x = groupby
    hue = splitby
        
    if groupby is not None:
        variable_key.append(groupby)
        order = categorical_order(adata.obs[groupby], order)
        n_groups = len(order)
        
        if splitby is None:
            palette = assign_palette(
                adata, groupby, palette=palette).to_dict()
            hue = groupby
            hue_order = order
    else:
        splitby = None
        
    if splitby is not None:
        variable_key.append(splitby)
        hue_order = categorical_order(adata.obs[splitby], hue_order)
        n_splits = len(hue_order)
        
        palette = assign_palette(
            adata, splitby, palette=palette).to_dict()
    
    if len(variable_key) == 0:
        variable_key = None
   
    data = generate_var_value_df(
        adata, variable_key, features, use_raw=use_raw, layer=layer)
    
    # TODO oritentation = 'vertical' or 'horizontal'
    # TODO bug in seaborn 0.13.0 change 'x' to 'hue' 
    single_plot_kws = dict(
        x=x, 
        y='value', 
        order=order,
        hue=hue,
        hue_order=hue_order
    )
    kwargs.update(single_plot_kws)
    
    if kind == 'violin':
        if (n_splits > 1) & split:
            if n_splits == 2:
                split = True
            else:
                split = False
        else:
            split = False
            
        cat_kws = dict(
            inner=inner, 
            split=split, 
            cut=0, 
            width=0.95,
            linewidth=0.5, 
            scale=scale
        )
    elif kind == 'box':
        cat_kws = dict(
            linewidth=1,
            fliersize=1 if jitter else 0
        )
    elif kind == 'boxen':
        cat_kws = dict(
            linewidth=0.5,
            flier_kws={'s': 1}
        )
    elif kind == 'bar':
        cat_kws = {}
        
    cat_kws.update(kwargs)
    
    # plot
    aspect = set_value(
        aspect, (n_groups + n_splits) * 0.21 + 0.3)
 
    with mpl.style.context(style):
        g: sns.FacetGrid = sns.catplot(
            data=data, 
            col='variable', 
            kind=kind,
            col_wrap=ncols,
            sharex=sharex, 
            sharey=sharey, 
            height=height, 
            aspect=aspect, 
            legend=False,
            saturation=1, 
            palette=palette, 
            **cat_kws
            # gridspec_kws={"wspace":0.4}
        )
        
    axs = g.axes.flatten()
    
    # TODO plot_and_annotate_facets wrapper (too dirty)
    for i, ax in g.axes_dict.items():  # (col_name, ax)
        single_df = data[data['variable'] == i]
        single_plot_kws['data'] = single_df
        cat_kws['data'] = single_df
         
        # add significance test
        # TODO: input precompute P
       
        if sig:
            try:
                from statannotations.Annotator import Annotator
            except ImportError:
                raise ImportError(
                    'Please install the statannotations' 
                    '`pip3 install statannotations`')
            
            DEFAULT = {
                'test': 'Mann-Whitney', 
                'text_format': 'star',
                'comparisons_correction':"BH",
                'correction_format': 'replace',
                'loc':'inside'
            }
            
            if stata_configure is not None:
                DEFAULT.update(stata_configure)
            # wrapper below
            if (comp_pairs is None) & (groupby is not None):
                comp_pairs = list(combinations(order, 2))
                if splitby is not None:
                    group_col = single_df[groupby]
                    split_col = single_df[splitby]
                    groups = group_col.unique()
                    
                    comp_pairs = [
                        list(zip((group, group), cat)) 
                        for group in groups 
                        for cat in list(
                            combinations(
                                split_col[group_col==group].unique(), 2)
                            )
                        ]
                
            annotator = Annotator(pairs=comp_pairs, 
                                  plot=f'{kind}plot',
                                  ax=ax, **cat_kws) 
            
            annotator.configure(**DEFAULT)
            annotator.apply_and_annotate()
        
        if jitter:
            jitter = 0.4 if isinstance(jitter, bool) else jitter
            _jitter_kws = dict(
                color='.5',
                alpha=0.8,
                size=15000 / len(single_df)
            )
            jitter_plot = (
                partial(sns.stripplot, jitter=jitter, rasterized=True)
                if len(single_df) / n_groups > 500
                else partial(sns.swarmplot)
            )
            jitter_plot(
                ax=ax, 
                legend=None,  # seaborn 0.12.0
                **_jitter_kws,
                **single_plot_kws
            )

        set_rotation(ax, 'x', rotation)
        ax.set_title(
            i, 
            style=('italic'
                   if is_gene_name(adata, i, use_raw=use_raw) 
                   else 'normal'))

        for collection in ax.collections:
            collection.set_edgecolor(edgecolor)

    
    g.set_ylabels(ylabel) # Expression Level
    g.set_xlabels('')
   
    # if (len(g._legend_data) > 0) & (legend):
    if legend & (splitby is not None):
        #! this is a hack method, not a goood way
        style_color = _auoto_color(style)
        # colors = g._get_palette(data, splitby, hue_order, palette)
        # labels = g._legend_data.keys()
        
        colors = [palette[i] for i in hue_order]
        
        if kind == 'box':
            handles = [
                BOX(facecolor=color, 
                    edgecolor=style_color) 
                for color in colors]
            handlelength, handleheight = 1.75, 2
        else:
            handles = [
                Rectangle(
                    (0,0), 1, 1, 
                    facecolor=color, 
                    edgecolor=edgecolor) 
                for color in colors
            ]
            handlelength, handleheight = 1.25, 1.4
        
        plot_legend(
            fig=g.figure, 
            handles=handles, 
            labels=hue_order,
            labelcolor=style_color,
            handlelength=handlelength, 
            handleheight=handleheight)
        
    # deprecated in seaborn 0.13
    if not fill:
        for ax in g.axes.flatten():
        # hack to the matplotlib
        # now seborn support fill=False
        # https://stackoverflow.com/questions/75411222/
        # wrapper below
        #! only take effect on boxplot
            colors = []
            for collection in ax.collections:
                if isinstance(collection, PolyCollection):
                    colors.append(collection.get_facecolor())
                    collection.set_edgecolor(colors[-1])
                    collection.set_facecolor('none')
                    collection.set_linewidth(1.25)
                    
            if len(ax.lines) == 2 * len(colors):  # suppose inner=='box'
                for lin1, lin2, color in zip(
                    ax.lines[::2], ax.lines[1::2], colors):
                    lin1.set_color(color)
                    lin2.set_color(color)
                    
            if (len(g._legend_data) > 0) & (legend):
                for h in ax.legend_.legendHandles:
                    if isinstance(h, Rectangle):
                        h.set_edgecolor(h.get_facecolor())
                        h.set_facecolor('none')
                        h.set_linewidth(1.25)
        
    if tight_layout:
        g.fig.tight_layout()    
    
    if log:
        g.set(yscale="log")

    return axs[0] if len(g.axes_dict) == 1 else axs

    
# ==================================================================================
#                                   heatmap  # TODO waite for complex heatmap
# ==================================================================================

# TODO
# 1.base heatmap
# 2.cell gene expression heatmap
# 3.trajectory genes heatmap pandas rolling or numpy convolve


# ==================================================================================
#                                   ridge plot 
# ==================================================================================

def ridgeplot(
    adata: AnnData, 
    groupby:str, 
    features: Union[str, List], 
    palette: Optional[Union[str, Sequence]] = None, 
    title: Optional[str] = None,
    fill: bool = True,
    log: bool = False,
    order: Optional[Union[str, Sequence]] = None,
    **kwargs):
    try:
        from joypy import joyplot
    except ImportError:
        raise ImportError(
            "Please install joypy: `pip install joypy`")
    
    if isinstance(features, str):
        features = [features]
        
    data = fetch_data(adata, features + [groupby])
    group_num = data[groupby].nunique()
    if order is not None:
        data[groupby] = pd.Categorical(
            data[groupby], categories=order, ordered=True)
    
    if palette is None:
        if group_num <= 7:
            palette = mpl.cm.get_cmap('Set1')
        else:
            palette = mpl.cm.get_cmap('Spectral')
    else:
        if isinstance(palette, str):
            try:
                palette = mpl.cm.get_cmap(palette)
            except ValueError:
                raise ValueError(
                    "Please provide a valid colormap name.")
        elif isinstance(palette, Sequence):
            if len(palette) < group_num:
                raise ValueError(
                    "Please provide enough colors for the groups.")
    
    figsize = kwargs.pop('figsize', (6, 0.35 * group_num + 1.8))
    _, axs = joyplot(
        data, by=groupby, column=features, 
        ylim='own', colormap=palette, alpha=0.8, 
        fill=fill, title=title,
        xlabelsize='medium', ylabelsize='medium',
        figsize=figsize,
        **kwargs 
        )
    
    return axs



# ==================================================================================
#                               volcano plot
# ==================================================================================

# TODO: remove the rely on plotnine
# this name is not good
def all_volcano_plot(
    dge_result: Union[pd.DataFrame, str], 
    sep: str = ' ',
    groupby: str = 'celltype',
    x: str = 'avg_log2FC',
    y: str = 'p_val_adj',
    gene_col: str = 'gene',
    FC_cutoff: Optional[float] = None,
    P_cutoff: Optional[float] = None,
    fontsize: int = 5,
    top_n: int = 5,
    palette: Optional[Sequence] = None, 
    save_path: Optional[str] = None,
    text_size:int = 8,
    order: Optional[Sequence[str]]=None,
    cbar_width: float = 0.44,
    repel: bool = True,
    arrowprops: Optional[dict] = None,
    ):
    """
    volcano plot for all cell types

    Parameters
    ----------
    dge_result : Union[pd.DataFrame, str]
        differential gene expression result table
    celltype_col : str, optional
        celltype column, by default 'celltype'
    fold_change_col : str, optional
        fold change column, by default 'avg_log2FC'
    p_val_adj_col : str, optional
        ajust P value column, by default 'p_val_adj'
    FC_cutoff : Optional[float], optional
        fold change threshold, by default None
    P_cutoff : Optional[float], optional
        P value threshold, by default None
    fontsize : int, optional
        gene name text fontsize, by default 5
    top_n : int, optional
        top n gene to text, by default 5
    gene_col : str, optional
        gene name column, by default 'gene'
    cbar_color : dict, optional
        colorbar, by default None
    save_path : Optional[str], optional
        save path, by default None
    text_size : int, optional
        title text size, by default 12
    order : Optional[Sequence[str]], optional
        celltype order, by default None
    cbar_width : float, optional
        cbar width(0-0.5), by default 0.44
    adjust_texts : bool, optional
        whether adjust text, by default True
    arrowprops : Optional[dict], optional
        arrow props, by default None

    Returns
    -------
    ax
    """
    
    # read data
    rename = {
        x: 'avg_log2FC', 
        y: 'p_val_adj',
        groupby: 'celltype',
        gene_col: 'gene'}
    
    if isinstance(dge_result, str):
        if os.path.exists(dge_result):
            dge_result = (pd.read_csv(dge_result, index_col=0, sep=sep)
                            .rename(columns=rename))
        else:
            raise FileNotFoundError(f'{dge_result} not found')
        
    elif isinstance(dge_result, pd.DataFrame):
        dge_result = dge_result.rename(columns=rename)
    else:
        raise TypeError('dge_result should be a path or a dataframe')
    
    order = categorical_order(dge_result.celltype, order=order)
    dge_result['celltype'] = pd.Categorical(dge_result.celltype, categories=order)
    position_dict = dict(zip(order, np.arange(len(order)) + 0.5))
    
    # filter by logfc_threshold and pval_adj_threshold 
    if FC_cutoff is not None:
        dge_result = dge_result.query(f'avg_log2FC > {FC_cutoff}')
    if P_cutoff is not None:
        dge_result = dge_result.query(f'avg_log2FC < {P_cutoff}')
    
    # caculate min and max logfc 
    # used to set the colorbar height range
    volcano_height_df = (dge_result
                         .groupby(['celltype']).avg_log2FC
                         .agg(['min', 'max'])
                         .reset_index())
    
    volcano_height_df['x'] = (volcano_height_df['celltype']
                              .map(position_dict)
                              .astype(float))
    
    # gene label(two color) absolute log2FC to get the top n gene
    # x as the gene position
    jitter = np.random.uniform(-cbar_width, cbar_width, size=len(dge_result))
    
    dge_result = dge_result.assign(
        label = np.where(dge_result['p_val_adj'] < 0.01, 
                         'adjust P-val<0.01', 'adjust P-val>0.01'),
        fabs_log2_fc = np.fabs(dge_result['avg_log2FC']),
        x = dge_result['celltype'].map(position_dict).astype(float) + jitter)
    
    # get top n result
    top_gene = (
        dge_result
        .groupby(['celltype'])
        .apply(lambda x: x['gene'].isin(
            x.nlargest(top_n, columns='fabs_log2_fc')['gene'].values))
        .reset_index())
    
    # significant df of gene to add text
    dge_result.loc[top_gene.level_1, 'sig'] = top_gene.gene.values
    df_unsig = dge_result.query("sig == False")
    df_sig   = dge_result.query("sig == True")
    
    # palette
    cluster_color = list(get_palette(palette, 
                                     n_colors=len(order), 
                                     n_labels=len(order),
                                     as_hex=True))
    color_dict = dict(zip(order, cluster_color))
    volcano_height_df['color'] = volcano_height_df.celltype.map(color_dict)
    
    themes = (
        theme_classic() + 
        theme(axis_title=element_text(size=text_size, color='black'), 
            axis_line_y=element_line(color='black', size=1.2), 
            axis_line_x=element_blank(), 
            axis_text_x=element_blank(), 
            panel_grid=element_blank(), 
            legend_position=(0.25, 0.95), 
            legend_direction="vertical", 
            legend_text=element_text(size=8),
            axis_ticks_major_x=element_blank(),
            legend_title=element_blank()
            ))
    
    arrowprops = set_value(
        arrowprops, dict(arrowstyle='-', color='black', alpha=.5))
    
    p  = (
        ggplot() + 
        geom_col(
            data=volcano_height_df, 
            mapping=aes(x='x', y='min'), 
            fill="#dcdcdc",alpha=0.6,
            width=cbar_width * 2) + 
        geom_col(data=volcano_height_df, 
                 mapping=aes(x='x', y='max'),
                 fill="#dcdcdc",alpha=0.6,
                 width=cbar_width * 2) +
        geom_point(df_unsig, 
                    aes(x='x', y='avg_log2FC', color='label'), 
                    size=0.15, raster=True) + 
        geom_point(df_sig, 
                    aes(x='x', y='avg_log2FC', color='label'), 
                    size=0.5, raster=True) +
        geom_tile(volcano_height_df, 
                    aes(x='x', y=0, height=0.5, width=1), 
                    fill=volcano_height_df.color.values,
                    alpha=0.8, show_legend=False) + 
        
        scale_color_manual(name=None, values=("#FF5A5F", "gray")) + 
        labs(x='celltype', y="avg_log2FC") + 
        
        geom_text(volcano_height_df, 
                    aes(x='x', y=0, label='celltype'), 
                    size=6, color ="black",
                    ) + 
        
        geom_hline(yintercept= 0.5, linetype='dashed', color='black') +
        geom_hline(yintercept=-0.5, linetype='dashed', color='black') +
        themes
        )
    
    if repel:
        logger.info(
            'adjust text, this may takes few minutes...')
        
        fig = p.draw()
        ax = fig.get_axes()[0]
        
        # ax.plot([], dt_sig.avg_log2FC, 'o', color='red')

        txts = []
        for celltype in df_sig.celltype.unique():
            
            for _, row in df_sig[df_sig.celltype==celltype].iterrows():
                txt = plt.text(row.x, row.avg_log2FC, row.gene, 
                               fontsize=fontsize, style="italic")
                txts.append(txt)
                
        adjust_text(
            txts, precision=0.001,
            expand_text=(1.05, 1.), expand_points=(1.05, 1.01),
            force_text=(0.1, 0.01), force_points=(0.00125, 0.001),
            arrowprops=arrowprops
            )
        
        if save_path is not None:
            save_fig(save_path, dpi=300)
        
        return ax
    
    else:
        if save_path is not None:
            ggsave(p, save_path, dpi=300)
        return p
    

# TODO density & colormap
# TODO highlight genes list

# DGE_ARGS = dict()

class DegTable:
    def __init__(self, df, fc_col, p_col, 
                 groupby=None, gene_col='gene'):
        
        self.groupby = groupby if groupby in df.columns else None
        
        self.fc = df[fc_col]
        self.log_p_val = np.clip(-np.log10(df[p_col]), None, 300)
        self.genes = df[gene_col]
        
        if self.genes.duplicated().any():
            if groupby is None:
                raise ValueError('duplicated gene names found')
    
    def get_label(self, fc_cutoff, p_cutoff):
        mask = ((np.fabs(self.fc) >= fc_cutoff) & 
                (self.log_p_val >= -np.log10(p_cutoff)))
        
        label = np.where(mask, np.where(self.fc > 0, 'Up', 'Down'), 'Not sig')
        return pd.Categorical(label, categories=['Up', 'Down', 'Not sig'])
    
    def subset_by_gene(self, genes):
        mask = self.genes.isin(genes)
        fc = self.fc[mask]
        p_val = self.log_p_val[mask]
        gene = self.genes[mask]
        
        return fc, p_val, gene
    
    def top_n(self, n):
        if n > 0:
            idx = np.argsort(self.log_p_val)[-n:]
            fc = self.fc[idx]
            p_val = self.log_p_val[idx]
            gene = self.genes[idx]
            return fc, p_val, gene
        else:
            return np.array([]), np.array([]), np.array([])
    

def volcano_plot(
    deg_result: Union[pd.DataFrame, str],
    x: str = 'avg_log2FC',
    y: str = 'p_val_adj',
    gene_col: str = 'gene',
    fc_cutoff: float = 0.25,
    p_cutoff: float = 1e-4,
    alpha: float = 0.95,
    dot_size: float = 5,
    fontsize: int = 10,
    repel: bool = True,
    palette: np.ndarray = None,
    top_n: float = 30,
    highlight_genes: list = None,
    figsize: tuple = (4, 3),
    **kwargs
    ) -> Axes:
    """
    volcano plot

    Parameters
    ----------
    dge_result : Union[pd.DataFrame, str]
        result of differential gene expression analysis
    x : str, optional
        x, by default 'avg_log2FC'
    y : str, optional
        y, by default 'p_val_adj'
    FC_cutoff : float, optional
        log2 fold change threshold, by default 0.25
    P_cutoff : float, optional
        adjust p value threshold, by default 1e-4
    alpha : float, optional
        alpha of point, by default 0.95
    dot_size : float, optional
        point size, by default 1.5
    fontsize : int, optional
        font size, by default 10
    adjust_texts : bool, optional
        whether adjust text, by default True
    palette : np.ndarray, optional
        palette, by default None

    Returns
    -------
    Axes
    
    Examples
    ----------
    >>> volcano_plot(dge_result, x='avg_log2FC', y='p_val_adj', top_n=50, gene_col='gene')
    """
    
    deg_result = DegTable(deg_result, x, y, gene_col=gene_col)
    label = deg_result.get_label(fc_cutoff, p_cutoff)

    palette = set_value(palette, ['#FAD77BFF', 'gray', '#85D4E3FF'])
    ax = scatter(np.array([deg_result.fc, deg_result.log_p_val]).T, 
                 colors=label, palette=palette,  s=dot_size,
                 aspect='auto', figsize=figsize, **kwargs)
    
    ax.axvline(-fc_cutoff, color='k', ls='--')
    ax.axvline( fc_cutoff, color='k', ls='--')
    ax.axhline(-np.log10(p_cutoff), color='k', ls='--')

    txts = []
    xs, ys, genes = deg_result.top_n(top_n)
    if highlight_genes is not None:
        xs_h, ys_h, genes_h = deg_result.subset_by_gene(highlight_genes)
        mask = np.isin(genes, genes_h)
        xs, ys, genes = xs[~mask], ys[~mask], genes[~mask]
        
        # plot highlight genes
        for x, y, t in zip(xs_h, ys_h, genes_h):
            text = ax.text(x, y, t, fontsize=fontsize, fontstyle='italic', color='red')
            txts.append(text)
    
    # plot top n genes
    for x, y, t in zip(xs, ys, genes):
        text = ax.text(x, y, t, 
                       fontstyle='italic',
                       fontsize=fontsize)
        txts.append(text)

    if repel:
        logger.info(
            'adjust text, this may takes few minutes...')
        adjust_text(txts, arrowprops=dict(arrowstyle="-", color='grey', lw=0.2))
    
    return ax


def para_volcano(adata, groupby, group1, group2, gene_col='names',
                 deg_result=None, fc_col='logfoldchanges', p_col='pvals_adj',
                 fc_cutoff=0.5, p_cutoff=1e-3, top_n=10, highlight_genes=None,
                 features=None, layer=None,  dot_size=3, highlight_color='red',
                 palette=None, repel=True, figsize=(4,4), 
                 **kwargs):
    
    from dev_feature import pseudobulk_expression
    avg_exp = pseudobulk_expression(adata, groupby, layer=layer, features=features).loc[[group1, group2], :].T
    
    if deg_result is None:
        from scanpy.tools import rank_genes_groups
        from scanpy.get import rank_genes_groups_df
        deg_result = rank_genes_groups(adata, groupby, method='wilcoxon', layer=layer,
                                       reference=group1, groups=[group2], use_raw=False)
        deg_result = rank_genes_groups_df(adata, group2)
    
    deg_result = DegTable(deg_result, fc_col=fc_col, p_col=p_col, gene_col=gene_col)
    labels = deg_result.get_label(fc_cutoff, p_cutoff)
    gene2label = dict(zip(deg_result.genes, labels))
    labels = avg_exp.index.map(gene2label)
    
    palette = set_value(palette, ['#FAD77BFF', 'gray', '#85D4E3FF'])
    ax = scatter(avg_exp.values, colors=labels, #aspect='auto', 
                 figsize=figsize, dot_size=dot_size, palette=palette,
                 **kwargs)
    
    txts_up = []
    txts_down = []
    
    _, _, genes = deg_result.top_n(top_n)
    xs, ys = avg_exp.loc[genes, group1], avg_exp.loc[genes, group2]

    if highlight_genes is not None:
        _, _, genes_h = deg_result.subset_by_gene(highlight_genes)
        mask = np.isin(genes, genes_h)
        xs, ys, genes = xs[~mask], ys[~mask], genes[~mask]
        xs_h, ys_h = avg_exp.loc[genes_h, group1], avg_exp.loc[genes_h, group2]
        for x, y, t in zip(xs_h, ys_h, genes_h):
            text = ax.text(x, y, t, fontstyle='italic', color=highlight_color)
            if gene2label[t] == 'Up':
                txts_up.append(text)
            elif gene2label[t] == 'Down':
                txts_down.append(text)
            else:
                pass
    
    for x, y, t in zip(xs, ys, genes):
        text = ax.text(x, y, t, fontstyle='italic')
        
        if gene2label[t] == 'Up':
            txts_up.append(text)
        elif gene2label[t] == 'Down':
            txts_down.append(text)
        else:
            pass
        
    ax.set_xlabel(f'log({group1} gene expression)')
    ax.set_ylabel(f'log({group2} gene expression)')
    ax.set_title(f'{group1} vs {group2}')
    
    if repel:
        logger.info(
            'adjust text, this may takes few minutes...')
        arrowprops = dict(arrowstyle="-", color='grey', lw=0.2)
        adjust_text(txts_up, only_move={'text': 'y+'},
                    arrowprops=arrowprops)
        adjust_text(txts_down, only_move={'text': 'x+'},
                    arrowprops=arrowprops)
        
        
    return ax
    
    
# ==================================================================================
#                                   abundance analysis
# ==================================================================================

def fisher_exact_test(adata, groupby, splitby):
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import chi2_contingency 
    
    row_vec = adata.obs[splitby]
    col_vec = adata.obs[groupby]
    
    row_order = categorical_order(row_vec)
    col_order = categorical_order(col_vec)
    p_values = np.zeros((len(row_order), len(col_order)))
    R_oe = np.zeros((len(row_order), len(col_order)))
    a = pd.crosstab(row_vec, col_vec)
    b = a.values.sum(axis=0, keepdims=True) - a
    c = a.values.sum(axis=1, keepdims=True) - a
    d = adata.shape[0] - a - b - c
    
    for i, row in enumerate(row_order):
        for j, col in enumerate(col_order):
            obs = np.array([[a.loc[row, col], b.loc[row, col]],
                            [c.loc[row, col], d.loc[row, col]]])
            
            res = chi2_contingency(obs)
            R_oe[i, j] = (obs / res.expected_freq)[0, 0]
            p_values[i, j] = res[1]
    
    R_oe = pd.DataFrame(R_oe, index=row_order, columns=col_order).T
    p_values = pd.DataFrame(p_values, index=row_order, columns=col_order).T
    
    # Adjust p-values using the Benjamini-Hochberg method
    adjusted_p_values = multipletests(p_values.values.flatten(), method='fdr_bh')[1]

    # Reshape the adjusted p-values back to the original shape
    adjusted_p_values = adjusted_p_values.reshape(p_values.shape)

    # Update the p-values dataframe with the adjusted values
    p_values_adjusted = p_values.copy()
    p_values_adjusted.values[:] = adjusted_p_values
    
    return R_oe, p_values_adjusted


# TODO without summaray
def calc_prop(
    df, 
    cluster_col, 
    group_col=None, 
    split_col=None,
    percent=True, 
    summary: Union[str, List[str], Callable] = 'mean'):
    
    groups = []
    
    if split_col == group_col: 
        split_col = None

    for group in [group_col, split_col]:
        if group is not None:
            groups.append(group)
            
    grouper = df.groupby(groups) if len(groups) > 0 else df
    prop = (grouper
            .value_counts(normalize=percent)
            .reset_index(name='proportion'))
    
    if group_col is not None:
        prop = (prop
                .groupby([group_col, cluster_col])
                .proportion
                .agg(summary))
    
    return prop
          

def calc_cell_proportion(
    adata: AnnData, 
    groupby: str, # cell type
    splitby: str = None, # group
    replicates: str = None, # replicate sample
    return_proportion: bool = True,
    summary: Union[str, List[str], Callable] = 'mean',
    ):
    
    groups = [groupby, splitby, replicates]
    groups = [group for group in groups if group is not None]
    meta = adata.obs[groups]
    prop = calc_prop(
        meta, *groups, summary=summary, 
        percent=return_proportion)
    return prop


def diffrential_proportion(
    adata: AnnData, 
    cluster_col: str, # cell type
    group_col: str, # group
    group_pairs: List[Tuple[str, str]] = None,
    sample_col: str = None, # replicate sample
    percent: bool = True,
    order: List[str] = None,
    method: Literal['permutation', 'fisher'] = 'fisher', # TODO
    summary: Union[str, List[str], Callable] = 'mean',
    n: int = 500,
    ):
    
    from rich.progress import track
    from itertools import combinations
    
    # get proportion of cell type in each group
    groups = [cluster_col, group_col, sample_col]
    groups = [group for group in groups if group is not None]
    meta = fetch_data(adata, groups)
    
    if group_pairs is None:
        group_pairs = combinations(
            categorical_order(meta[group_col]), 2)
    
    def calc_and_pivot(*groups):
        prop = calc_prop(
            meta, *groups, 
            summary=summary, 
            percent=percent
        )
        
        res = pd.pivot_table(
            prop, 
            index=cluster_col, 
            columns=group_col, 
            values='proportion'
        ).reset_index()
        
        return res
    
    pre_value = calc_and_pivot(*groups)
    
    # do permutation test
    res_list = []
    # create a 1D array
    arr1d = meta[group_col].values.reshape(-1, 1)
    groups = [cluster_col, group_col]

    for _ in track(range(n), description='Do permutation...'):
        arr = np.random.permutation(arr1d)
        # meta[group_col] = arr2d[:, i]
        meta[group_col] = arr
        res_list.append(calc_and_pivot(*groups))
        
    results = pd.concat(res_list, axis=0)

    order = categorical_order(meta[cluster_col], order)
    for pair in group_pairs:
        for cluster in order:
            
            mask_perm = (results[cluster_col] == cluster)
            mask_real = (pre_value[cluster_col] == cluster)
            perm_pair = (
                results[mask_perm][pair[0]] - results[mask_perm][pair[1]]
            )
            real_pair = (
                pre_value[mask_real][pair[0]] - pre_value[mask_real][pair[1]]
            ).values
            
            # normality = shapiro(perm_pair).pvalue
            print(real_pair)
            
            ax = sns.histplot(perm_pair, color='grey')
            ax.axvline(np.fabs(real_pair), color='red')
            ax.set_title(f'{cluster} {pair[0]} - {pair[1]}')
        
            p_value = (perm_pair.values > real_pair).sum() / n
            p_value = min(p_value, 1 - p_value)
            
            print(pair, p_value)
    

# TODO
# for time series data, line chart and area chart may be better

def cell_component(
    adata: AnnData,
    groupby: str, 
    splitby: str = None,
    position: str = 'fill', # dodge
    flow: bool = False, # TODO add flow
    orientation: str = 'vertical',
    width: float = 0.65,
    wspace: float = 0.1,
    percent: bool = True,
    split_order: Union[List, str] = None,
    group_order: Union[List, str] = None,
    compare_pair: Union[bool, Sequence] = False,
    kind: Literal['bar', 'ribbon'] = 'bar',
    palette: Union[str, Sequence] = None, 
    text_ratio: float = 0.08,
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: Union[int, str] = 11,
    rotation: Optional[int] = None,
    ):
    # in future deprecate plotnine

    groups = [groupby]
    
    if splitby is not None:
        # groups - > [grou_by, splitby]
        groups.append(splitby)
        
    data = calc_cell_proportion(
        adata, groupby, splitby, 
        return_proportion=percent).reset_index()

    split_order = categorical_order(data[splitby], split_order)
    group_order = categorical_order(data[groupby], group_order)
    
    for col, order in zip(groups, [group_order, split_order]):
        # when splitby is None, split_order is ignored
        data = data[data[col].isin(order)]
        data[col] = pd.Categorical(data[col], categories=order)
    
    x, y = 'proportion', splitby
    
    # if compare:
    #     cell_pos = []
    #     for group in np.unique(count_cell[groupby]):
    #         # caculate the cusum of the proportion column and keep the celltype order
    #         per_group = count_cell.query(f"{groupby} == @group")

    #         order = categorical_order(per_group[splitby])
    #         per_group.set_index(splitby, inplace=True)

    #         per_group = per_group.loc[order[::-1], :]
    #         per_group['up'] = per_group.proportion.cumsum()
    #         per_group['low'] = per_group['up'] - per_group['proportion']
    #         cell_pos.append(per_group)
        
    #     cell_pos = pd.concat(cell_pos)
        
    #     group_order = categorical_order(adata.obs[splitby])
    #     for group in np.unique(count_cell[splitby]):
    #         per_cell = cell_pos.query(f"{splitby} == @group")
            
    #         for idx, 
    
    # ---------------------- compare pair ----------------------
    if compare_pair:
        
        if len(split_order) < 2:
            raise ValueError(
                f"{split_order} must have at least 2 groups")
        
        elif len(split_order) == 2:
            compare_pair = split_order
        
        else:
            if isinstance(compare_pair, bool):
                raise ValueError(
                    "compare_pair must be a pair of group")
            
            elif isinstance(compare_pair, Sequence):
                if len(compare_pair) != 2:
                    raise ValueError(
                        "compare_pair must be a pair of group")
        
        mosaic = mosaic_layout(
            0, 0, 
            parent=[['bar_left', 'text', 'bar_right']]
            )
        
        bar_ratio = (1 - text_ratio) / 2
        
        fig = plt.figure(
            figsize=set_value(
                figsize, (4, len(split_order) / 3)
                ))
        
        axd = fig.subplot_mosaic(
            mosaic, 
            width_ratios = [bar_ratio, text_ratio, bar_ratio], 
            # height_ratios=height_ratios + [height],
            gridspec_kw={'wspace': wspace, 'hspace': 0.},
            sharey=True)
        
        # plot the bar
        ax_bars = [axd['bar_left'], axd['bar_right']]
        palette = set_value(palette, ['#7FDCC5', '#7FC5DC'])
        x, y = 'proportion', groupby
        
        for group, ax, c in zip(compare_pair, ax_bars, palette):
            sns.barplot(
                data=data[data[splitby] == group], 
                x=x, y=y, ax=ax, 
                width=width, palette=[c])
            ax.set_title(group)

        # plot the text
        axd['text'].set_xlim(0, 1)
        for text in axd['bar_left'].get_yticklabels():
            pos = text.get_position()[1]
            label = text.get_text()
            axd['text'].text(0.5, pos, label, 
                            # transform=text_tranform,
                            fontsize=fontsize,
                            ha='center', va='center')
        
        # set the axis
        for ax in [axd['bar_left'], axd['bar_right']]:
            from matplotlib.ticker import FormatStrFormatter
            ax.set(xlabel='', ylabel='', yticks=[])
            ax.spines[:].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_major_formatter(
                FormatStrFormatter('%.1f'))
            
        set_cax_limit(axd['bar_right'], axd['bar_left'], sharex=True)
        axd['bar_left'].invert_xaxis()
        axd['text'].axis('off')
    
    else:
        # to draw the ribbon plot
        # need x, y_min, y_max
        
        palette = assign_palette(adata, groupby, palette=palette)
        p = (
            ggplot(data, aes(x=splitby, y='proportion', fill=groupby)) +
            scale_fill_manual(values=palette) + 
            theme_matplotlib() + 
            theme(
                axis_text_x=element_text(
                    rotation=rotation,
                    **_set_labelsalign(rotation))
                ) +
            theme(figure_size=figsize)
            )
        
        if kind == 'bar':
            p += geom_col(stat='identity', position=position, show_legend=True, width=width)
        
        elif kind == 'ribbon':
            rank_dict = dict(zip(order, range(len(order))))
            data['x'] = data[splitby].map(rank_dict).astype(int)
            p = (p + 
                 geom_area(aes(x='x')) + 
                 
                 scale_y_continuous(expand=(0, 0)) +
                 scale_x_continuous(breaks=list(rank_dict.values()), 
                                    labels=list(rank_dict.keys()),
                                    expand=(0, 0)
                                    ))
        else:
            pass

        return p
    

# ==================================================================================
#                           spatial cell interaction
# ==================================================================================

# TODO: other methods that caculate the score
def calculate_score(
    adata, 
    lr_pairs, 
    neighbors_key, 
    min_exp=0, 
    verbose=True, 
    key_add='cci_score',
    spot_comp=None, # TODO
    min_spots=None,
    nb_workers=4
    ):
    if isinstance(lr_pairs, str):
        lr_pairs = [lr_pairs]
        
    if verbose:
        print(
            'filtered out the lr pairs which is not'
            'unique or has gene not in the `adata.var`')   
        
    lr_pairs = np.unique(lr_pairs)
    lr_pairs = [item for item in lr_pairs 
                if np.all([is_gene_name(adata, gene) 
                           for gene in item.split("_")])]
    lr_genes = '_'.join(lr_pairs).split('_')
    genes = np.unique('_'.join(lr_pairs).split('_'))
    lr_pairs_rev = [item.split("_")[1] + "_" + item.split("_")[0]
                    for item in lr_pairs]
    lr_genes_rev = '_'.join(lr_pairs_rev).split('_')

    df = adata[:, genes].copy().to_df()
    
    if verbose:
        print('calculating cci score...') 
        
    spot_lr1 = df[lr_genes]
    spot_lr2 = df[lr_genes_rev]
    

    def mean_lr2(x):
        # get lr2 expressions from the neighbour(s)
        nbs = spot_lr2.loc[adata.obs.loc[x.name, neighbors_key], :]

        if nbs.shape[0] > 0:  # if neighbour exists
            return nbs.sum() / nbs.shape[0]
        else:
            return 0

    # mean of lr2 expressions from neighbours of each spot
    try:
        from pandarallel import pandarallel
    except ImportError:
        raise ImportError(
            'Please install the pandarallel, `pip3 install pandarallel')
        
    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers, verbose=1)  
    nb_lr2 = spot_lr2.parallel_apply(mean_lr2, axis=1)
    
    # keep value of nb_lr2 only when lr1 is also expressed on the spots
    spot_lr = (spot_lr1.values * (nb_lr2.values > min_exp) + 
               (spot_lr1.values > min_exp) * nb_lr2.values)
    
    columns = [lr_pairs[i // 2]  
               for i in range(len(spot_lr2.columns))]
    columns = pd.MultiIndex.from_arrays((columns, 
                                        spot_lr2.columns))
    scores = pd.DataFrame(spot_lr, 
                          index=adata.obs_names, 
                          columns=columns)
    scores = scores.T.reset_index().groupby(['level_0']).sum().T
    scores.index.name = 'LR pairs'
    
    if min_spots is not None and isinstance(min_spots, int):
        lrs_bool = (scores > 0).sum(axis=0) > min_spots
        exp_pairs =  lrs_bool.sum()
        if exp_pairs > 0: 
            scores = scores.loc[:, lrs_bool]
            if verbose:
                print(f'filtered out {len(lr_pairs) - exp_pairs} lr pairs')
        else:
            raise ValueError(f'lrs expressed on less than {min_spots}, '
                              'try to decrease min_exp')
        
    if key_add is not None:   
        adata.obsm[key_add] = scores 
        return adata
    else:
        return scores
    
    
def get_lr_features(adata, lr_expr, lr_pairs, quantiles): # modified
    
    quantiles = np.array(quantiles)
    # Determining indices of LR pairs #
    l_indices, r_indices = [], []
    for lr in lr_pairs:
        l_, r_ = lr.split("_")
        l_indices.extend(np.where(lr_expr.columns.values == l_)[0])
        r_indices.extend(np.where(lr_expr.columns.values == r_)[0])  

    interpolation="nearest"
    # Calculating the non-zero quantiles of gene expression
    def nonzero_quantile(expr):
        """Calculating the non-zero quantiles."""
        nonzero_expr = expr[expr > 0] # all=0
        quants = np.quantile(nonzero_expr, 
                             q=quantiles, 
                             interpolation=interpolation)
        return quants

    def nonzero_median(expr):
        """Calculating the non-zero median."""
        nonzero_expr = expr[expr > 0]
        median = np.median(nonzero_expr)
        return median

    def zero_props(expr):
        """Calculating the non-zero pro."""
        gene_props = (expr == 0).sum() / len(expr)
        return gene_props
    
    summary = lr_expr.T.agg([nonzero_quantile, nonzero_median, 
                             zero_props, np.median], axis=1)
    l_summary = summary.iloc[l_indices, :]
    r_summary = summary.iloc[r_indices, :]
    
    lr_median_means = (l_summary.nonzero_median.values +  
                       r_summary.nonzero_median.values) / 2
    lr_prop_means   = (l_summary.zero_props.values +  
                       r_summary.zero_props.values) / 2
    
    median_order    = np.argsort(lr_median_means)
    prop_order      = np.argsort(lr_prop_means * -1)
    
    median_ranks = [np.where(median_order == i)[0][0] 
                    for i in range(len(lr_pairs))]
    prop_ranks   = [np.where(prop_order == i)[0][0] 
                    for i in range(len(lr_pairs))]
    mean_ranks = np.array([median_ranks, prop_ranks]).mean(axis=0)
    
    columns=["nonzero-median", "zero-prop", 
             "median_rank", "prop_rank", "mean_rank"]
    lr_features = pd.DataFrame(columns=columns, index=lr_pairs)
    lr_features[columns[0: 2]] = summary[['nonzero_median', 
                                          'zero_props']].values.mean(axis=0)
    lr_features[columns[2]] = median_ranks
    lr_features[columns[3]] = prop_ranks
    lr_features[columns[4]] = mean_ranks # pd.assign
    
    q_cols = [f'L_q{quantile}' for quantile in quantiles] + \
             [f'R_q{quantile}' for quantile in quantiles]

    q_values = np.hstack([l_summary.nonzero_quantile.values.tolist(),
                          r_summary.nonzero_quantile.values.tolist()])

    lr_features[q_cols] = q_values
    adata.uns['lr_features'] = lr_features
    return lr_features  


def get_similar_genes(ref_quants: np.ndarray,
                      n_genes: int,
                      candidate_quants: np.ndarray,
                      candidate_genes: np.ndarray,
                      ):
    ref_quants = ref_quants.reshape(-1, 1)  
    dists = (np.fabs(ref_quants - candidate_quants) /
             (ref_quants + candidate_quants)).sum(axis=0)
    
    # remove the zero-dists since this 
    # indicates they are the same gene
    dists = dists[dists > 0]
    candidate_quants = candidate_quants[:, dists > 0]
    candidate_genes  = candidate_genes[dists > 0]
    order = np.argsort(dists)
    
    # Retrieving desired number of genes 
    similar_genes = candidate_genes[order[0: n_genes]]
    return similar_genes

def gen_rand_pairs(genes1: np.ndarray, genes2: np.ndarray, n_pairs: int):
    """Generates random pairs of genes."""

    rand_pairs = list()
    for _ in range(0, n_pairs):
        l_rand = np.random.choice(genes1, 1)[0]
        r_rand = np.random.choice(genes2, 1)[0]
        rand_pair = "_".join([l_rand, r_rand])
        while rand_pair in rand_pairs or l_rand == r_rand:
            l_rand = np.random.choice(genes1, 1)[0]
            r_rand = np.random.choice(genes2, 1)[0]
            rand_pair = "_".join([l_rand, r_rand])

        rand_pairs.append(rand_pair)

    return rand_pairs


def get_lr_bg(
    adata,
    neighbors_key,
    min_exp,
    lr_,
    l_quant,
    r_quant,
    genes,
    candidate_quants,
    gene_bg_genes,
    n_genes,
    n_pairs,
    spot_comp=None, # TODO
):
    """Gets the LR-specific background & bg spot indices."""
    
    l_, r_ = lr_.split("_")
    if l_ not in gene_bg_genes:
        l_genes = get_similar_genes(
            l_quant, n_genes, candidate_quants, genes  # group_l_props,
        )
        gene_bg_genes[l_] = l_genes
    else:
        l_genes = gene_bg_genes[l_]
        
    if r_ not in gene_bg_genes:
        r_genes = get_similar_genes(
            r_quant, n_genes, candidate_quants, genes  # group_r_props,
        )
        gene_bg_genes[r_] = r_genes
    else:
        r_genes = gene_bg_genes[r_]
        
    rand_pairs = gen_rand_pairs(l_genes, r_genes, n_pairs)
    background = calculate_score(
        adata=adata, lr_pairs=rand_pairs,
        neighbors_key=neighbors_key, min_exp=min_exp,
        verbose=False,key_add=None)
    
    return background
    

def _permutation(
    adata: AnnData,
    lr_scores: np.ndarray,
    lr_pairs: np.array,
    n_pairs: int,
    neighbors_key: List,
    min_exp: float,
    het_vals: np.array = None, # TODO
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.05,
    verbose: bool = True,
    save_bg=False,
    neg_binom=False,
    quantiles=(0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 1)
    ):
    """
    Calls significant spots by creating random gene pairs with similar
    expression to given LR pair; only generate background for spots
    which have score for given LR.
    """
    quantiles = np.array(quantiles)

    lr_genes = np.unique([lr_.split("_") for lr_ in lr_pairs])
    genes = np.array([gene for gene in adata.var_names if gene not in lr_genes])
    candidate_expr = adata[:, genes].to_df().values

    n_genes = round(np.sqrt(n_pairs) * 2) 
    if len(genes) < n_genes:
        print(
            "Exiting since need at least "
            f"{n_genes} genes to generate {n_pairs} pairs."
        )
        return

    if n_pairs < 100:
        print(
            "Exiting since `n_pairs < 100`, need much larger number of pairs to "
            "get accurate backgrounds (e.g. 1000)."
        )
        return
    lr_expr = adata[:, lr_genes].copy().to_df()
    lr_feats = get_lr_features(
        adata, lr_expr, lr_pairs, quantiles)
    l_quants = lr_feats.loc[lr_pairs, 
                            [col for col in lr_feats.columns if "L_" in col]
                            ].values
    r_quants = lr_feats.loc[lr_pairs, 
                            [col for col in lr_feats.columns if "R_" in col]
                            ].values
    
    candidate_quants = np.apply_along_axis(np.quantile, 0, candidate_expr, 
                                           q=quantiles, interpolation="nearest"
                                           )
    
    pvals = np.ones(lr_scores.shape, dtype=np.float32)
    # do permutation
    from tqdm import tqdm 
    gene_bg_genes = dict()
    pbar = tqdm(lr_pairs)
    if verbose: 
        print("Performing permutation...")
        
    for idx, lr_ in enumerate(pbar):
        pbar.set_description(f"LR pairs {idx + 1}")
        pbar.set_postfix(LR=lr_pairs[idx])
        background = get_lr_bg(
            adata=adata,
            neighbors_key=neighbors_key,
            min_exp=min_exp,
            lr_=lr_,
            l_quant=l_quants[idx, :],
            r_quant=r_quants[idx, :],
            genes=genes,
            candidate_quants=candidate_quants,
            gene_bg_genes=gene_bg_genes,
            n_genes=n_genes,
            n_pairs=n_pairs,
        )
        lr_score = lr_scores[lr_].values
        spot_indices = np.where(lr_score > 0)[0]
        
        if save_bg:
            adata.uns["lrs_to_bg"][lr_] = background

        if not neg_binom:  # Calculate empirical p-values per-spot
            n_greater = (background.values[spot_indices, :] 
                        >= lr_score[spot_indices].reshape(-1, 1)).sum(axis=1)
            
            n_greater = np.where(n_greater != 0, n_greater, 1)
            pvals[spot_indices, idx] = n_greater / background.shape[0]
            
        else: # Negative bionomial method
            pass
    
    if verbose:
        print('adjust p value...')        
    # adjust p value
    from statsmodels.stats.multitest import multipletests
    
    def MHT(ar, adj_method):
        return multipletests(ar, method=adj_method)[1]
    
    pvals_adj = np.apply_along_axis(MHT, 1, pvals, adj_method=adj_method)
    log10pvals_adj = -np.log1p(pvals_adj)
    adata.uns['LR_Pvals'] = pvals_adj
    print(pvals_adj)
    

def cci_score(
    adata: AnnData,
    lr_pairs: Union[list, np.ndarray],
    distance: Union[int, float] = 5,
    spot_comp: pd.DataFrame = None, 
    verbose: bool = True, 
    key_add: str = 'cci_score',
    min_exp: Union[int, float] = 0, 
    use_raw: bool = False,
    min_spots: int = 20, 
    n_pairs: int = 1000,
    adj_method: str = "fdr_bh",
    n_workers=4
    ):  # reference: stlearn
    """
    calculate cci score for each LR pair and do permutation test

    Parameters
    ----------
    adata : `AnnData`
        anndata object
    lr_pairs : Union[list, np.ndarray]
        LR pairs
    distance : Union[int, float], optional
        the distance between spots which are considered as neighbors , by default 5
    spot_comp : `pd.DataFrame`, optional
        spot component of different cells, by default None
    key_add : str, optional
        key added in `adata.obs`, by default 'cci_score'
    min_exp : Union[int, float], optional
        the min expression of ligand or receptor gene when caculate reaction strength, by default 0
    use_raw : bool, optional
        whether to use counts in `adata.raw.X`, by default False
    min_spots : int, optional
        the min number of spots that score > 0, by default 20
    n_pairs : int, optional
        number of pairs to random sample, by default 1000
    adj_method : str, optional
        adjust method of p value, by default "fdr_bh"
    n_wokers : int, optional
        num of worker when calculate_score, by default 4

    """
    # TODO: add the weight of cell(bin composition)
    # TODO: capitalize the genes    

    # get neighbors 
    import scipy.spatial as spatial
    neighbors = []
    point_tree = spatial.cKDTree(adata.obsm['spatial'])

    #TODO neighbors with radius or n_neighbors 
    for idx, point in enumerate(adata.obsm['spatial']):
        neighbor = point_tree.query_ball_point(point, distance)  # 这里5是指距离
        self_neighbor = [adata.obs_names[idx]]
        neighbor.remove(idx)
        other_neighbor = adata.obs_names[neighbor]
        
        if distance == 0:
            neighbors.append(self_neighbor)
        elif distance > 0:
            neighbors.append(other_neighbor)
        else:
            raise ValueError("`distance` should > 0")
    neighbors_key = f'neighbors_{distance}'   
    adata.obs[neighbors_key] = neighbors
    
    # calulate lr scores
    adata = calculate_score(
        adata=adata,
        lr_pairs=lr_pairs,
        neighbors_key=neighbors_key,
        min_exp=min_exp,
        verbose=verbose,
        nb_workers=n_workers,
        spot_comp=spot_comp,
        key_add=key_add, 
        min_spots=min_spots
        )
    
    lr_scores = adata.obsm[key_add]
    lr_pairs = lr_scores.columns.tolist()
    
    # permutation
    _permutation(
        adata=adata, 
        lr_scores=lr_scores,
        lr_pairs=lr_pairs,
        n_pairs=n_pairs,
        neighbors_key=neighbors_key,
        min_exp=min_exp,
        adj_method=adj_method
        )




# find neighbors
# ==================================================================================
#                           data smothing and imputation
# ==================================================================================

# magic
# spage
# tangram


# ------------------
# 1. data smothing
# ------------------

# ref: hotspot, https://hotspot.readthedocs.io/en/latest/Spatial_Tutorial.html
# ref: KNN kernel smoothing, https://towardsdatascience.com/make-your-knn-smooth-with-gaussian-kernel-7673fceb26b9
# ref: palantir 
# ref: magic


def compute_weights(distances: np.ndarray, neighborhood_factor: float = 3):
    """
    Computes weights on the nearest neighbors based on a
    gaussian kernel and their distances

    Kernel width is set to the num_neighbors / neighborhood_factor's distance
    
    Parameters
    ----------
    distances : np.ndarray
        cells x neighbors ndarray
    neighborhood_factor : float, optional
        determin sigma, by default 3

    Returns
    -------
    np.array
        cells x neighbors ndarray
        
    """
    from math import ceil
    
    # when kernel width / sigma is 3, 0.99 of the weight is within the kernel
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)

    sigma = distances[:, [radius_ii-1]]
    sigma[sigma == 0] = 1

    weights = np.exp(-1 * distances**2 / sigma**2)

    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm

    return weights


def neighbors_and_weights(data, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True):
    """
    Computes nearest neighbors and associated weights for data
    Uses euclidean distance between rows of `data`

    Parameters
    ----------
    data: pd.Dataframe 
        num_cells x num_features

    Returns
    -------
    neighbors: pd.Dataframe 
        num_cells x n_neighbors
    weights: pd.Dataframe 
        num_cells x n_neighbors

    """
    from pynndescent import NNDescent
    from sklearn.neighbors import NearestNeighbors
    
    
    coords = data.values

    if approx_neighbors:
        index = NNDescent(coords, n_neighbors=n_neighbors)
        ind, dist = index.neighbor_graph
    else:
        tree = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm="ball_tree")
        tree.fit(coords)
        dist, ind = tree.kneighbors()
        
    weights = compute_weights(
        dist, neighborhood_factor=neighborhood_factor)

    ind = pd.DataFrame(ind, index=data.index)
    neighbors = ind
    weights = pd.DataFrame(weights, 
                           index=neighbors.index,
                           columns=neighbors.columns)

    return neighbors, weights


def make_weights_non_redundant(neighbors, weights):
    w_no_redundant = weights.copy()
    
    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]

            if j < i:
                continue

            for k2 in range(neighbors.shape[1]):
                if neighbors[j, k2] == i:
                    w_ji = w_no_redundant[j, k2]
                    w_no_redundant[j, k2] = 0
                    w_no_redundant[i, k] += w_ji

    return w_no_redundant


def neighbor_smoothing_row(vals, neighbors, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


    vals: expression matrix (genes x cells)
    neighbors: neighbor indices (cells x K)
    weights: neighbor weights (cells x K)
    _lambda: ratio controlling self vs. neighborhood
    """

    out = np.zeros_like(vals, dtype=np.float64)
    out_denom = np.zeros_like(vals, dtype=np.float64)

    N = neighbors.shape[0]  # Cells
    K = neighbors.shape[1]  # Neighbors

    for i in range(N):

        xi = vals[i]

        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            xj = vals[j]

            # calculate weighted values 
            # at same for neighbors time
            out[i] += xj*wij
            out[j] += xi*wij

            out_denom[i] += wij
            out_denom[j] += wij

    out /= out_denom

    out = (out * _lambda) + (1 - _lambda) * vals

    return out

def data_smothing(coords: np.ndarray, 
                  value_df: pd.DataFrame, 
                  approx_neighbors: bool = True, 
                  n_neighbors: int = 50):
    
    coords = pd.DataFrame(coords, index=value_df.index, columns=['x', 'y'])
    
    # compute neighbors and weights
    neighbors, weights = neighbors_and_weights(
        coords, n_neighbors=n_neighbors, approx_neighbors=approx_neighbors
        )
    
    neighbors = neighbors.loc[value_df.index, :]
    weights = weights.loc[value_df.index, :]
    
    # remove redundant weights
    w_no_redundant = make_weights_non_redundant(neighbors.values, weights.values)
    
    # smoothing
    value_df = value_df.apply(
        lambda x: neighbor_smoothing_row(
            x.values, neighbors.values, w_no_redundant, _lambda=.9
            ), axis=0
        )
    
    return value_df
    
# just for visualization    
def neighbor_smooth(adata, 
                    features , 
                    reduction, 
                    n_neighbors=50, 
                    approx_neighbors=True,
                    verbose=True):
    value_df = fetch_data(adata, features)
    coords = get_cell_embeddings(adata, reduction=reduction)
    value_df = data_smothing(
        coords, value_df, approx_neighbors=approx_neighbors, n_neighbors=n_neighbors)
    
    if verbose:
        print('smoothed values are added to adata.obs')
    adata.obs['smoothed_' + value_df.columns] = value_df.values


# ==================================================================================
#                               spatial clustering
# ==================================================================================


    
# TODO: add moran's I

# TODO wrapper these function to a class
# ==================================================================================
#                                 deconvolution
# ==================================================================================


# -------------------
# cell2location utils
# -------------------

def sample_yield(adata, batch_size=30000):
    # shuffled_df = df.sample(frac=1)
    idx = adata.obs.index.values.copy() #!
    
    np.random.shuffle(idx)
    num_samples = len(idx)
    num_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)
        yield adata[idx[start_index:end_index]]


def regression_model(adata_ref,  
                     batch_key, 
                     labels_key, 
                     use_gpu=True,
                     sc_model=None,
                     save_path=None
                     ):
    
    from cell2location.models import RegressionModel
    
    if sc_model is None:
        # prepare anndata for the regression model
        RegressionModel.setup_anndata(
            adata=adata_ref,
            # 10X reaction / sample / batch
            batch_key=batch_key,
            # cell type, covariate used for constructing signatures
            labels_key=labels_key,
            )

        # create and train the regression model
        mod = RegressionModel(adata_ref)
        print('the sc setup data as below')
        mod.view_anndata_setup()

        print('Use all data for training ' 
              '(validation not implemented yet, `train_size=1`)')
        
        mod.train(max_epochs=200, 
                  batch_size=2500,
                  train_size=1, 
                  lr=0.005, 
                  use_gpu=use_gpu)

        # plot ELBO loss history during training
        # removing first 20 epochs from the plot
        mod.plot_history(20)
        
    else:
        mod = RegressionModel.load(f"{sc_model}", adata_ref)
        
    # In this section, we export the estimated 
    # cell abundance (summary of the posterior distribution).
    adata_ref = mod.export_posterior(
        adata_ref, 
        sample_kwargs={'num_samples': 1000, 
                       'batch_size': 2500, 
                       'use_gpu': use_gpu})
    
    # Save anndata object with results
    if save_path is not None:
        mod.save(f"{save_path}/reference_signatures", overwrite=True)
        adata_file = f"{save_path}/sc.h5ad"
        adata_ref.write(adata_file)
        print(f'{adata_file} saved!')
    
    # mod.plot_QC()
    return adata_ref

def c2l_model(ref,
              adata,
              N_cells_per_location=4,
              model_kwargs={}, 
              use_gpu=True,
              sp_model=None,
              verbose=True,
              save_path=None
              ):
    
    from cell2location.models import Cell2location
    
    if isinstance(ref, AnnData):
        try:
        # export estimated expression in each cluster
            if 'means_per_cluster_mu_fg' in ref.varm.keys():
                inf_aver = ref.varm['means_per_cluster_mu_fg'][
                    [f'means_per_cluster_mu_fg_{i}' for i in ref.uns['mod']['factor_names']]
                    ].copy()
            else:
                inf_aver = ref.var[
                    [f'means_per_cluster_mu_fg_{i}' for i in ref.uns['mod']['factor_names']]
                    ].copy()
            inf_aver.columns = ref.uns['mod']['factor_names']
        except KeyError:
            raise KeyError('Please train the regression model first')
        
    elif isinstance(ref, pd.DataFrame):
        inf_aver = ref.copy()
    else:
        return

    # find mitochondria-encoded (MT) genes
    adata.var['mt_gene'] = [gene.startswith('mt-') for gene in adata.var_names]
    # remove MT genes for spatial mapping (keeping their counts in the object)
    adata = adata[:, ~adata.var_names.str.startswith('mt-')]

    intersect = np.intersect1d(adata.var_names, inf_aver.index)
    adata = adata[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    Cell2location.setup_anndata(adata=adata)

    # create and train the model
    mod = Cell2location(
        adata, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=N_cells_per_location,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection (using default here):
        detection_alpha=20,
        **model_kwargs)
    
    mod.view_anndata_setup()
    mod.train(max_epochs=8000,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=use_gpu)

    adata = mod.export_posterior(
        adata, 
        sample_kwargs={'num_samples': 500, 
                       'batch_size': mod.adata.n_obs, 
                       'use_gpu': use_gpu})

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    # mod.plot_history(1000)
    # plt.legend(labels=['full data training']);

    if save_path is not None:
        adata_file = f"{save_path}/sp.h5ad"
        mod.save(f"{save_path}/cell2location_map", overwrite=True)
        adata.write(adata_file)
        print(f'{adata_file} saved!')
        
    return adata

def run_cell2location(
    adata: AnnData, 
    adata_ref: AnnData, 
    batch_key: str = 'batch',
    labels_key: str ='celltype',
    sample_key=None,
    filtered: bool = False,
    save_path: Optional[str]=None, #TODO 
    sc_model=None, 
    sp_model=None,
    N_cells_per_location: int = 4, 
    model_kwargs:dict = {},
    use_gpu: bool = False,
    verbose: bool = True,
    batch_size : int = 30000
    ):
    import torch
    
    if use_gpu:
        torch.cuda.empty_cache()
    
    # filter the object
    if not filtered:
        from cell2location.utils.filtering import filter_genes
        
        if verbose: 
            print('filter single cell data...')
            
        selected = filter_genes(
            adata_ref, 
            cell_count_cutoff=5,
            cell_percentage_cutoff2=0.03,
            nonz_mean_cutoff=1.12)
        
        adata_ref = adata_ref[:, selected].copy()
    
    if verbose: 
        print('Estimation of reference cell type signatures...')
    
    adata_ref = regression_model(
        adata_ref, 
        save_path=save_path,
        batch_key=batch_key,
        labels_key=labels_key,
        use_gpu=use_gpu,
        sc_model=sc_model
    ) 

    if verbose: 
        print('training cell2location model...')
    
    abundance_df = []
    for adata_i in sample_yield(adata, batch_size=batch_size):
        adata_i = c2l_model(
            adata_ref,
            adata_i,
            N_cells_per_location=N_cells_per_location, 
            model_kwargs=model_kwargs,
            sp_model=sp_model,
            use_gpu=use_gpu)
        
        df = adata_i.obsm['q05_cell_abundance_w_sf']
        abundance_df.append(df)
    
    res = pd.concat(abundance_df)
    res = res.loc[adata.obs.index, :]
    
    return res


# -------------------
#      tangram
# -------------------

def run_tangram(
    adata, 
    adata_ref, 
    labels_key='celltype',
    marker_list=None, 
    use_gpu=False,
    verbose=False,
    top_n=10,
    **kwargs):
    try:
        import tangram as tg
    except ImportError:
        raise ImportError('Please install tangram `pip install tangram-sc`')
    import scanpy as sc
    
    if marker_list is None:
        print('marker list is not provided, run rank_genes_groups to get the marker list')
        sc.tl.rank_genes_groups(adata_ref, labels_key, use_raw=False)
        markers = pd.DataFrame(
            adata_ref.uns["rank_genes_groups"]["names"]).iloc[0: top_n]
        
        genes_sc = np.unique(markers.melt().value.values)
        genes_st = adata.var_names.values
        marker_list = np.intersect1d(genes_sc, genes_st)   
    
    # map training gene
    tg.pp_adatas(adata_ref, adata, genes=marker_list)
    
    # fit the model
    # return the probability of each cell type in each spot
    ad_map = tg.map_cells_to_space(
        adata_sc=adata_ref,
        adata_sp=adata,
        device='cuda:0' if use_gpu else 'cpu',
        **kwargs
        )   # cell * spot matrix
    
    tg.project_cell_annotations(ad_map, adata, annotation=labels_key)
    
    return ad_map, adata


def run_scope(
    adata, 
    adata_ref, 
    labels_key='celltype',
    use_gpu=False,
    verbose=False,
    trained=False,
    **kwargs):
    from scvi.external import RNAStereoscope, SpatialStereoscope
    
    intersect = np.intersect1d(adata_ref.var_names, adata.var_names)
    adata_ref = adata_ref[:, intersect].copy()
    adata     = adata[:, intersect].copy()
    
    # Learn cell-type specific gene expression from scRNA-seq data
    RNAStereoscope.setup_anndata(adata_ref, labels_key=labels_key)
    train = not trained
    if train:
        sc_model = RNAStereoscope(adata_ref)
        sc_model.train(max_epochs=100)
        # sc_model.history["elbo_train"][10:].plot()
        sc_model.save("scmodel", overwrite=True)
    else:
        sc_model = RNAStereoscope.load("scmodel", adata=adata_ref)
        print("Loaded RNA model from file!")
        
    # Infer proportion for spatial data
    
    adata.layers["counts"] = adata.X.copy()
    SpatialStereoscope.setup_anndata(adata, layer="counts")
    
    train = not trained
    if train:
        spatial_model = SpatialStereoscope.from_rna_model(adata, sc_model)
        spatial_model.train(max_epochs=2000)
        # spatial_model.history["elbo_train"][10:].plot()
        spatial_model.save("stmodel", overwrite=True)
    else:
        spatial_model = SpatialStereoscope.load("stmodel", adata=adata)
        print("Loaded Spatial model from file!")
        
    adata.obsm["deconvolution"] = spatial_model.get_proportions()
    
    for ct in adata.obsm["deconvolution"].columns:
        adata.obs[ct] = adata.obsm["deconvolution"][ct]
    
    return adata
    

#TODO moran's I greay's C

#? this take from cell2location
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


def get_rgb_function(cmap, min_value, max_value):
    """
    Generate a function to map continous values to RGB values
     
    using colormap between min_value & max_value.
    
    """

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

    if min_value == max_value:
        logger.warning(
            "Max_color is equal to min_color. It might be because of the data or bad parameter choice. "
            "If you are using plot_contours function try increasing max_color_quantile parameter and"
            "removing cell types with all zero values."
        )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap((np.clip(x, min_value, max_value) - min_value) / (max_value - min_value))

    return func


def rgb_to_ryb(rgb):
    """
    Converts colours from RGB colorspace to RYB

    Parameters
    ----------

    rgb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    """
    Converts colours from RYB colorspace to RGB

    Parameters
    ----------

    ryb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]


def plot_spatial_general(
    coords: Union[np.ndarray, pd.Series],
    value_df: pd.DataFrame,
    labels: Optional[list] = None,
    text:Optional[pd.DataFrame] = None,
    dot_size: float = 4.0,
    alpha_scaling: float = 1.0,
    max_col: Sequence = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    kind: str = "spatial",
    max_color_quantile: float = 0.98,
    show_img: bool = True,
    img: Optional[np.ndarray] = None,
    img_alpha: float = 1.0,
    adjust_text: bool = False,
    plt_axis:str = "off",
    invert_y: bool = False,
    x_y_labels: Sequence = ("", ""),
    crop_x: Optional[Sequence]=None,
    crop_y: Optional[Sequence]=None,
    text_box_alpha:str = 0.9,
    reorder_cmap=range(7),
    style: str = "fast",
    colorbar_position:str = "bottom",
    colorbar_label_kw: dict = {},
    colorbar_shape: dict = {},
    colorbar_tick_size: int = 12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
    ax: Optional[Axes] = None,
    fig: Optional[plt.Figure] = None,
    figsize: Optional[Sequence] = None,
    raster: bool = False,
    ):
    """ Plot spatial abundance of cell types (regulatory programmes) with colour gradient and interpolation
       
    This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap)

    'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white' taken from cell2loaction
       
    Parameters
    ----------
    value_df : pd.DataFrame
         cell abundance or other features (only 7 allowed, columns) across locations (rows)
    coords : np.ndarray
        x and y coordinates (in columns) to be used for ploting spots
    labels : list
        labels of cell types
    text : pd.DataFrame , optional
        with x, y coordinates, text to be printed
    dot_size : float, optional
        diameter of circles, by default 4.0
    alpha_scaling : float, optional
        adjust color alpha, by default 1.0
    max_col : tuple, optional
        crops the colorscale maximum value for each column in value_df, by default (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    max_color_quantile : float, optional
        crops the colorscale at x quantile of the data, by default 0.98
    show_img : bool, optional
        whether to show img, by default True
    img : np.ndarray, optional
        tissue image, If not provided a black background image is used, by default None
    img_alpha : float, optional
        transparency of the image, by default 1.0
    adjust_text : bool, optional
        move text label to prevent overlap, by default False
    plt_axis : str, optional
        show axes?, by default "off"
    axis_y_flipped : bool, optional
        flip y axis to match coordinates of the plotted image, by default True
    x_y_labels : tuple, optional
        xy labels, by default ("", "")
    text_box_alpha : float, optional
        _description_, by default 0.9
    reorder_cmap : _type_, optional
        reorder colors to make sure you get the right color for each category, by default range(7)
    style : str, optional
        plot style (matplolib.style.context), by default "fast"
    colorbar_position : str, optional
        'bottom', 'right' or None, by default "bottom"
    colorbar_label_kw : dict, optional
        dict that will be forwarded to ax.set_label(), by default {}
    colorbar_shape : dict, optional
        cbar shape, by default {'vertical_gaps': 1.5, 'horizontal_gaps': 1.5,
                                    'width': 0.2, 'height': 0.2}
    colorbar_tick_size : int, optional
        colorbar ticks label size, by default 12
    colorbar_grid : tuple, optional
        tuple of colorbar grid (rows, columns), by default None
    image_cmap : str, optional
        matplotlib colormap for grayscale image, by default "Greys_r"
    white_spacing : int, optional
        percent of colorbars to be hidden, by default 20

    Returns
    -------
    figure

    """

    if value_df.shape[1] > 7:
        raise ValueError("Maximum of 7 cell types / factors can be plotted at the moment")
    
    labels = set_value(labels, value_df.columns)

    # create color maps
    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate(
            [[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)]
            )

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)

    # Create linearly scaled colormaps
    # ['#F0E442', '#D55E00', '#56B4E9', '#009E73', '#5A14A5', '#C8C8C8', '#323232']
    
    YellowCM = create_colormap(240, 228, 66)  # #F0E442 
    RedCM    = create_colormap(213, 94, 0)  # #D55E00
    BlueCM   = create_colormap(86, 180, 233)  # #56B4E9
    GreenCM  = create_colormap(0, 158, 115)  # #009E73
    GreyCM   = create_colormap(200, 200, 200)  # #C8C8C8
    WhiteCM  = create_colormap(50, 50, 50)  # #323232
    PurpleCM = create_colormap(90, 20, 165)  # #5A14A5

    cmaps = [YellowCM, RedCM, BlueCM,
             GreenCM, PurpleCM, GreyCM, WhiteCM]

    cmaps = [cmaps[i] for i in reorder_cmap]

    # set style
    with mpl.style.context(style):

        fig = plt.figure(figsize=figsize) if fig is None else fig

        if colorbar_position == "right":

            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {"vertical_gaps": 1.5, "horizontal_gaps": 0,
                     "width": 0.15, "height": 0.2}
            
            shape.update(colorbar_shape)

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {"vertical_gaps": 0.3, "horizontal_gaps": 0.6, 
                     "width": 0.2, "height": 0.035}
            shape.update(colorbar_shape)

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True) if ax is None else ax

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        # show image
        if img is not None and show_img:
            ax.imshow(img, 
                      aspect="equal", alpha=img_alpha, 
                      origin="lower", cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if invert_y:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        counts = value_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))

        colors  = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)

        for c in c_ord:
            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min([np.quantile(counts[:, c], 
                                                      max_color_quantile), 
                                          max_col[c]]
                                         )

            rgb_function = get_rgb_function(cmap=cmaps[c], 
                                            min_value=min_color_intensity, 
                                            max_value=max_color_intensity)

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(vmin=min_color_intensity,
                                        vmax=max_color_intensity)

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(labels[c], 
                                  **{**{"size": 'medium', "color": max_color, "alpha": 1},
                                     **colorbar_label_kw})

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w ** 2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = ((colors_ryb * kernel_weights).sum(axis=1) 
                               / kernel_weights.sum(axis=1))

        weighted_colors = np.zeros((weights.shape[0], 4))

        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)

        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)
        
        if kind == 'contour':
            coords, _ = _parse_coord(coords, is_contour=True)
            gdf = gpd.GeoDataFrame(geometry=coords)
            gdf.plot(ax=ax, color=weighted_colors, rasterized=raster)
        else:
            ax.scatter(x=coords[:, 0], y=coords[:, 1], 
                       c=weighted_colors, s=dot_size ** 2,
                       rasterized=raster)
        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props))

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))

    return ax


def plot_spatial(adata, colors, img_key="hires", show_img=False, **kwargs):
    """Plot spatial abundance of cell types (regulatory programmes) with colour gradient
    and interpolation (from Visium anndata).

    This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap).
    'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white'

    :param adata: adata object with spatial coordinates in adata.obsm['spatial']
    :param color: list of adata.obs column names to be plotted
    :param kwargs: arguments to plot_spatial_general
    :return: matplotlib figure
    """

    if show_img is True:
        kwargs["show_img"] = True
        kwargs["img"] = list(adata.uns["spatial"].values())[0]["images"][img_key]

    # location coordinates
    if "spatial" in adata.uns.keys():
        kwargs["coords"] = (
            adata.obsm["spatial"] * list(adata.uns["spatial"].values())[0]["scalefactors"][f"tissue_{img_key}_scalef"]
        )
    else:
        if kwargs['kind'] == 'contour':
            kwargs["coords"] = adata.obs['contour']
        else:
            kwargs["coords"] = adata.obsm["spatial"]

    ax = plot_spatial_general(value_df=fetch_data(adata, colors), **kwargs)  # cell abundance values

    return ax


import scanpy.external as sce

# filter out group with less than min_number of cells
def _filter_group(adata, groupby, min_number):
    """
    filter out group with less than min_number of cells
    """
    group_count = adata.obs[groupby].value_counts()
    group_count = group_count[group_count > min_number]
    adata = adata[adata.obs[groupby].isin(group_count.index), :]
    return adata

