"""
This type stub file was generated by pyright.
"""

from ._core import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from .axisgrid import _facet_docs
from ._decorators import _deprecate_positional_args
from ._docstrings import DocstringComponents, _core_docs

"""Plotting functions for visualizing distributions."""
_dist_params = dict(multiple="""
multiple : {{"layer", "stack", "fill"}}
    Method for drawing multiple elements when semantic mapping creates subsets.
    Only relevant with univariate data.
    """, log_scale="""
log_scale : bool or number, or pair of bools or numbers
    Set a log scale on the data axis (or axes, with bivariate data) with the
    given base (default 10), and evaluate the KDE in log space.
    """, legend="""
legend : bool
    If False, suppress the legend for semantic variables.
    """, cbar="""
cbar : bool
    If True, add a colorbar to annotate the color mapping in a bivariate plot.
    Note: Does not currently support plots with a ``hue`` variable well.
    """, cbar_ax="""
cbar_ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the colorbar.
    """, cbar_kws="""
cbar_kws : dict
    Additional parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.
    """)
_param_docs = DocstringComponents.from_nested_components(core=,, facets=DocstringComponents(_facet_docs), dist=DocstringComponents(_dist_params), kde=DocstringComponents.from_function_params(KDE.__init__), hist=DocstringComponents.from_function_params(Histogram.__init__), ecdf=DocstringComponents.from_function_params(ECDF.__init__))
class _DistributionPlotter(VectorPlotter):
    semantics = ...
    wide_structure = ...
    flat_structure = ...
    def __init__(self, data=..., variables=...) -> None:
        ...
    
    @property
    def univariate(self):
        """Return True if only x or y are used."""
        ...
    
    @property
    def data_variable(self):
        """Return the variable with data for univariate plots."""
        ...
    
    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        ...
    
    def plot_univariate_histogram(self, multiple, element, fill, common_norm, common_bins, shrink, kde, kde_kws, color, legend, line_kws, estimate_kws, **plot_kws):
        ...
    
    def plot_bivariate_histogram(self, common_bins, common_norm, thresh, pthresh, pmax, color, legend, cbar, cbar_ax, cbar_kws, estimate_kws, **plot_kws):
        ...
    
    def plot_univariate_density(self, multiple, common_norm, common_grid, fill, legend, estimate_kws, **plot_kws):
        ...
    
    def plot_bivariate_density(self, common_norm, fill, levels, thresh, color, legend, cbar, cbar_ax, cbar_kws, estimate_kws, **contour_kws):
        ...
    
    def plot_univariate_ecdf(self, estimate_kws, legend, **plot_kws):
        ...
    
    def plot_rug(self, height, expand_margins, legend, **kws):
        ...
    


class _DistributionFacetPlotter(_DistributionPlotter):
    semantics = ...


def histplot(data=..., *, x=..., y=..., hue=..., weights=..., stat=..., bins=..., binwidth=..., binrange=..., discrete=..., cumulative=..., common_bins=..., common_norm=..., multiple=..., element=..., fill=..., shrink=..., kde=..., kde_kws=..., line_kws=..., thresh=..., pthresh=..., pmax=..., cbar=..., cbar_ax=..., cbar_kws=..., palette=..., hue_order=..., hue_norm=..., color=..., log_scale=..., legend=..., ax=..., **kwargs):
    ...

@_deprecate_positional_args
def kdeplot(x=..., *, y=..., shade=..., vertical=..., kernel=..., bw=..., gridsize=..., cut=..., clip=..., legend=..., cumulative=..., shade_lowest=..., cbar=..., cbar_ax=..., cbar_kws=..., ax=..., weights=..., hue=..., palette=..., hue_order=..., hue_norm=..., multiple=..., common_norm=..., common_grid=..., levels=..., thresh=..., bw_method=..., bw_adjust=..., log_scale=..., color=..., fill=..., data=..., data2=..., **kwargs):
    ...

def ecdfplot(data=..., *, x=..., y=..., hue=..., weights=..., stat=..., complementary=..., palette=..., hue_order=..., hue_norm=..., log_scale=..., legend=..., ax=..., **kwargs):
    ...

@_deprecate_positional_args
def rugplot(x=..., *, height=..., axis=..., ax=..., data=..., y=..., hue=..., palette=..., hue_order=..., hue_norm=..., expand_margins=..., legend=..., a=..., **kwargs):
    ...

def displot(data=..., *, x=..., y=..., hue=..., row=..., col=..., weights=..., kind=..., rug=..., rug_kws=..., log_scale=..., legend=..., palette=..., hue_order=..., hue_norm=..., color=..., col_wrap=..., row_order=..., col_order=..., height=..., aspect=..., facet_kws=..., **kwargs):
    ...

def distplot(a=..., bins=..., hist=..., kde=..., rug=..., fit=..., hist_kws=..., kde_kws=..., rug_kws=..., fit_kws=..., color=..., vertical=..., norm_hist=..., axlabel=..., label=..., ax=..., x=...):
    """DEPRECATED: Flexibly plot a univariate distribution of observations.

    .. warning::
       This function is deprecated and will be removed in a future version.
       Please adapt your code to use one of two new functions:

       - :func:`displot`, a figure-level function with a similar flexibility
         over the kind of plot to draw
       - :func:`histplot`, an axes-level function for plotting histograms,
         including with kernel density smoothing

    This function combines the matplotlib ``hist`` function (with automatic
    calculation of a good default bin size) with the seaborn :func:`kdeplot`
    and :func:`rugplot` functions. It can also fit ``scipy.stats``
    distributions and plot the estimated PDF over the data.

    Parameters
    ----------
    a : Series, 1d-array, or list.
        Observed data. If this is a Series object with a ``name`` attribute,
        the name will be used to label the data axis.
    bins : argument for matplotlib hist(), or None, optional
        Specification of hist bins. If unspecified, as reference rule is used
        that tries to find a useful default.
    hist : bool, optional
        Whether to plot a (normed) histogram.
    kde : bool, optional
        Whether to plot a gaussian kernel density estimate.
    rug : bool, optional
        Whether to draw a rugplot on the support axis.
    fit : random variable object, optional
        An object with `fit` method, returning a tuple that can be passed to a
        `pdf` method a positional arguments following a grid of values to
        evaluate the pdf on.
    hist_kws : dict, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.hist`.
    kde_kws : dict, optional
        Keyword arguments for :func:`kdeplot`.
    rug_kws : dict, optional
        Keyword arguments for :func:`rugplot`.
    color : matplotlib color, optional
        Color to plot everything but the fitted curve in.
    vertical : bool, optional
        If True, observed values are on y-axis.
    norm_hist : bool, optional
        If True, the histogram height shows a density rather than a count.
        This is implied if a KDE or fitted density is plotted.
    axlabel : string, False, or None, optional
        Name for the support axis label. If None, will try to get it
        from a.name if False, do not set a label.
    label : string, optional
        Legend label for the relevant component of the plot.
    ax : matplotlib axis, optional
        If provided, plot on this axis.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    See Also
    --------
    kdeplot : Show a univariate or bivariate distribution with a kernel
              density estimate.
    rugplot : Draw small vertical lines to show each observation in a
              distribution.

    Examples
    --------

    Show a default plot with a kernel density estimate and histogram with bin
    size determined automatically with a reference rule:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns, numpy as np
        >>> sns.set_theme(); np.random.seed(0)
        >>> x = np.random.randn(100)
        >>> ax = sns.distplot(x)

    Use Pandas objects to get an informative axis label:

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> x = pd.Series(x, name="x variable")
        >>> ax = sns.distplot(x)

    Plot the distribution with a kernel density estimate and rug plot:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, rug=True, hist=False)

    Plot the distribution with a histogram and maximum likelihood gaussian
    distribution fit:

    .. plot::
        :context: close-figs

        >>> from scipy.stats import norm
        >>> ax = sns.distplot(x, fit=norm, kde=False)

    Plot the distribution on the vertical axis:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, vertical=True)

    Change the color of all the plot elements:

    .. plot::
        :context: close-figs

        >>> sns.set_color_codes()
        >>> ax = sns.distplot(x, color="y")

    Pass specific parameters to the underlying plot functions:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, rug=True, rug_kws={"color": "g"},
        ...                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
        ...                   hist_kws={"histtype": "step", "linewidth": 3,
        ...                             "alpha": 1, "color": "g"})

    """
    ...

