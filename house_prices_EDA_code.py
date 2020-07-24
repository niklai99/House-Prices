import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
from jupyterthemes import jtplot
jtplot.reset()

def get_data():
    file = 'data/train.csv'
    df_train = pd.read_csv(file)
    return df_train

def sale_distr(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #distribution plot
    sns.distplot(sale_euro, color = '#007EFF', hist = True, ax = ax)

    #axis limits
    ax.set_xlim(left = sale_euro.min(), right = sale_euro.max())

    #axis labels
    ax.set_xlabel('Sale Price [€]', fontsize = 18)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 18)

    #title
    ax.set_title('Sale Prices Distribution', fontsize = 22)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #mean
    m = 'Mean: ' + format(sale_euro.mean(), '.0f')

    #standard deviation
    std = 'St. Dev.: ' + format(sale_euro.std(), '.0f')

    #skewness
    s = 'Skewness: ' + format(sale_euro.skew(), '.2f')

    #kurtosis
    k = 'Kurtosis: ' + format(sale_euro.kurt(), '.2f')

    ax.text(50e4, 5e-6, m + '\n' + '\n' + std + '\n' + '\n' + s + '\n' + '\n' + k, fontsize = 18)

    fig.savefig('plots/sale_distr.png', dpi = 300)

    plt.show()

def lin(x, a, b):
    return a * x + b

def scatter_gr_liv_area(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis='both', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #conversion from ft^2 to m^2
    conv = 0.092903
    gr_liv_area = df_train['GrLivArea'] * conv

    #fit model
    par_lin, cov_lin = curve_fit(lin, gr_liv_area, sale_euro)

    #scatter plot
    sns.scatterplot(x = gr_liv_area, y = sale_euro, color = '#007EFF', s = 60, ax = ax)

    #fit plot
    ax.plot(gr_liv_area, lin(gr_liv_area, *par_lin), color = '#ff8100', linewidth = 3, linestyle = 'dotted')

    #axis labels
    ax.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #title
    ax.set_title('Sale Prices vs Living Area', fontsize = 22)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    ax.set_xlim(left = 0, right = 550)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    fig.savefig('plots/scatter_gr_liv_area.png', dpi = 300)

    plt.show()

def scatter_basement_surface(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis='both', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #conversion from ft^2 to m^2
    conv = 0.092903
    basement_surface = df_train['TotalBsmtSF'] * conv

    #fit model
    par_lin, cov_lin = curve_fit(lin, basement_surface, sale_euro)

    #scatterplot
    sns.scatterplot(x = basement_surface, y = sale_euro, color = '#007EFF', s = 60, ax = ax)

    #fit plot
    ax.plot(basement_surface, lin(basement_surface, *par_lin), color = '#ff8100', linewidth = 3, linestyle = 'dotted')

    #axis labels
    ax.set_xlabel(r'Total Basement Area [m$^2$]', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #title
    ax.set_title('Sale Prices vs Basement Area', fontsize = 22)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    ax.set_xlim(left = -10, right = 600)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    fig.savefig('plots/scatter_basement_surface.png', dpi = 300)

    plt.show()

def strip_overall_qual(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis='y', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #strip plot
    sns.stripplot(x = df_train['OverallQual'], y = sale_euro, ax = ax, palette = 'husl')

    #axis labels
    ax.set_xlabel('Overall Quality from 1 to 10', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #axis title
    ax.set_title('Sale Prices vs Overall Quality', fontsize = 22)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    ax.set_xlim(left = -0.5, right = 9.5)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    fig.savefig('plots/strip_overall_qual.png', dpi = 300)

    plt.show()

def strip_year_building(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis='y', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #strip plot
    sns.stripplot(x = df_train['YearBuilt'], y = sale_euro, ax = ax)

    #axis labels
    ax.set_xlabel('Year of Builing', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #title
    ax.set_title('Sale Prices vs Year of Builing', fontsize = 22)

    #axis limits
    ax.set_xlim(left = -2, right = 112)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax.set(xticks=np.arange(-2, 118, 5), xticklabels = np.arange(1890, 2010, 5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    fig.savefig('plots/strip_year_building.png', dpi = 300)

    plt.show()

def corr_matrix(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)

    #correlation matrix
    corr_mat = df_train.corr()

    #heatmap
    sns.heatmap(corr_mat, square = True, ax = ax, cmap = 'viridis')

    fig.savefig('plots/corr_matrix.png', dpi = 300)

    plt.show()

def zoomed_corr_matrix(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)

    #number of variables for heatmap
    k = 10 

    #saleprice correlation matrix
    cols = df_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale = 1.25)
    hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 15}, yticklabels = cols.values, xticklabels = cols.values, ax = ax, cmap = 'viridis')

    fig.savefig('plots/zoomed_corr_matrix.png', dpi = 300)

    plt.show()

def correlated_scatters(df_train):
    #figure setup
    fig = plt.figure(figsize=(18,20))
    ax1 =  plt.subplot2grid((4, 2), (0, 0), rowspan=1, colspan=1)
    ax2 =  plt.subplot2grid((4, 2), (0, 1), rowspan=1, colspan=1)
    ax3 =  plt.subplot2grid((4, 2), (1, 0), rowspan=1, colspan=1)
    ax4 =  plt.subplot2grid((4, 2), (1, 1), rowspan=1, colspan=1)
    ax5 =  plt.subplot2grid((4, 2), (2, 0), rowspan=1, colspan=1)
    ax6 =  plt.subplot2grid((4, 2), (2, 1), rowspan=1, colspan=1)
    ax7 =  plt.subplot2grid((4, 2), (3, 0), rowspan=1, colspan=1)
    ax8 =  plt.subplot2grid((4, 2), (3, 1), rowspan=1, colspan=1)
    ax1.grid(axis='both', linewidth = .3)
    ax2.grid(axis='both', linewidth = .3)
    ax3.grid(axis='both', linewidth = .3)
    ax4.grid(axis='both', linewidth = .3)
    ax5.grid(axis='both', linewidth = .3)
    ax6.grid(axis='both', linewidth = .3)
    ax7.grid(axis='both', linewidth = .3)
    ax8.grid(axis='both', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #conversion from ft^2 to m^2
    conv = 0.092903
    gr_liv_area = df_train['GrLivArea'] * conv
    garage_area = df_train['GarageArea'] * conv
    basement_surface = df_train['TotalBsmtSF'] * conv
    first_surface = df_train['1stFlrSF'] * conv

    #scatter plots
    sns.scatterplot(x = df_train['OverallQual'], y = sale_euro, color = '#007EFF', s = 60, ax = ax1)
    sns.scatterplot(x = gr_liv_area, y = sale_euro, color = '#007EFF', s = 60, ax = ax2)
    sns.scatterplot(x = df_train['GarageCars'], y = sale_euro, color = '#007EFF', s = 60, ax = ax3)
    sns.scatterplot(x = garage_area, y = sale_euro, color = '#007EFF', s = 60, ax = ax4)
    sns.scatterplot(x = basement_surface, y = sale_euro, color = '#007EFF', s = 60, ax = ax5)
    sns.scatterplot(x = first_surface, y = sale_euro, color = '#007EFF', s = 60, ax = ax6)
    sns.scatterplot(x = df_train['FullBath'], y = sale_euro, color = '#007EFF', s = 60, ax = ax7)
    sns.scatterplot(x = df_train['YearBuilt'], y = sale_euro, color = '#007EFF', s = 60, ax = ax8)

    #axis labels
    ax1.set_xlabel('Overall Quality', fontsize = 15)
    ax1.set_ylabel('Sale Price [€]', fontsize = 15)
    ax2.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 15)
    ax2.set_ylabel('Sale Price [€]', fontsize = 15)
    ax3.set_xlabel('Cars fitting in the Garage', fontsize = 15)
    ax3.set_ylabel('Sale Price [€]', fontsize = 15)
    ax4.set_xlabel(r'Garage Area [m$^2$]', fontsize = 15)
    ax4.set_ylabel('Sale Price [€]', fontsize = 15)
    ax5.set_xlabel(r'Basement Surface [m$^2$]', fontsize = 15)
    ax5.set_ylabel('Sale Price [€]', fontsize = 15)
    ax6.set_xlabel(r'First Floor Surface [m$^2$]', fontsize = 15)
    ax6.set_ylabel('Sale Price [€]', fontsize = 15)
    ax7.set_xlabel('Bathrooms', fontsize = 15)
    ax7.set_ylabel('Sale Price [€]', fontsize = 15)
    ax8.set_xlabel('Year of Building', fontsize = 15)
    ax8.set_ylabel('Sale Price [€]', fontsize = 15)

    #axis ticks
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax1.yaxis.get_offset_text().set_fontsize(12)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax3.yaxis.get_offset_text().set_fontsize(12)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax4.yaxis.get_offset_text().set_fontsize(12)
    ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax4.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax5.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax5.yaxis.get_offset_text().set_fontsize(12)
    ax5.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax5.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax6.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax6.yaxis.get_offset_text().set_fontsize(12)
    ax6.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax6.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax7.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax7.yaxis.get_offset_text().set_fontsize(12)
    ax7.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax7.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    ax8.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax8.yaxis.get_offset_text().set_fontsize(12)
    ax8.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax8.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    fig.savefig('plots/correlated_scatters.png', dpi = 300)


    jtplot.reset()
    plt.show()

def missing_data_df(df_train):
    #total amount of missing data
    total = df_train.isnull().sum().sort_values(ascending = False)

    #missing data % 
    percent = ( df_train.isnull().sum() / df_train.isnull().count() ).sort_values(ascending = False)

    #make a dataframe
    missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

    return missing_data

def standardize_data(df_train):
    #standardizing data
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro
    saleprice_scaled = StandardScaler().fit_transform(sale_euro[:, np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
    high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

def sales_gr_liv_area_outliers(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis = 'both', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #conversion from ft^2 to m^2
    conv = 0.092903
    gr_liv_area = df_train['GrLivArea'] * conv

    #scatter plot
    sns.scatterplot(x = gr_liv_area, y = sale_euro, color = '#007EFF', s = 60, ax = ax)

    #bottom right outliers
    sns.scatterplot(x = gr_liv_area[gr_liv_area > 420], y = sale_euro, color = '#FF6C00', s = 60, ax = ax)

    #top right (fake) outliers
    sns.scatterplot(x = gr_liv_area, y = sale_euro[sale_euro > 6e5], color = '#01AE00', s = 60, ax = ax)

    #axis labels
    ax.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #title
    ax.set_title('Sale Prices vs Living Area', fontsize = 22)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    ax.set_xlim(left = 0, right = 550)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    fig.savefig('plots/sales_gr_liv_area_outliers.png', dpi = 300)

    plt.show()

def normal_sales(df_train):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #distribution plot
    sns.distplot(sale_euro, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"})

    #axis limits
    ax.set_xlim(left = sale_euro.min(), right = sale_euro.max())

    #axis labels
    ax.set_xlabel('Sale Price [€]', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Sale Prices Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 18})

    #probability plot
    stats.probplot(sale_euro, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel('Ordered Price Sales [€]', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    fig.savefig('plots/normal_sales.png', dpi = 300)

    plt.show()

def normal_sales_log(sale_euro_log):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #distribution plot
    sns.distplot(sale_euro_log, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"}, norm_hist = True)

    #axis limits
    ax.set_xlim(left = sale_euro_log.min(), right = sale_euro_log.max())

    #axis labels
    ax.set_xlabel('Sale Price [€] (log)', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Sale Prices Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 15})

    #probability plot
    stats.probplot(sale_euro_log, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel('Ordered Price Sales [€] (log)', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/normal_sales_log.png', dpi = 300)

    plt.show()

def normal_gr_liv_area(df_train):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #conversion from ft^2 to m^2
    conv = 0.092903
    gr_liv_area = df_train['GrLivArea'] * conv

    #distribution plot
    sns.distplot(gr_liv_area, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"})

    #axis limits
    ax.set_xlim(left = gr_liv_area.min(), right = gr_liv_area.max())

    #axis labels
    ax.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Above Ground Living Areas Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 18})

    #probability plot
    stats.probplot(gr_liv_area, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel(r'Ordered Living Areas [m$^2$]', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/normal_gr_liv_area.png', dpi = 300)

    plt.show()

def normal_gr_liv_area_log(gr_liv_area_log):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #distribution plot
    sns.distplot(gr_liv_area_log, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"}, norm_hist = True)

    #axis limits
    ax.set_xlim(left = gr_liv_area_log.min(), right = gr_liv_area_log.max())

    #axis labels
    ax.set_xlabel(r'Above Ground Living Area [m$^2$] (log)', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Above Ground Living Areas Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 15})

    #probability plot
    stats.probplot(gr_liv_area_log, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel(r'Ordered Living Areas [m$^2$] (log)', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/normal_gr_liv_area_log.png', dpi = 300)

    plt.show()

def normal_basement_surface(df_train):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #conversion from ft^2 to m^2
    conv = 0.092903
    basement_surface = df_train['TotalBsmtSF'] * conv

    #distribution plot
    sns.distplot(basement_surface, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"})

    #axis limits
    ax.set_xlim(left = basement_surface.min(), right = basement_surface.max())

    #axis labels
    ax.set_xlabel(r'Total Basement Area [m$^2$]', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Basement Areas Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 18})

    #probability plot
    stats.probplot(basement_surface, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel(r'Ordered Basement Areas [m$^2$]', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/normal_basement_surface.png', dpi = 300)

    plt.show()

def normal_basement_surface_log(basement_surface_log):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax =  plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 =  plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    #distribution plot
    sns.distplot(basement_surface_log, color = '#007EFF', hist = True, ax = ax, fit = norm, kde_kws={"label": "KDE"}, fit_kws={"label": "Gaussian Fit", "color": "#FF7F00"})

    #axis limits
    ax.set_xlim(left = basement_surface_log.min(), right = basement_surface_log.max())

    #axis labels
    ax.set_xlabel(r'Total Basement Area [m$^2$] (log)', fontsize = 15)
    ax.set_ylabel('Normalized Occurrencies', fontsize = 15)

    #title
    ax.set_title('Basement Areas Distribution', fontsize = 18)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    #legend
    ax.legend(loc = 'best', prop = {'size': 18})

    #probability plot
    stats.probplot(basement_surface_log, plot = ax2)

    #style
    ax2.get_lines()[0].set_color('#007EFF')
    ax2.get_lines()[1].set_color('#FF7F00')

    #axis labels
    ax2.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax2.set_ylabel(r'Ordered Basement Areas [m$^2$] (log)', fontsize = 15)

    #title
    ax2.set_title('Probability Plot', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/normal_basement_surface_log.png', dpi = 300)

    plt.show()

def scatter_homoscedasticity_1(sale_euro, gr_liv_area, sale_euro_log, gr_liv_area_log):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax1 = plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 = plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)
    
    ax1.grid(axis='both', linewidth = .3)
    ax2.grid(axis='both', linewidth = .3)

    #scatter plot
    sns.scatterplot(x = gr_liv_area, y = sale_euro, color = '#007EFF', s = 60, ax = ax1)

    #axis labels
    ax1.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 15)
    ax1.set_ylabel('Sale Price [€]', fontsize = 15)

    #title
    ax1.set_title('Sale Prices vs Living Area', fontsize = 18)

    #axis ticks
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax1.yaxis.get_offset_text().set_fontsize(12)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    ax1.set_xlim(left = 0, right = 450)
    ax1.set_ylim(bottom = 0, top = 7.5e5)

    #scatter plot log
    sns.scatterplot(x = gr_liv_area_log, y = sale_euro_log, color = '#007EFF', s = 60, ax = ax2)

    #axis labels
    ax2.set_xlabel(r'Above Ground Living Area [m$^2$] (log)', fontsize = 15)
    ax2.set_ylabel('Sale Price [€] (log)', fontsize = 15)

    #title
    ax2.set_title('Sale Prices vs Living Area (log)', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/scatter_homoscedasticity_1.png', dpi = 300)

    plt.show()

def scatter_homoscedasticity_1_kde(sale_euro, gr_liv_area, sale_euro_log, gr_liv_area_log):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax3 = plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax4 = plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    sns.kdeplot(gr_liv_area, sale_euro, cmap = 'rainbow', ax = ax3, shade = True, shade_lowest = False, cbar = True, gridsize=200)

    #axis labels
    ax3.set_xlabel(r'Above Ground Living Area [m$^2$]', fontsize = 15)
    ax3.set_ylabel('Sale Price [€]', fontsize = 15)

    #title
    #ax3.set_title('Sale Prices vs Living Area', fontsize = 18)

    #axis ticks
    ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax3.yaxis.get_offset_text().set_fontsize(12)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    sns.kdeplot(gr_liv_area_log, sale_euro_log, cmap = 'rainbow', ax = ax4, shade = True, shade_lowest = False, cbar = True, gridsize=200)

    #axis labels
    ax4.set_xlabel(r'Above Ground Living Area [m$^2$] (log)', fontsize = 15)
    ax4.set_ylabel('Sale Price [€] (log)', fontsize = 15)

    #title
    #ax4.set_title('Sale Prices vs Living Area', fontsize = 18)

    #axis ticks
    ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax4.yaxis.get_offset_text().set_fontsize(12)
    ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/scatter_homoscedasticity_1_kde.png', dpi = 300)

    plt.show()

def scatter_homoscedasticity_2(sale_euro, basement_surface_with_basement, sale_euro_log, basement_surface_log_with_basement):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax1 = plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax2 = plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)
    ax1.grid(axis='both', linewidth = .3)
    ax2.grid(axis='both', linewidth = .3)

    #scatter plot
    sns.scatterplot(x = basement_surface_with_basement, y = sale_euro, color = '#007EFF', s = 60, ax = ax1)

    #axis labels
    ax1.set_xlabel(r'Basement Surface [m$^2$]', fontsize = 15)
    ax1.set_ylabel('Sale Price [€]', fontsize = 15)

    #title
    ax1.set_title('Sale Prices vs Basement Surface', fontsize = 18)

    #axis ticks
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    #ax1.set_xlim(left = 0, right = 550)
    ax1.set_ylim(bottom = 0, top = 7.5e5)

    #scatter plot log
    sns.scatterplot(x = basement_surface_log_with_basement, y = sale_euro_log, color = '#007EFF', s = 60, ax = ax2)

    #axis labels
    ax2.set_xlabel(r'Basement Surface [m$^2$] (log)', fontsize = 15)
    ax2.set_ylabel('Sale Price [€] (log)', fontsize = 15)

    #title
    ax2.set_title('Sale Prices vs Basement Surface (log)', fontsize = 18)

    #axis ticks
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    #axis limits
    #ax2.set_xlim(left = 0, right = 550)
    #ax2.set_ylim(bottom = 0, top = 7.5e5)

    fig.savefig('plots/scatter_homoscedasticity_2.png', dpi = 300)

    plt.show()

def scatter_homoscedasticity_2_kde(sale_euro_basement, basement_surface_with_basement, sale_euro_log_basement, basement_surface_log_with_basement):
    #figure setup
    fig = plt.figure(figsize=(18,8))
    ax3 = plt.subplot2grid((1, 20), (0, 0), rowspan=1, colspan=9)
    ax4 = plt.subplot2grid((1, 20), (0, 11), rowspan=1, colspan=9)

    sns.kdeplot(basement_surface_with_basement, sale_euro_basement, cmap = 'rainbow', ax = ax3, shade = True, shade_lowest = False, cbar = True, gridsize=200)

    #axis labels
    ax3.set_xlabel(r'Basement Surface [m$^2$]', fontsize = 15)
    ax3.set_ylabel('Sale Price [€]', fontsize = 15)

    #title
    #ax3.set_title('Sale Prices vs Living Area', fontsize = 18)

    #axis ticks
    ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax3.yaxis.get_offset_text().set_fontsize(12)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))

    sns.kdeplot(basement_surface_log_with_basement, sale_euro_log_basement, cmap = 'rainbow', ax = ax4, shade = True, shade_lowest = False, cbar = True, gridsize=200)

    #axis labels
    ax4.set_xlabel(r'Basement Surface [m$^2$] (log)', fontsize = 15)
    ax4.set_ylabel('Sale Price [€] (log)', fontsize = 15)

    #title
    #ax4.set_title('Sale Prices vs Living Area', fontsize = 18)

    #axis ticks
    ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax4.yaxis.get_offset_text().set_fontsize(12)
    ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig('plots/scatter_homoscedasticity_2_kde.png', dpi = 300)

    plt.show()

def strip_neighborhood(df_train):
    #figure setup
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(axis='y', linewidth = .3)

    #conversion dollar to euro
    conv_dollar_to_euro = 0.88
    sale_euro = df_train['SalePrice'] * conv_dollar_to_euro

    #strip plot
    sns.stripplot(x = df_train['Neighborhood'], y = sale_euro, ax = ax)

    #axis labels
    ax.set_xlabel('Neighborhood', fontsize = 18)
    ax.set_ylabel('Sale Price [€]', fontsize = 18)

    #title
    ax.set_title('Sale Prices by Neighborhood', fontsize = 22)

    #axis limits
    #ax.set_xlim(left = -2, right = 112)
    ax.set_ylim(bottom = 0, top = 7.5e5)

    #axis ticks
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (3,3))
    #ax.set(xticks=np.arange(-2, 118, 5), xticklabels = np.arange(1890, 2010, 5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    #fig.savefig('plots/strip_year_building.png', dpi = 300)

    plt.show()