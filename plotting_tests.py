
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy as np
import pandas as pd
"""
"""
n = 50
x, y, z, s, ew = np.random.rand(5, n)
c, ec = np.random.rand(2, n, 4)
area_scale, width_scale = 500, 5

fig, ax = plt.subplots()
#sc = ax.scatter(x, y, s=np.square(s)*area_scale, c=c, edgecolor=ec, linewidth=ew*width_scale)
sc = ax.scatter(x, y, s=np.square(s)*area_scale, c=c)
ax.grid()

plt.show()
"""

"""
x = np.arange(1,10) # generates range of numbers from 0 to 9 since set (, ]
y = 2 * x
#z = ['One','Two','Three'] # list, was set with {}
z = x ** 2

fig, ax = plt.subplots()
ax.plot(x, y, 'bo-')

for X, Y, Z in zip(x, y, z):
    if Y > 6:
        # Annotate the points 5 _points_ above and to the left of the vertex
        # Annotate only points for y-axis value > 6
        ax.annotate('{}'.format(Z), xy=(X,Y), xytext=(-25, 0), ha='right',
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', shrinkA=0)
                )
plt.show()
"""


"""
#z = ['One','Two','Three'] # list, was set with {}
n=10
c = np.random.rand( n, 4)
df = pd.read_csv('full_results.csv', header=0, index_col=0)
#df.plot( kind='scatter', x='C', y='E', s=df['B'])
#plt.show()

x_data = 'Income_EJ'
y_data = 'Race_EJ'
s_data = 'Deaths'
rank_data = 'Deaths'

x = df[[ x_data ]].values
y = df[[ y_data ]].values
s = df[[ s_data ]].values

# maximum bubble size
bubblemax = 1000
smax, smin = np.amax( s ), np.amin( s )
s = s / smax * bubblemax + 10
srad = np.sqrt( s / np.pi )

# calculate base value for top ten 
df_ranked = df.rank( axis=0, ascending=False )

label_rank = 11
idx = df_ranked[ df_ranked[ rank_data ] < label_rank ].index.tolist()

offset = 3

fig, ax = plt.subplots()



ax.scatter( x, y, s=s, c=c)
ax.set_alpha( 0.6)

for X, Y, Z in zip( 
    df[ x_data ][ idx ].values,
    df[ y_data ][ idx ].values,
    df[ x_data ][idx].index
    ):  # took out R = srad, which goes in xytext
    ax.annotate('{}'.format( Z ), xy=( X, Y ), 
        xytext=(20+offset,20), ha='right', textcoords='offset points',
        arrowprops=dict( arrowstyle='-', shrinkA=0), fontsize=9
        )

at = AnchoredText( 'Label: Top 10 by %s \nBubble size: %s' % (rank_data, s_data),
                  prop=dict(fontsize=9), frameon=True,
                  loc=2,
                  )
at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
ax.add_artist(at)

ax.grid()

plt.ylabel( y_data, fontsize=9 )
plt.xlabel( x_data, fontsize=9 )

plt.show()
"""

import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",linewidth=0, stacked=True, ax = axe, legend=False, grid=False, **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0,n_df*n_col,n_col): # len(h) = n_col * n_df
        for j,pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x()+1/float(n_df+1)*i/float(n_col))
                rect.set_hatch(H*(i/n_col))
                rect.set_width(1/float(n_df+1))

    axe.set_xticks((np.arange(0, 2*n_ind, 2)+1/float(n_df+1))/2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0,0,color = "gray", hatch=H*i))

    l1 = axe.legend(h[:n_col],l[:n_col],loc=[1.01,0.5])
    if labels is not None:
        l2 = plt.legend(n,labels,loc=[1.01,0.1]) 
    axe.add_artist(l1)
    return axe

# create fake dataframes
df1 = pd.DataFrame(np.random.rand(4,5),index=["A","B","C","D"],columns=["I","J","K","L","M"])
df2 = pd.DataFrame(np.random.rand(4,5),index=["A","B","C","D"],columns=["I","J","K","L","M"])
df3 = pd.DataFrame(np.random.rand(4,5),index=["A","B","C","D"],columns=["I","J","K","L","M"])

# Then, just call :
plot_clustered_stacked([df1,df2,df3],["df1","df2","df3"])
plt.show()