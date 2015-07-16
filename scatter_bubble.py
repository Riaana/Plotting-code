
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