import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import linalg as LA
from numpy import delete
from collections import Counter, defaultdict
from datetime import datetime as dt
from os import listdir
import statsmodels.api as sm
from scipy.optimize import curve_fit
import re
import sys
import random


# change the version number here!
__version__ = "$version = py3.0.1$"

def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush()


'''
attention_data

type: a toy data for explanations
'''

attention_data = [['a',0],['a',1],['a',2],\
                    ['b',1],['b',2],['c',1],\
                    ['c',2],['c',3],['d',2],\
                    ['d',3],['e',0],['e',4],\
                    ['f',0],['f',4],['g',0],\
                    ['g',4],['g',5],['h',0],\
                    ['h',5],['i',6]]

############################
# construct flow network
############################

def constructFlowNetwork(C):
    '''
    C is an array of two dimentions, e.g., 
    C = np.array([[user1, item1], 
                  [user1, item2], 
                  [user2, item1], 
                  [user2, item3]])
    Return a balanced flow network
    '''
    E=defaultdict(lambda:0)
    E[('source',C[0][1])]+=1
    E[(C[-1][1],'sink')]+=1
    F=zip(C[:-1],C[1:])
    for i in F:
        if i[0][0]==i[1][0]:
            if i[0][1]!=i[1][1]: # delete self-loop
                E[(i[0][1],i[1][1])]+=1
            #E[(i[0][1],i[1][1])]+=1 # keep the self-loop
        else:
            E[(i[0][1],'sink')]+=1
            E[('source',i[1][1])]+=1
    G=nx.DiGraph()
    for i,j in E.items():
        x,y=i
        G.add_edge(x,y,weight=j)
    return G

    
###################
# flow functions
###################

def flowBalancing(G):
    H = G.copy()
    O = G.out_degree(weight='weight')
    I = G.reverse().out_degree(weight='weight')
    for i in dict(O).keys():
        if i =='sink' or i=='source':
            continue
        de = I[i]-O[i]
        if de > 0:
            H.add_edge(i,'sink',weight=de)
        elif de < 0:
            H.add_edge('source',i,weight=-de)
    return H

def flowDistanceFromSource(G): #input a balanced nx graph
    R = G.reverse()
    mapping = {'source':'sink','sink':'source'} 
    H = nx.relabel_nodes(R,mapping)
    #---------initialize flow distance dict------
    L = dict((i,1) for i in G.nodes())
    #---------prepare weighted out-degree dict------
    T = G.out_degree(weight='weight')
    #---------iterate until converge------------
    ls = np.array(list(L.values()))
    delta = len(L)*0.01 + 1
    k=0
    while delta > len(L)*0.01:
        k+=1
        if k>20:
            break
        for i in L:
            l=1
            for m,n in H.edges(i):
                try:
                    l+=L[n]*list(H[m][n].values())[0]/float(T[m])
                except Exception as e:
                    print(e)
                    l = l
                    pass
            L[i]=l
        delta = np.sum(np.abs(np.array(list(L.values())) - ls))
        
        ls = np.array(list(L.values()))
        
    #---------clean the result-------
    del L['sink']
    for i in L:
        L[i]-=1
    L['sink'] = L.pop('source')
    return L

# L = flowDistanceFromSource(G)

def outflow(G,node):
    '''
    return the out-flow for a balanced flow network
    '''
    n=0
    for i in G.edges(node):
        n+=G[i[0]][i[1]].values()[0]
    return n

def inflow(G,node):
    '''
    return the in-flow for a balanced flow network
    '''
    return outflow(G.reverse(), node)    


def getAICI(H):
    '''
    return AI & CI
    Source: Wu and Zhang 2013 The decentralized flow structure of clickstreams on the web. EPJB
    '''
    # H = flowBalancing(G)
    F1=nx.to_numpy_matrix(H)
    sourcep=list(H.nodes()).index('source')
    sinkp=list(H.nodes()).index('sink')
    F1oJ=F1[sourcep,]
    AI = F1.sum(0)

    if sinkp > sourcep:

        F1oJ=delete(F1oJ,sourcep,1)
        F1oJ=delete(F1oJ,sinkp-1,1)
        AI=delete(AI,sourcep,1)
        AI=delete(AI,sinkp-1,1)
        F1[sinkp,sinkp]=1
        M = F1 / F1.sum(axis=1)
        M = delete(M, sinkp, 0) 
        M = delete(M, sinkp, 1) 
        I = np.identity(len(M))
        U =  LA.inv( I - M)
        U = delete(U, sourcep, 0) 
        U = delete(U, sourcep, 1)
    else:
        F1oJ=delete(F1oJ,sinkp,1)
        F1oJ=delete(F1oJ,sourcep-1,1)
        AI=delete(AI,sinkp,1)
        AI=delete(AI,sourcep-1,1)
        F1[sinkp,sinkp]=1
        M = F1 / F1.sum(axis=1)
        M = delete(M, sinkp, 0) 
        M = delete(M, sinkp, 1) 
        I = np.identity(len(M))
        U =  LA.inv( I - M)
        U = delete(U, sourcep-1, 0) 
        U = delete(U, sourcep-1, 1)     
    def calculateCi(i):
        Gi = np.sum(np.dot(F1oJ,U[:,i]))/U[i,i]
        return np.sum(U[i,:])*Gi

    CI = map( lambda x:calculateCi(x),range(len(U)) )
    AI = AI.tolist()[0]
    return np.array(AI), np.array(list(CI))
    


# ai, ci = getAICI(G)
# alloRegressPlot(ai, ci, xlab='$A_i$', ylab='$C_i$', col='r', mark='o', loglog=True)

def networkDissipate(G):
    '''
    return flowToSink,totalFlow,flowFromSource
    '''
    D=defaultdict(lambda:[0,0,0])#toSink,totalflow,fromSource
    for x,y in G.edges(): # when x=y, it is a self-loop
        w = list(G[x][y].values())[0] 
        if y == 'sink':
            D[x][0]+=w
        if x != 'source':
            D[x][1]+=w
        elif x == 'source':
            D[y][2]+=w
    return D

# di = networkDissipate(G)
# toSink,totalflow,fromSource = np.array(list(di.values())).T
# toflow = totalflow-toSink

def getFlowMatrix(G,nodelist=None):
    '''
    read Graph and construct flowMatrix
    '''
    if nodelist is None:
        FM = nx.to_numpy_matrix(G)

    FM = nx.to_numpy_matrix(G,nodelist)
    return FM

def getMarkovMatrix(m):
    '''
    read flowMatrix and construct MarkovMatrix      
    '''
    n = len(m)
    mm = np.zeros((n,n),np.float)
    for i in range(n):
        for j in range(n):
            if m[i,j]>0:
                mm[i,j] = float(m[i,j])/float((m[i,0:].sum()))

    return mm

def getUmatrix(G):
    '''
    computing the cumulative MarkovMatrix of infinite steps  
    '''
    H = flowBalancing(G)
    F1=nx.to_numpy_matrix(H)
    sourcep=list(H.nodes()).index('source')
    sinkp=list(H.nodes()).index('sink')
    F1oJ=F1[sourcep,]
    if sinkp > sourcep:
        F1oJ=delete(F1oJ,sourcep,1)
        F1oJ=delete(F1oJ,sinkp-1,1)
        F1[sinkp,sinkp]=1
        M = F1 / F1.sum(axis=1)
        M = delete(M, sinkp, 0) 
        M = delete(M, sinkp, 1) 
        I = np.identity(len(M))
        U =  LA.inv( I - M)
        U = delete(U, sourcep, 0) 
        U = delete(U, sourcep, 1)
    else:
        F1oJ=delete(F1oJ,sinkp,1)
        F1oJ=delete(F1oJ,sourcep-1,1)
        F1[sinkp,sinkp]=1
        M = F1 / F1.sum(axis=1)
        M = delete(M, sinkp, 0) 
        M = delete(M, sinkp, 1) 
        I = np.identity(len(M))
        U =  LA.inv( I - M)
        U = delete(U, sourcep-1, 0) 
        U = delete(U, sourcep-1, 1)     
    return U

def averageFlowLength(G):
    '''
    return the average flow length
    '''
    H = flowBalancing(G)
    F1=nx.to_numpy_matrix(H)
    sinkp=list(H.nodes()).index('sink')
    sourcep=list(H.nodes()).index('source')
    F1[sinkp,sinkp]=1
    M = F1 / F1.sum(axis=1)
    M = np.delete(M, sinkp, 0) 
    M = np.delete(M, sinkp, 1) 
    I = np.identity(len(M))
    U =  LA.inv( I - M)
    if sinkp > sourcep:
        L = np.sum(U[sourcep,])
    else:
        L = np.sum(U[sourcep-1,])
    return L 

def getAverageTimeMatrix(G):
    '''
    return the average time matrix
    '''
    H = flowBalancing(G)

    hn = H.nodes()
    hn.remove('source') 
    hn.insert(0,'source')

    m = getFlowMatrix(H,hn)
    M = getMarkovMatrix(m)

    U = getUmatrix(M)
    T = np.dot(U , U)
    T = np.dot(M , T)
    K = np.divide(T,U)

    return K
  

def log_binning(x, y, bin_count=20):
    max_x = np.log10(max(x))
    max_y = np.log10(max(y))
    max_base = max([max_x,max_y])
    xx = [i for i in x if i>0]
    min_x = np.log10(np.min(xx))
    bins = np.logspace(min_x,max_base,num=bin_count)
    bin_means_y = (np.histogram(x,bins,weights=y)[0] / np.histogram(x,bins)[0])
    bin_means_x = (np.histogram(x,bins,weights=x)[0] / np.histogram(x,bins)[0])
    return bin_means_x,bin_means_y
    
def alloRegressPlotLogBinning(xdata,ydata,col,mark,xlab,ylab, loglog):
    ti, di = log_binning(xdata,ydata,bin_count=10)
    x=np.log(ti);y=np.log(di);
    xx = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,xx).fit()
    constant=res.params[0];beta=res.params[1]; r2=res.rsquared
    plt.plot(xdata,ydata,mark,color=col, alpha = 0.3)
    plt.plot(ti, di, 'ro')
    xs = np.linspace(min(ti),max(ti),100)
    plt.plot(xs,np.exp(constant)*xs**beta,color='r',linestyle='-', 
             label = '$\\gamma$ = '+ str(np.round(beta,2)) + ' , ' \
         + '$R^2$ = ' + str(np.round(r2,3)))
    plt.legend(loc = 0)
    if loglog == True:
        plt.xscale('log');plt.yscale('log')
    plt.xlabel(xlab, fontsize = 20);plt.ylabel(ylab, fontsize = 20)
    minx,maxx=plt.xlim(); miny,maxy=plt.ylim()

#######################
# fitting function
########################

def powerLawExponentialCutOffPlot(data, xlab, ylab):
    '''
    Plot fitted powerLaw distribution with Exponential CutOff
    '''
    t = np.array(sorted(data,key=lambda x:-x))
    r = np.array(range(len(data))) +1
    r = r/float(np.max(r))
    y = np.log(r)
    x1 = np.log(t)
    x2 = t
    x = np.column_stack((x1,x2))
    x = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,x).fit()
    L,alpha,lambde = res.params
    r2 = res.rsquared
    plt.plot(t,r,".",color="SteelBlue",alpha=0.75,markersize=10)
    plt.plot(t, np.exp(L) * t ** alpha * np.exp(lambde * t),"r-")
    plt.xscale('log'); plt.yscale('log')
    plt.ylim(ymax = 1)
    plt.xlabel(xlab, fontsize = 20)
    plt.ylabel(ylab, fontsize = 20)
    return [L,alpha,lambde, r2]

def DGBDPlot(data):
    '''
    plot fitted DGBD distribution

    Source: Wu, L. and J. Zhang (2011), Accelerating growth and size-dependent distribution ofhuman online activities. Physical Review E, 84 (2): 026113-026117.
    '''
    t=np.array(sorted(data,key=lambda x:-x))
    r=np.array(range(1,len(data)+1))   
    y = np.log(t)
    x1 = np.log(max(r)+1-r)
    x2 = np.log(r)
    x = np.column_stack((x1,x2))
    x = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,x).fit()
    [A,b,a] = res.params
    plt.plot(r,t,"o",color="b")
    plt.plot(r, np.exp(A)*(max(r)+1-r)**b*r**a,"r-")
    plt.yscale('log')
    plt.text(max(r)/2,max(t)/50,"b=" + str(round(b,2)) + ", a=" + str(round(a,2)))
    plt.xlabel(r'$Rank$', fontsize = 20)
    plt.ylabel(r'$Frequency$', fontsize = 20)

    
def gini_coefficient(v):
    '''
    return the bins, yvals, gini_val for gini_coefficient
    '''
    bins = np.linspace(0., 100., 11)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return bins, yvals, gini_val



def plotPowerlaw(data,ax,col,xlab):
    '''
    plot power power distribution with powerlaw package
    '''
    import powerlaw
    fit = powerlaw.Fit(data,xmin=1)
    fit.plot_pdf(color = col, linewidth = 2)
    fit = powerlaw.Fit(data)
    a,x = (fit.power_law.alpha,fit.power_law.xmin)
    fit.power_law.plot_pdf(color = col, linestyle = 'dotted', ax = ax, \
                            label = r"$\alpha = %d \:\:, x_{min} = %d$" % (a,x))
    ax.set_xlabel(xlab, fontsize = 20)
    ax.set_ylabel('$Probability$', fontsize = 20)
    plt.legend(loc = 0, frameon = False)
    
def plotCCDF(data,ax,col,xlab):
    '''
    plot CCDF power power distribution with powerlaw package
    '''
    import powerlaw
    fit = powerlaw.Fit(data,xmin=1)
    fit.plot_ccdf(color = col, linewidth = 2)
    fit = powerlaw.Fit(data)
    a,x = (fit.power_law.alpha,fit.power_law.xmin)
    fit.power_law.plot_ccdf(color = col, linestyle = 'dotted', ax = ax, \
                            label = r"$\alpha = %d \:\:, x_{min} = %d$" % (a,x))
    ax.set_xlabel(xlab, fontsize = 16)
    ax.set_ylabel('$CCDF$', fontsize = 16)
    plt.legend(loc = 0, frameon = False)

def alloRegressPlot(xdata,ydata,col,mark,xlab,ylab, loglog):
    '''
    Plot the fitted allowmetric growth 
    '''
    x=np.log(xdata+1);y=np.log(ydata+1);
    xx = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,xx).fit()
    constant=res.params[0];beta=res.params[1]; r2=res.rsquared
    plt.plot(xdata,ydata,mark,color=col, label = None, alpha = 0.3)
    if loglog == True:
        plt.xscale('log');plt.yscale('log')
    plt.xlabel(xlab, fontsize = 20);plt.ylabel(ylab, fontsize = 20)
    minx,maxx=plt.xlim(); miny,maxy=plt.ylim()
    xs = np.linspace(min(xdata),max(xdata),100)
    plt.plot(xs,np.exp(constant)*xs**beta,color='r',linestyle='-', 
            label = '$\\alpha$ = '+ str(np.round(beta,2)) + ' , ' \
         + '$R^2$ = ' + str(np.round(r2,2)))
    plt.legend(loc = 0, frameon = False)

def linearRegressPlot(xdata,ydata,col,mark,xlab,ylab):
    '''
    Plot the fitted linear Regression
    '''
    x=xdata;y=ydata
    xx = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,xx).fit()
    constant=res.params[0];beta=res.params[1]; r2=res.rsquared
    plt.plot(xdata,ydata,mark,color=col)
    plt.xlabel(xlab, fontsize = 20);plt.ylabel(ylab, fontsize = 20)
    minx,maxx=plt.xlim(); miny,maxy=plt.ylim()
    plt.text(min(xdata)+(max(xdata)-min(xdata))/10,
         min(ydata)+(max(ydata)-min(ydata))/2,
         '$\\alpha$ = '+ str(np.round(beta,2)) + ' , ' \
         + '$R^2$ = ' + str(np.round(r2,2)) )
    xs = np.linspace(min(xdata),max(xdata),100)
    plt.plot(xs,constant + xs*beta,color='r',linestyle='-')

######################
# Plotting Function
######################

def WebtoTree(G):
    H = flowBalancing(G)
    T = H.out_degree(weight='weight')
    R = H.reverse()
    L = flowDistanceFromSource(H)
    L['source']=0
    S=defaultdict(lambda:[])
    for i in H.nodes():
        if i!='source':
            es=R[i]
            w,k=sorted([(es[j]['weight'],j) for j in es],reverse=True)[0]
            S[k].append(i)
    return H,T,L,S



def circle(r):
    radius = r
    angle = random.uniform(0, 2*np.pi)
    return [radius*np.sin(angle), radius*np.cos(angle)]




#Tree ploting functions
# from http://billmill.org/pymag-trees/

class Tree:
    def __init__(self, node="", *children):
        self.node = node
        self.width = len(node)
        if children: self.children = children
        else:        self.children = []
    def __str__(self): 
        return "%s" % (self.node)
    def __repr__(self):
        return "%s" % (self.node)
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice): 
            return self.children[key]
        if isinstance(key, str):
            for child in self.children:
                if child.node == key: return child
    def __iter__(self): return self.children.__iter__()
    def __len__(self): return len(self.children)
    def addChild(self,nodeName): self.children.append(nodeName)

class DrawTree(object):
    def __init__(self, tree, parent=None, depth=0, number=1):
        self.x = -1.
        self.y = depth
        self.tree = tree
        self.children = [DrawTree(c, self, depth+1, i+1) 
                         for i, c
                         in enumerate(tree.children)]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        #this is the number of the node in its group of siblings 1..n
        self.number = number

    def left(self): 
        return self.thread or len(self.children) and self.children[0]

    def right(self):
        return self.thread or len(self.children) and self.children[-1]

    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self: return n
                else:            n = node
        return n

    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and self != \
        self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling
    lmost_sibling = property(get_lmost_sibling)

    def __str__(self): return "%s: x=%s mod=%s" % (self.tree, self.x, self.mod)
    def __repr__(self): return self.__str__()        
        
def buchheim(tree):
    dt = firstwalk(DrawTree(tree))
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt

def third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        third_walk(c, n)

def firstwalk(v, distance=1.):
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.lbrother().x + distance
        else:
            v.x = 0.
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            firstwalk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        #print "finished v =", v.tree, "children"
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        ell = v.children[0]
        arr = v.children[-1]
        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v

def apportion(v, default_ancestor, distance):
    w = v.lbrother()
    if w is not None:
        #in buchheim notation:
        #i == inner; o == outer; r == right; l == left; r = +; l = -
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor

def move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    #print wl.tree, "is conflicted with", wr.tree, 'moving', subtrees, 'shift', shift
    #print wl, wr, wr.number, wl.number, shift, subtrees, shift/subtrees
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift

def execute_shifts(v):
    shift = change = 0
    for w in v.children[::-1]:
        #print "shift:", w, shift, w.change
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change

def ancestor(vil, v, default_ancestor):
    #the relevant text is at the bottom of page 7 of
    #"Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al, (2002)
    #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757&rep=rep1&type=pdf
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor

def second_walk(v, m=0, depth=0, min=None):
    v.x += m
    v.y = depth

    if min is None or v.x < min:
        min = v.x

    for w in v.children:
        min = second_walk(w, m + v.mod, depth+1, min)

    return min

def generateTree(edgeDic):
    allNodes={}
    for k,v in edgeDic.items():
        if k in allNodes:
            n=allNodes[k]
        else:
            n=Tree(k,)
            allNodes[k]=n
        for s in v:
            if s in allNodes:
                cn=allNodes[s]
            else:
                cn=Tree(s,)
                allNodes[s]=cn
            allNodes[k].addChild(cn)
    return allNodes

def width(apex,xm=0):
    if not apex.children:
        return xm
    for child in apex.children:
        if child.x > xm:
            xm = child.x
            #print xm
        xm = width(child,xm)
    return xm

def depth(root,node,h=0):
    if str(root.tree)==str(node):
        h= root.y
    else:
        for i in root.children:
            h = depth(i,node,h)
    return h

def angleCo(x,y,xm,ym):
    angle=2*ym*np.pi*x/(xm+1)
    nx,ny=y*np.sin(angle), y*np.cos(angle)
    return nx,ny

def drawt(ax,root,rawVersion,circle,J,U,max_x,max_y):
    x=root.x
    if rawVersion==True:
        y=root.y
    else:
        y=J[str(root.tree)]
    if circle == True:
        x,y=angleCo(x,y,max_x,max_y)
    if str(root.tree)!='source':
        ax.scatter(x, y, facecolor='c',lw = 0,alpha=1,
                    s=200*U[str(root.tree)]/max(U.values())+3,zorder=2)
        ax.text(x, y, root.tree, color = 'red', fontsize = 10, rotation = -45)
    for child in root.children:
        drawt(ax,child,rawVersion,circle,J,U,max_x,max_y) ###MARK###

def drawconn(ax,root,rawVersion,circle,J,max_x,max_y):
    rootx=root.x
    if rawVersion==True:
        rooty=root.y
    else:
        rooty=J[str(root.tree)]
    if circle == True:
        rootx,rooty=angleCo(rootx,rooty,max_x,max_y)
    for child in root.children: 
        childx=child.x
        if rawVersion==True:
            childy=child.y
        else:
            childy=J[str(child.tree)]
        if circle == True:
            childx,childy=angleCo(childx,childy,max_x,max_y)
        '''
        plt.plot([rootx, childx],[rooty,childy],linestyle='-',
                 linewidth=0.1,color='grey',alpha=0.2,zorder=1)
        '''
        ax.annotate('',
            xy=(childx, childy), xycoords='data',
            xytext=(rootx, rooty), textcoords='data',
            size=5, va="center", ha="center",zorder=1,
            arrowprops=dict(arrowstyle="-|>",
                            connectionstyle="arc3,rad=-0.2",fc='white',
                            ec='white',alpha=0.5), 
            )
        drawconn(ax,child,rawVersion,circle,J,max_x,max_y)
 
def plotTree(G,ax):
    H,T,L,S=WebtoTree(G)
    V={str(k):map(str,v) for k,v in S.items()}
    treeDic=generateTree(V)
    tree=treeDic['source']
    d = buchheim(tree)
    J={str(k):v for k,v in L.items()}
    U={str(k):v for k,v in T.items()}
    max_y=depth(d,max(L, key=L.get))
    max_x=width(d)
    ax.set_axis_bgcolor('#1f2838')
    drawconn(ax,d,False,False,J,max_x,max_y)
    drawt(ax,d,False,False,J,U,max_x,max_y)


def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):
    # https://www.udacity.com/wiki/creating-network-graphs-with-python
    # graph is a list, each element contains two nodes

    # create networkx graph
    G=nx.DiGraph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.axis('off')
    plt.show()

######################
# drawing network
#######################

def drawDemoNetwork():
    gd = constructFlowNetwork(attention_data)
    # drawing a demo network
    fig = plt.figure(figsize=(12, 8),facecolor='white')
    pos={0: np.array([ 0.2 ,  0.8]),
     2: np.array([ 0.2,  0.2]),
     1: np.array([ 0.4,  0.6]),
     6: np.array([ 0.4,  0.4]),
     4: np.array([ 0.7,  0.8]),
     5: np.array([ 0.7,  0.5]),
     3: np.array([ 0.7,  0.2 ]),
     'sink': np.array([ 1,  0.5]),
     'source': np.array([ 0,  0.5])}
    width=[float(d['weight']*1.2) for (u,v,d) in gd.edges(data=True)]
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in gd.edges(data=True)])
    nx.draw_networkx_edge_labels(gd,pos,edge_labels=edge_labels, font_size = 15, alpha = .5)
    nx.draw(gd, pos, node_size = 3000, node_color = 'orange',
            alpha = 0.2, width = width, edge_color='orange',style='solid')
    nx.draw_networkx_labels(gd,pos,font_size=18)
    plt.show()

def drawFlowNetwork(G):
    plt.figure(figsize=(16,8))
    pos = nx.circular_layout(G)
    for node in G.nodes():
        if node == 'source':
            pos[node] = np.array([-1, 0])
        elif node == 'sink':
            pos[node] = np.array([1, 0])
        else:
            if pos[node][0] == -1:
                pos[node][0] += 0.5

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, nodelist=None, node_size=10, node_color='g', node_shape='o', alpha=1.0, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, label=None)
    nx.draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=1.0, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=True, label=None)
    nx.draw_networkx_labels(G, pos, labels=None, fontproperties=fontprop, font_size=20, font_color='k', font_family='Microsoft YaHei', font_weight='normal', alpha=1.0, ax=None)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k',
                                 font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None, rotate=True)
    plt.axis('off')
    plt.show()
    
def alloRegressPlot(xdata,ydata,col,mark,xlab,ylab, loglog):
    '''
    Plot the fitted allowmetric growth 
    '''
    x=np.log(xdata+1);y=np.log(ydata+1);
    xx = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,xx).fit()
    constant=res.params[0];beta=res.params[1]; r2=res.rsquared
    plt.plot(xdata,ydata,mark,color=col, label = None, alpha = 0.3)
    if loglog == True:
        plt.xscale('log');plt.yscale('log')
    plt.xlabel(xlab, fontsize = 20);plt.ylabel(ylab, fontsize = 20)
    minx,maxx=plt.xlim(); miny,maxy=plt.ylim()
    xs = np.linspace(min(xdata),max(xdata),100)
    plt.plot(xs,np.exp(constant)*xs**beta,color='r',linestyle='-', 
            label = '$\\alpha$ = '+ str(np.round(beta,2)) + ' , ' \
         + '$R^2$ = ' + str(np.round(r2,2)))
    plt.legend(loc = 0, frameon = False)
    
# fig = plt.figure(figsize=(12, 8),facecolor='white')
# ax = fig.add_subplot(2,2,1)
# alloRegressPlotLogBinning(totalflow,toSink,'g','o','$T_i$','$D_i$', True)
# ax = fig.add_subplot(2,2,2)
# alloRegressPlotLogBinning(fromSource,totalflow,'b','o','$S_i$','$T_i$', True)
# ax = fig.add_subplot(2,2,3)
# alloRegressPlotLogBinning(fromSource,toSink,'yellow','o','$S_i$','$D_i$', True)
# ax = fig.add_subplot(2,2,4)
# alloRegressPlotLogBinning(toflow,toSink,'orange','o','$F_i$','$D_i$', True)
# plt.tight_layout()

# fig = plt.figure(figsize=(12, 8),facecolor='white')
# ax = fig.add_subplot(2,2,1)
# alloRegressPlot(totalflow,toSink,'g','o','$T_i$','$D_i$', True)
# ax = fig.add_subplot(2,2,2)
# alloRegressPlot(fromSource,totalflow,'b','o','$S_i$','$T_i$', True)
# ax = fig.add_subplot(2,2,3)
# alloRegressPlot(fromSource,toSink,'yellow','o','$S_i$','$D_i$', True)
# ax = fig.add_subplot(2,2,4)
# alloRegressPlot(toflow,toSink,'orange','o','$F_i$','$D_i$', True)
# plt.tight_layout()
