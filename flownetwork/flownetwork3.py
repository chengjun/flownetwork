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

attention_data = [['a',0],['a',1],['a',2],\
                    ['b',1],['b',2],['c',1],\
                    ['c',2],['c',3],['d',2],\
                    ['d',3],['e',0],['e',4],\
                    ['f',0],['f',4],['g',0],\
                    ['g',4],['g',5],['h',0],\
                    ['h',5],['i',6]]

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
    
def flowBalancing(G):
    H = G.copy()
    O = G.out_degree(weight='weight')
    I = G.reverse().out_degree(weight='weight')
    for i in O:
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

def getAICI(H):
    '''
    return AI & CI
    Source: Wu and Zhang 2013 The decentralized flow structure of clickstreams on the web. EPJB
    '''
    # H = flowBalancing(G)
    F1=nx.to_numpy_matrix(H)
    sourcep=H.nodes().index('source')
    sinkp=H.nodes().index('sink')
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
