import numpy as np

def nearestLink(Pi,p0):
    if len(np.shape(p0))==1:
        p0 = np.reshape(p0,[1,len(p0)])
    Pr = Pi[1:,:]-Pi[:-1,:]
    P  = np.ones([len(Pr),1])@p0-Pi[:-1,:]
    a = sum((Pr*P).T)
    b = sum((Pr*Pr).T)
    klist = np.divide(a,b, out = np.zeros_like(a), where=b!=0)
    klim = np.clip(klist,0,1)
    Pn = np.diag(klim) @ Pr
    PnP = Pn - P
    l2 = sum((PnP*PnP).T)
    idx = np.argmin(l2)
    pout = Pn[idx,:]+Pi[idx,:]
    return pout, idx

if __name__ == "__main__":
    poslist = np.array([[1,1],[1,2],[2,2]])
    pos = np.array([[1.1,2.1]])
    print(nearestLink(poslist,pos))
