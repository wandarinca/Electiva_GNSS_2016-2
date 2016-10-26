import numpy as np
import gpstk
import pandas as pd
from numba import autojit
from numpy.linalg import norm
import numpy as np
@autojit
def compute_distances(rc, svs):
    # return np.array( [np.sqrt((rc[0]-sv[0])**2 + (rc[1]-sv[1])**2) for sv in svs] )
    return np.linalg.norm(rc-svs, axis=1)

def compute_raim_ls(svs, svs_clocks, prs, max_iters=200, max_remove_sats=3):
    rx,rA,rb,rd = None, None, None, None
    min_rms_b = np.inf
    min_rms_b_idxs = None
    for nsats in range(np.max([4,len(prs)-max_remove_sats]),len(prs)+1):
        for i in combinations(range(len(o.prns)), nsats):
            i = np.array(i)
            x,A,b,d = compute_least_squares_position(np.array(o.prns_pos)[i],  
                                                     np.array(o.prns_clockbias)[i], 
                                                     np.array(o.P1)[i], max_iters)
            rms_b = norm(b)/np.sqrt(len(b))
            if rms_b < min_rms_b:
                min_rms_b = rms_b
                rx,rA,rb,rd = x,A,b,d
                min_rms_b_idxs = i
    return rx,rA,rb,rd


@autojit
def predict_pseudoranges(x, prns_pos, prns_clockbias):
    c = 299792458
    rhos    = compute_distances(x[:3], prns_pos)
    pranges = rhos + x[3]-c*prns_clockbias
    return rhos, pranges

@autojit
def apply_earth_rotation_to_svs_position(svs, prs):
    c = 299792458
    we = 7.2921159e-5
    rpos = np.zeros(svs.shape)
    pos = np.array(svs)
    for i in range(len(pos)):
        dt = prs[i]/c
        theta = we*dt
        R = np.array([[np.cos(theta), np.sin(theta),0.],[-np.sin(theta), np.cos(theta),0.],[0.,0.,1.]])
        rpos[i] = R.dot(pos[i])
    svs = np.array(rpos)
    return svs

def compute_least_squares_position(svs, svs_clocks, prs, max_iters=200, apply_earth_rotation=True):
    #svs xyz sats
    #prs pseudorangos observados
    if apply_earth_rotation:
        svs = apply_earth_rotation_to_svs_position(svs, prs)
    
    if len(svs)==0 or len(prs)==0:
        return np.array([0.,0.,0.,0.]),None, None, None

    ri = np.array([0.,0.,0.,0.]) #posicion computada

    #for i in range(max_iters):
    delta,i = 1,0
    while (norm(delta)>1e-8 and i<max_iters):
        rhos, pranges = predict_pseudoranges(ri, svs, svs_clocks)
        b = prs - pranges
        A = np.hstack(((ri[:3]-svs)/rhos[:,None],np.ones((len(b), 1))))
        delta =  np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(b)
        ri += delta
        i+=1
    return ri, A, b, delta

def compute_least_squares_position_ignore_clock(svs, prs, max_iters=200, apply_earth_rotation=True):
    if apply_earth_rotation:
        svs = apply_earth_rotation_to_svs_position(svs, prs)

    if len(svs)==0 or len(prs)==0:
        return np.array([0,0,0])

    ri = np.array([0,0,0]).astype(float)
    for i in range(max_iters):
        oldri = ri.copy()
        p_computed = compute_distances(ri, svs)
        b = prs - p_computed
        A = (ri-svs)/p_computed[:,None]
        delta =  np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(b)
        ri += delta
        if np.linalg.norm(delta)<1e-8:
            break
    return ri,A,b,delta


def compute_raim_position(gps_week, gps_sow, prns, prns_pos, pranges,  bcestore):
    if len(prns)==0 or len(prns_pos)==0:
        return np.array([0,0,0])
    t = gpstk.GPSWeekSecond(gps_week, gps_sow).toCommonTime()
    prnList = [gpstk.SatID(int(i[3:])) for i in prns]
    satVector = gpstk.seqToVector(list(prnList), outtype='vector_SatID')
    rangeVector = gpstk.seqToVector([float(i) for i in pranges])
    noTropModel = gpstk.ZeroTropModel()
    raimSolver = gpstk.PRSolution2()
    raimSolver.RAIMCompute(t, satVector, rangeVector, bcestore, noTropModel)   
    r = np.array([raimSolver.Solution[0], raimSolver.Solution[1], raimSolver.Solution[2]])
    return r

def rinex_to_dataframe(obsfile, navfile):
    c = 299792458.

    #observation_types=["P1", "P2", "L1", "L2"]
    observation_types=["C1", "P2", "L1", "L2"]
    obsHeader, obsData = gpstk.readRinex3Obs(obsfile)
    navHeader, navData = gpstk.readRinex3Nav(navfile)
    # setup ephemeris store to look for satellite positions
    bcestore = gpstk.GPSEphemerisStore()
    for navDataObj in navData:
        ephem = navDataObj.toGPSEphemeris()
        bcestore.addEphemeris(ephem)
    bcestore.SearchNear()
    navData.close()

    rec_pos = [obsHeader.antennaPosition[0], obsHeader.antennaPosition[1], obsHeader.antennaPosition[2]]
    requested_obstypes = observation_types
    obsidxs = []
    obstypes = []
    obsdefs = np.array([i for i in obsHeader.R2ObsTypes])
    for i in requested_obstypes:
        w = np.where(obsdefs==i)[0]
        if len(w)!=0:
            obsidxs.append(w[0])
            obstypes.append(i)
        else:
            print ("WARNING! observation `"+i+"` no present in file")
    obsidxs, obstypes
    print obsdefs #que tipos hay

    r = []
    for obsObject in obsData:
        prnlist = []
        obsdict = {}
        prnspos = []
        prns_clockbias = []
        prns_relcorr = []
        prnselev = []
        prnsaz   = []
        for i in obstypes:
            obsdict[i]=[]

        gpsTime = gpstk.GPSWeekSecond(obsObject.time)

        for satID, datumList in obsObject.obs.iteritems():
            if satID.system == satID.systemGPS:
                prnlist.append("".join(str(satID).split()))
                try:
                    eph   = bcestore.findEphemeris(satID, obsObject.time)
                except:
                    print "no encontrada!"

                for i in range(len(obsidxs)):
                    obsdict[obstypes[i]].append(obsObject.getObs(satID, obsidxs[i]).data)
                
                P1 = obsObject.getObs(satID, obsidxs[0]).data
                svTime = obsObject.time - P1/c
                svXvt = eph.svXvt(svTime)
                svTime += - svXvt.getClockBias() + svXvt.getRelativityCorr()
                svXvt = eph.svXvt(svTime)
                
                prnspos.append([svXvt.x[0], svXvt.x[1], svXvt.x[2]])
                prns_clockbias.append(svXvt.getClockBias())
                prns_relcorr.append(svXvt.getRelativityCorr())

                prnselev.append(obsHeader.antennaPosition.elvAngle(svXvt.getPos()))
                prnsaz.append(obsHeader.antennaPosition.azAngle(svXvt.getPos()))
                
                correct_sod = np.round(gpstk.YDSTime(obsObject.time).sod/30.0)*30 #mod 30s
                
        r.append([gpsTime.getWeek(), gpsTime.getSOW(), correct_sod, np.array(prnlist), np.array(prnspos), np.array(prns_clockbias), np.array(prns_relcorr), np.array(prnselev), np.array(prnsaz)] + [np.array(obsdict[i]) for i in obstypes])

    names=["gps_week", "gps_sow","tod", "prns", "prns_pos", "prns_clockbias", "prns_relcorr", "prns_elev", "prns_az"] + obstypes
    r = pd.DataFrame(r, columns=names)
    obsData.close()
    return r, bcestore, np.array(rec_pos)


# translated from  matlab code
def Delta_Rho_Compute(Rhoc, SV_Pos, Rcv_Pos, b):
    m,n = SV_Pos.shape
    Delta_Rho = np.zeros(m)
    for i in range(m):
        Rho0 = np.linalg.norm(SV_Pos[i]-Rcv_Pos)+b
        Delta_Rho[i] = Rhoc[i] - Rho0
    return Delta_Rho

def G_Compute(SV_Pos, Rcv_Pos):
    m,n = SV_Pos.shape
    dX = SV_Pos - Rcv_Pos
    Nor = np.sqrt(np.sum(dX**2,axis=1)).reshape(-1,1)
    Unit_Mtrix = dX/Nor
    G = np.hstack( (-Unit_Mtrix, np.ones((len(Unit_Mtrix),1))))
    return G

def Rcv_Pos_Compute(SV_Pos, SV_Rho):
    Num_Of_SV=len(SV_Pos)
    if Num_Of_SV<4:
        return np.array([0,0,0]), 0
    Rcv_Pos, Rcv_b =np.array([0,0,0]), 0
    B1=1
    END_LOOP=100
    count=0
    while (END_LOOP > B1):
        G = G_Compute(SV_Pos, Rcv_Pos);
        Delta_Rho = Delta_Rho_Compute(SV_Rho, SV_Pos, Rcv_Pos, Rcv_b);
        Delta_X = np.linalg.pinv(G.T.dot(G)).dot(G.T).dot(Delta_Rho)
        Rcv_Pos = (Rcv_Pos.T + Delta_X[:3]).T
        Rcv_b = Rcv_b + Delta_X[3];
        END_LOOP = (Delta_X[0]**2 + Delta_X[1]**2 + Delta_X[2]**2)**.5;
        count = count+1;
        if count>10:
            END_LOOP=B1/2;
    return Rcv_Pos, Rcv_b  

"""
import pyproj
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
def lla2ecef(lat,lon,alt, isradians=True):
    return pyproj.transform(lla, ecef, lon, lat, alt, radians=isradians)

def ecef2lla(X,Y,Z, isradians=True):
    lon, lat, alt = pyproj.transform(ecef, lla, X,Y,Z, radians=isradians)
    return lat, lon, alt

def get_dop(o, sigma=5):
    x,A,b,d = compute_least_squares_position(o.prns_pos, o.prns_clockbias, o.P1)
    return get_dop_raw(x,A,b,d,sigma)

def get_dop_raw(x,A,b,d,sigma=5):

    Cs = sigma*np.eye(len(o.P1))
    Cx = sigma**2 * np.linalg.pinv(A.T.dot(A))
    lat, lon, alt = ecef2lla(x[0], x[1], x[2])
    G = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                  [-np.sin(lon),             -np.cos(lon),             0 ],
                  [np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])
    Cl = G.dot(Cx[:3,:3]).dot(G.T)
    VDOP = Cl[2,2]
    HDOP = np.sqrt(Cl[0,0]**2+Cl[1,1]**2)
    return VDOP, HDOP"""
