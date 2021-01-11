## module containg MD protocols that we want to use

import numpy as np
import hoomd
from numba import njit
from hoomd import md
import helper
import potentials
import os
import sys
#import pyvoro


def prepare_HTL(N, phi, seed, ensemble="NVE"):
    # High temperature liquid

    L = helper.phi_to_L_simplebi_2D(phi=phi,N=N)
    snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L,dimensions=2), particle_types=['A', 'B'])
    np.random.seed(seed)
    snapshot.particles.position[:,:2] = np.random.uniform(low=-L/2,high=L/2,size=(N,2))
    snapshot.particles.velocity[:] = np.zeros_like(snapshot.particles.velocity[:])
    snapshot.particles.typeid[:int(N/2)] = 0
    snapshot.particles.typeid[int(N/2):] = 1
    system = hoomd.init.read_snapshot(snapshot)
    nlist = md.nlist.cell()
    mypair = potentials.set_interaction_potential_hertzian(nlist)
    nve = hoomd.md.integrate.nve(group=hoomd.group.all())
    fire_coarse = hoomd.md.integrate.mode_minimize_fire(dt=0.1, alpha_start=0.1, ftol=1e-7, Etol=1e-7)
    i = 0
    while not(fire_coarse.has_converged()):
        hoomd.run(1000,quiet=True)
        print(i)
        i += 1
    nve.disable()
    del fire_coarse
    del nve
    snapshot2 = system.take_snapshot(all=True)
    return system, snapshot2

def prepare_ESL():
    # Equilibrated supercooled liquid
    pass

def prepare_GQ():
    # Gradual quench
    pass

def create_nlist(snapshot, N=None):
    pos = snapshot.particles.position[:,:2]
    box = snapshot.box.Lx # assume square geometry
    if N is None:
        N = int(np.ceil(box)) + 1

    assert N > 1
    be = np.linspace(-box/2, box/2, N)
    A = np.empty((N-1,N-1),dtype=object)
    for i,v in enumerate(A):
        for j,u in enumerate(v):
            A[i,j] = []
    for i, p in enumerate(pos[:,:2]):
        tmp = np.digitize(p, be)
        A[tmp[0]-1, tmp[1]-1].append(i)
    return A, np.diff(be[:2]), be

def pbc_dist(p1, p2, L):
    # assume the box is square and unstrained
    # p1 should not move, only p2
    c1 = np.array(p1 - p2 > L/2, dtype=np.float64)*L
    c2 = np.array(p2 - p1 > L/2, dtype=np.float64)*L
    p2 += c1
    p2 -= c2
    return p2 - p1 # return vector from p1 to p2

def yield_stress_routine(system, snapshot, N1, N2, potential=potentials.set_interaction_potential_hertzian, dump=None, dry=False):
    maxStrain = 3e-1
    strainStep = 1e-3
    steps = int(np.rint(maxStrain/strainStep))
    total_steps = steps + 1
    orig_pos = snapshot.particles.position[:,:2]

    print(snapshot)

    yss = []

    if dry:
        pass
        #system.

    #if dump is not None:
    rootdir = f"/home/ian/Documents/Projects/hoomd_prod/test/"
    analyzerManyVariables_quantities = ['pressure',
        'potential_energy', 'volume', 'pressure_xy',
        'pressure_xx', 'pressure_yy', 'xy'] 
    for alpha in np.linspace(0, np.pi/2, 19):
        context = hoomd.context.initialize("--mode=cpu")
        snapshot.particles.position[:,:2] = helper.rotate(orig_pos, alpha)
        system_tmp = hoomd.init.read_snapshot(snapshot)
        nlist = md.nlist.cell()
        mypair = potential(nlist)
        group_fire = hoomd.group.tags(0, tag_max=N1-1)
        #group_all = hoomd.group.all()
        #nve = hoomd.md.integrate.nve(group=hoomd.group.all())
        initial_steps = 0
        vlist = [[initial_steps,0]]
        for i in np.arange(1):
            vlist.append([initial_steps+(steps+i*4*steps)*1000, maxStrain])
        print(vlist)
        hoomd.util.quiet_status()
        var = hoomd.variant.linear_interp(vlist)
        updater = hoomd.update.box_resize(xy = var, period=1000, phase=0)
        if dump is not None:
            log = hoomd.analyze.log(filename=None,
                    quantities=analyzerManyVariables_quantities, period=1000,
                    overwrite=True, phase=999)
            filedump = hoomd.dump.gsd(filename=rootdir+f"/traj_{dump}_{alpha:.4f}.gsd", 
                            overwrite=True, period=10000, 
                            group=hoomd.group.all(), phase=9999)
        else:
            log = hoomd.analyze.log(quantities=analyzerManyVariables_quantities, 
                    period=1000, overwrite=True, phase=999)
        
        fire = hoomd.md.integrate.mode_minimize_fire(dt=0.1, alpha_start=0.1, ftol=1e-7, Etol=1e-7, group=group_fire)
        idx = 0

        stress = []
        while True:
            if (idx >= total_steps):
                break
            fire.reset()
            while not(fire.has_converged()):
                hoomd.run(1000,quiet=True)
                idx += 1
            stress.append(log.query('pressure_xy'))
            if idx%10 ==0:
                print(total_steps, idx, log.query('xy'),  log.query('pressure_xy'))
        


        #nve.disable()
        del context
        yss.append([alpha, stress])
        #break
    return yss
    

    

def find_yield_stress_distribution(system, snapshot, R1=5, R2=7, potential=potentials.set_interaction_potential_hertzian, label="0"):
    # using original method by Falk
    assert R1 < R2
    # by default in the NVE ensemble
    #system = hoomd.init.read_snapshot(snapshot)
    L = snapshot.box.Lx # assume square
    nlist, l0, bins = create_nlist(snapshot)
    nbins = len(bins) - 1
    d1 = int(np.floor(R1/l0)) # TODO this might be wrong
    d2 = int(np.floor(R2/l0))

    yield_stresses = []

    assert len(bins) - 1 > 2*d2

    for i in np.arange(100, snapshot.particles.N):
        p = snapshot.particles.position[i,:2]
        v = np.digitize(p, bins)
        N1 = []
        P1 = []
        T1 = []
        N2 = []
        P2 = []
        T2 = []
        target = 0
        target_idx = 0
        print(v[0], v[1], nbins)
        for j in np.arange(-d2-1,d2+1):
            for k in np.arange(-d2-1,d2+1):
                for i2 in nlist[(v[0]+j)%nbins, (v[1]+k)%nbins]:
                    p2 = snapshot.particles.position[i2,:2]
                    dp = pbc_dist(p, p2, L)
                    dist = np.linalg.norm(dp)
                    if dist < R2:
                        if dist < R1:
                            if dist < .5:
                                target = i2
                                target_idx = len(N1)
                            N1.append(i2)
                            P1.append(dp)
                            T1.append(snapshot.particles.typeid[i2])
                        else:
                            N2.append(i2)
                            P2.append(dp)
                            T2.append(snapshot.particles.typeid[i2])
        P1.extend(P2)
        T1.extend(T2)
        pos = np.array(P1)
        typ = np.array(T1)
        print(target, target_idx)
        print(len(N1), len(N2))
        print(type(pos), pos.shape)
        print(len(P1), len(P2))
        print(pos[target_idx])
        #pos -= np.mean(pos, axis=0)
        tsnap = hoomd.data.make_snapshot(N=len(N1)+len(N2), box=hoomd.data.boxdim(L=(R2+1)*2,dimensions=2), 
            particle_types=['A', 'B'])
        tsnap.particles.position[:,:2] = pos
        tsnap.particles.velocity[:] = np.zeros_like(tsnap.particles.velocity[:])
        tsnap.particles.typeid[:] = typ
        yield_stress = yield_stress_routine(system, tsnap, len(N1), len(N2), potential=potential, dump=f"{label}_{i}", dry=True)
        yield_stresses.append(yield_stress)
        #break

    return yield_stresses

        
        
                        
def find_yield_stress_top(snapshot, T1=1, T2=2):
    pass
        
def oscillatory_shear_inf_quench(gpu=-1, seed=0, maxIter=50, N=100, maxStrain=1e-2, strainStep=1e-4,
            overwrite=False, outdir="/home/ian/Documents/Projects/hoomd_prod/test/", filePrefix="", pressure=1e-2):

    if gpu == -1:
        hoomd.context.initialize("--mode=cpu")
    elif int(gpu) >= 0:
        hoomd.context.initialize("--mode=gpu  --gpu="+gpu)
    else:
        raise SystemExit("Bad GPU number...")
    
    phi = .7
    tphi = .9
    pressure = pressure
    min_steps = 1000

    maxStrain = maxStrain
    strainStep = strainStep
    steps = int(np.rint(maxStrain/strainStep))
    filePref = filePrefix
    if filePref != "":
        filePref += "_"
    if not os.path.isdir(outdir):
        raise SystemExit("Output directory does not exist! Exiting.")
    seeddir = f"{outdir}/{filePref}constPressure{pressure}_N{N}_seed{seed}_maxStrain{maxStrain}_strainStep{strainStep}"
    os.makedirs(seeddir,exist_ok=True)

    def phi_to_L(phi=0.9,N=16000):
        return np.power(0.5*N*np.pi*((5/12)**2 + (7/12)**2)/phi,1/2) 

    L=phi_to_L(phi=phi,N=N)
    tL=phi_to_L(phi=tphi,N=N)
    snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L,dimensions=2), particle_types=['A', 'B'])
    np.random.seed(seed)
    snapshot.particles.position[:,:2] = np.random.uniform(low=-L/2,high=L/2,size=(N,2))
    snapshot.particles.velocity[:] = np.zeros_like(snapshot.particles.velocity[:])
    snapshot.particles.typeid[:int(N/2)] = 0
    snapshot.particles.typeid[int(N/2):] = 1
    system = hoomd.init.read_snapshot(snapshot)
    nlist = md.nlist.cell()
    mypair = potentials.set_interaction_potential_hertzian(nlist)

    #until I write randomizing code...
    coarse_dt=0.25
    fine_dt=0.25

    analyzerManyVariables_quantities = ['pressure',
            'potential_energy', 'volume', 'pressure_xy',
            'pressure_xx', 'pressure_yy', 'xy'] 

    nve = hoomd.md.integrate.nve(group=hoomd.group.all())
    #nph = hoomd.md.integrate.nph(group=hoomd.group.all(), P=pressure, tauP=1.0, gamma=0.1)
    fire_coarse = hoomd.md.integrate.mode_minimize_fire(dt=fine_dt, alpha_start=0.1, ftol=1e-7, Etol=1e-7)

    i = 0

    initial_steps = 0
    coarse_steps = 1000
    tot_steps = coarse_steps*100
    vlist = [[0,L], [tot_steps,tL]]
    var = hoomd.variant.linear_interp(vlist)
    updater = hoomd.update.box_resize(L = var, period=coarse_steps, phase=0)
    # come up from jamming

    @njit
    def is_periodic(energy, tol):
        e1 = energy[-1]
        e2 = energy[-1]
        if np.abs(e1-e2)/(e1+e2) < tol:
            return True
        else:
            return False

    log2 = hoomd.analyze.log(filename=seeddir+"/log_tmp.dat",
            quantities=analyzerManyVariables_quantities, period=min_steps, phase=min_steps-1)
    while not(fire_coarse.has_converged()) or i*coarse_steps < tot_steps:
        fire_coarse.reset()
        hoomd.run(coarse_steps,quiet=True)
        print(i)
        i += 1
    i=0

    fire_coarse.reset()

    nve.disable()
    updater.disable()

    nph = hoomd.md.integrate.nph(group=hoomd.group.all(), P=pressure, tauP=1.0, gamma=0.1)
    fire_coarse = hoomd.md.integrate.mode_minimize_fire(dt=coarse_dt, alpha_start=0.1, ftol=1e-7, Etol=1e-7)
    while not(fire_coarse.has_converged()) or np.abs(log2.query('pressure') - pressure)/pressure > 1e-4:
        print(log2.query('pressure'))
        hoomd.run(coarse_steps,quiet=True)
        print(i)
        if i > 10000:
            break
        i += 1
    i=0

    energy = []
    energy.append(log2.query('potential_energy'))

    log2.disable()

    vlist = [[initial_steps,0]]
    for i in np.arange(maxIter):
        vlist.extend([[initial_steps+(steps+i*4*steps)*min_steps, maxStrain],
            [initial_steps+(3*steps+i*4*steps)*min_steps, -maxStrain],
            [initial_steps+(4*steps+i*4*steps)*min_steps, 0]])
    print(vlist)

    if os.path.exists(seeddir+"/log.dat") and not overwrite:
        raise SystemExit("Log file already exists! I don't want to do anything stupid!")

    hoomd.util.quiet_status()
    var = hoomd.variant.linear_interp(vlist)
    updater = hoomd.update.box_resize(xy = var, period=min_steps, phase=0)
    log = hoomd.analyze.log(filename=seeddir+"/log.dat",
            quantities=analyzerManyVariables_quantities, period=min_steps//10,
            overwrite=True, phase=min_steps//10-1)
    filedump = hoomd.dump.gsd(filename=seeddir+"/traj.gsd", 
                    overwrite=True, period=min_steps, 
                    group=hoomd.group.all(), phase=min_steps-1)

    total_steps = steps*4*maxIter+1
    period = steps*4
    print(total_steps)
    fire = hoomd.md.integrate.mode_minimize_fire(dt=fine_dt, alpha_start=0.1, ftol=1e-7, Etol=1e-7)
    #for idx in range(total_steps):
    idx = 0
    tol = 1e-5
    lastwasgood = False
    done = False
    while True:
        if (idx >= total_steps) or done:
            break
        if fire.has_converged():
            fire.reset()
        hoomd.run(min_steps,quiet=True)
        idx += 1
        #if idx%10 ==0:
        if idx%period == 0:
            energy.append(log.query('potential_energy'))
            if lastwasgood:
                if is_periodic(energy, tol):
                    done = True
                else:
                    lastwasgood = False
            elif is_periodic(energy, tol):
                lastwasgood = True
