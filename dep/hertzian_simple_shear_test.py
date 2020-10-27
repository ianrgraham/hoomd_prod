import argparse
import numpy as np
import hoomd
from hoomd import md
import sys
import time
# sys.path.append("/home1/ridout/md-scripts")
import module_MD_potentials

parser = argparse.ArgumentParser(description="Quench to low T at high phi. Then, ramp down phi.")
parser.add_argument('-g','--gpu',help="GPU device ID (omit flag for cpu)",default="-1")
parser.add_argument('-s','--seed',help="Seed.", type=int, default=0)
cmdargs = parser.parse_args()

if cmdargs.gpu == "-1":
    hoomd.context.initialize("")
elif int(cmdargs.gpu) >= 0:
    hoomd.context.initialize(" --mode=gpu  --gpu="+cmdargs.gpu)
else:
    print("Bad GPU number...")
    raise SystemExit


def phi_to_L(phi=0.7,N=16000):
    return np.power(0.5*N*np.pi*((5/12)**2 + (7/12)**2)/phi,1/2) 

N = 10000
#system = hoomd.init.create_lattice(unitcell=hoomd.lattice.bcc(a=1.0), n=L);
L=phi_to_L(phi=0.7,N=N)
snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L,dimensions=2), particle_types=['A', 'B'])
np.random.seed(cmdargs.seed)
snapshot.particles.position[:] = np.random.uniform(low=-L/2,high=L/2,size=(N,3))
snapshot.particles.position[:,2] = 0.0
snapshot.particles.typeid[:int(N/2)] = 0
snapshot.particles.typeid[int(N/2):] = 1
snapshot.particles.velocity[:] = np.zeros((N, 3))
system = hoomd.init.read_snapshot(snapshot)
nlist = md.nlist.cell()
module_MD_potentials.set_interaction_potential_harmonic(nlist) # or hertzian

#until I write randomizing code...
inf_T = 10
zero_T = 0
inf_steps = 100000

dt=0.0025
'''
analyzerManyVariables_header     = "# we used (at least initially) dt="+str(dt)+"   See _notes.txt for more comments. "
analyzerManyVariables_quantities = ['temperature', 'pressure', 'potential_energy', 'kinetic_energy', 'momentum', 'volume'] #, 'translational_kinetic_energy', 'rotational_kinetic_energy'] #, 'volume', 'num_particles']
analyzer_NVT_filename = "/home1/igraham/shared-igraham/states/MD/harm/ramplog.dat"
analyzerManyVariables_NVT = hoomd.analyze.log(filename=analyzer_NVT_filename, \
        quantities=analyzerManyVariables_quantities, period=analyzer_period, \
        header_prefix = analyzerManyVariables_header+' we start by quenching at high phi, then ramp down phi' , overwrite=False, phase=0)
'''
tt = np.linspace(0.0, 0.00001, 2)
#traj = np.concatenate([tt, tt[1:-1][::-1],tt, tt[1:-1][::-1],tt, tt[1:-1][::-1],tt, tt[1:-1][::-1]])
traj = np.tile(tt, 100)
#md.integrate.mode_standard(dt=dt)
fire = hoomd.md.integrate.mode_minimize_fire(dt=0.0005, alpha_start=0.99, ftol=1e-10, Etol=1e-10)
nve = hoomd.md.integrate.nve(group=hoomd.group.all())
while not(fire.has_converged()):
    hoomd.run(1000,quiet=True)
    #print(i*tsteps)
    #i += 1
for strain in traj:
    
    updater = hoomd.update.box_resize(xy = strain, period=None)
    fire.reset()
    #hoomd.md.update.zero_momentum()
    i = 0
    tsteps = 100
    #hoomd.run(tsteps,quiet=True)
    begin = time.time()
    while not(fire.has_converged()):
        hoomd.run(tsteps,quiet=True)
        print(i*tsteps)
        i += 1
    print(fire.get_energy())
    print(strain, time.time() - begin)


