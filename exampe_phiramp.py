import argparse
import numpy as np
import hoomd
from hoomd import md
import sys
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
    return np.power(0.5*N*(4/3)*np.pi*((5/12)**3 + (7/12)**3)/phi,1/3) 

L = 20
N = 2*L*L*L
#system = hoomd.init.create_lattice(unitcell=hoomd.lattice.bcc(a=1.0), n=L);
L=phi_to_L(phi=0.7,N=N)
snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(Lx=L,Ly=L,Lz=L), particle_types=['A', 'B'])
snapshot.particles.position[:] = np.random.uniform(low=-L/2,high=L/2,size=(N,3))
snapshot.particles.typeid[:int(N/2)] = 0
snapshot.particles.typeid[int(N/2):] = 1
system = hoomd.init.read_snapshot(snapshot)
nlist = md.nlist.cell()
module_MD_potentials.set_interaction_potential_harmonic(nlist)

dt=0.001
quench_T = 1e-5 
quench_phi = 0.7
ramp_rate = 1e-5
quench_steps = 10000000
analyzer_period = 10000
tauT = 1.0
print(phi_to_L(phi=0.7), phi_to_L(phi=0.63))
#until I write randomizing code...
inf_T = 10
inf_steps = 100000

analyzerManyVariables_header     = "# we used (at least initially) dt="+str(dt)+"   See _notes.txt for more comments. "
analyzerManyVariables_quantities = ['temperature', 'pressure', 'potential_energy', 'kinetic_energy', 'momentum', 'volume'] #, 'translational_kinetic_energy', 'rotational_kinetic_energy'] #, 'volume', 'num_particles']
analyzer_NVT_filename = "/home1/ridout/states/MD/harm/ramplog.dat"
analyzerManyVariables_NVT = hoomd.analyze.log(filename=analyzer_NVT_filename, \
        quantities=analyzerManyVariables_quantities, period=analyzer_period, \
        header_prefix = analyzerManyVariables_header+' we start by quenching at high phi, then ramp down phi' , overwrite=False, phase=0)



md.integrate.mode_standard(dt=dt)
hoomd.md.update.zero_momentum(int(1e9/N), phase=0)
np.random.seed(cmdargs.seed)
snap = system.take_snapshot()
vel = np.random.normal(0,inf_T**0.5, (N,3))
vel *= inf_T**0.5/np.std(vel)
snap.particles.velocity[:] = vel
system.restore_snapshot(snap)

hoomd.update.box_resize(L=phi_to_L(phi=0.7),period=None)
#run with a Nose-Hoover thermostat at effectively infinite T
integrator_nvt = md.integrate.nvt(group=hoomd.group.all(), kT=inf_T, tau=tauT)
hoomd.run(inf_steps )#, limit_hours=limit_hours, limit_multiple=limit_multiple)

#now quench
snap = system.take_snapshot()
vel = np.random.normal(0,quench_T**0.5, (N,3))
vel *= quench_T**0.5/np.std(vel)
snap.particles.velocity[:] = vel
system.restore_snapshot(snap)
integrator_nvt.disable()
integrator_nvt = md.integrate.nvt(group=hoomd.group.all(), kT=quench_T, tau=tauT)
hoomd.run(quench_steps)
#now adjust box volume
drop=0.6
compress_steps= drop / (dt * ramp_rate) 
updater = hoomd.update.box_resize(L = hoomd.variant.linear_interp([(0, phi_to_L(phi=0.7,N=N)), (compress_steps, phi_to_L(phi=0.7-drop,N=N))]),period=1000)
hoomd.run(compress_steps)
updater.disable()
hoomd.run(quench_steps)

