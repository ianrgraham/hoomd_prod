import argparse
import numpy as np
import hoomd
from hoomd import md
import sys
import time
import module_MD_potentials
import os

parser = argparse.ArgumentParser(description="Quench from infinite temperature to 0 above jamming. Shear until memory hopefully develops.")
parser.add_argument('-g','--gpu',help="GPU device ID (omit flag for cpu)",default="-1")
parser.add_argument('-s','--seed',help="Seed.", type=int, default=0)
parser.add_argument('-i','--iter',help="Iterations of the protocol.", type=int, default=50)
parser.add_argument('-n','--numParticles',help="Number of particles.", type=int, default=10000)
parser.add_argument('-m','--maxStrain',help="Maximum strain.", type=float, default=1e-2)
parser.add_argument('-y','--strainStep',help="Step size of strain.", type=float, default=1e-4)
parser.add_argument('-o','--overwrite',help="Overwrite any existing files (default False).", action='store_true')
parser.add_argument('-f','--fileSuffix',help="File suffix.", type=str, default="")
parser.add_argument('-p','--pressure',help="Pressure.", type=float, default=1e-2)
#parser.add_argument('-f','--forceTolerance',help="Maximum strain.", type=float, default=1e-8)
#parser.add_argument('-e','--energyTolerance',help="Maximum strain.", type=float, default=1e-8)
cmdargs = parser.parse_args()

if cmdargs.gpu == "-1":
    hoomd.context.initialize("--mode=cpu")
elif int(cmdargs.gpu) >= 0:
    hoomd.context.initialize("--mode=gpu  --gpu="+cmdargs.gpu)
else:
    print("Bad GPU number...")
    raise SystemExit

N = cmdargs.numParticles
maxIter = cmdargs.iter
phi = .7
tphi = .9
pressure = cmdargs.pressure
min_steps = 5000

maxStrain = cmdargs.maxStrain
strainStep = cmdargs.strainStep
steps = int(np.rint(maxStrain/strainStep))
fileSuf = cmdargs.fileSuffix
if fileSuf != "":
    fileSuf += "_"
seeddir = f"/home1/igraham/Projects/hoomd_test/test_data/{fileSuf}constPressure{pressure}_N{N}_seed{cmdargs.seed}_maxStrain{maxStrain}_strainStep{strainStep}"
os.makedirs(seeddir,exist_ok=True)

def phi_to_L(phi=0.9,N=16000):
    return np.power(0.5*N*np.pi*((5/12)**2 + (7/12)**2)/phi,1/2) 

L=phi_to_L(phi=phi,N=N)
tL=phi_to_L(phi=tphi,N=N)
snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L,dimensions=2), particle_types=['A', 'B'])
np.random.seed(cmdargs.seed)
snapshot.particles.position[:,:2] = np.random.uniform(low=-L/2,high=L/2,size=(N,2))
snapshot.particles.velocity[:] = np.zeros_like(snapshot.particles.velocity[:])
snapshot.particles.typeid[:int(N/2)] = 0
snapshot.particles.typeid[int(N/2):] = 1
system = hoomd.init.read_snapshot(snapshot)
nlist = md.nlist.cell()
mypair = module_MD_potentials.set_interaction_potential_hertzian(nlist)

#until I write randomizing code...
coarse_dt=0.025
fine_dt=0.25

analyzerManyVariables_quantities = ['pressure',
        'potential_energy', 'volume', 'pressure_xy',
        'pressure_xx', 'pressure_yy', 'xy'] 

nve = hoomd.md.integrate.nve(group=hoomd.group.all())
#nph = hoomd.md.integrate.nph(group=hoomd.group.all(), P=pressure, tauP=1.0, gamma=0.1)
fire_coarse = hoomd.md.integrate.mode_minimize_fire(dt=coarse_dt, alpha_start=0.1, ftol=1e-7, Etol=1e-7)

i = 0

initial_steps = 0

vlist = [[0,L], [1000000,tL]]
#vlist.extend([[initial_steps+(steps+i*4*steps)*min_steps, maxStrain],
#print(vlist)
var = hoomd.variant.linear_interp(vlist)
updater = hoomd.update.box_resize(L = var, period=min_steps, phase=0)
coarse_steps = 1000
# come up from jamming

log2 = hoomd.analyze.log(filename=seeddir+"/log_tmp.dat",
        quantities=analyzerManyVariables_quantities, period=min_steps, phase=min_steps-1)
while not(fire_coarse.has_converged()) or i*coarse_steps < 1000000:
    hoomd.run(coarse_steps,quiet=True)
    print(i)
    i += 1
i=0

nve.disable()

nph = hoomd.md.integrate.nph(group=hoomd.group.all(), P=pressure, tauP=1.0, gamma=0.1)

while not(fire_coarse.has_converged()) or np.abs(log2.query('pressure') - pressure)/pressure > 1e-3:
    print(log2.query('pressure'))
    hoomd.run(coarse_steps,quiet=True)
    print(i)
    if i > 10000:
        break
    i += 1
i=0

log2.disable()

vlist = [[initial_steps,0]]
for i in np.arange(maxIter):
    vlist.extend([[initial_steps+(steps+i*4*steps)*min_steps, maxStrain],
        [initial_steps+(3*steps+i*4*steps)*min_steps, -maxStrain],
        [initial_steps+(4*steps+i*4*steps)*min_steps, 0]])
print(vlist)

if os.path.exists(seeddir+"/log.dat") and not cmdargs.overwrite:
    print("File already exists! I don't want to do anything stupid!")
    sys.exit()

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
print(total_steps)
fire = hoomd.md.integrate.mode_minimize_fire(dt=fine_dt, alpha_start=0.1, ftol=1e-7, Etol=1e-7)
#for idx in range(total_steps):
idx = 0
while True:
    if (idx >= total_steps):
        break
    fire.reset()
    while not(fire.has_converged()):
        hoomd.run(min_steps,quiet=True)
        idx += 1
    if idx%10 ==0:
        print(total_steps, idx, log.query('xy'),  log.query('pressure_xy'))
        
