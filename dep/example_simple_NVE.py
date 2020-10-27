import hoomd
from hoomd import md
import sys, os
import json
import argparse
import module_MD_potentials
import numpy as np

'''
Given a state prepared in NVT at some T_0, launch NVE with randomized velocities drawn from some T (which may or may not equal T_0).
Sean Ridout, March 2019
Based on a script by François Landes.
'''

# Ian - this is originally Sean's code

#PARSE COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description="Run NVE dynamics from the NVT state, using freshly randomized velocities.")
parser.add_argument('-d', '--basedir', help="Directory in which the NVT folders seed=1, seed=2 etc. are found.")
parser.add_argument('-s', '--seed', help="Trajectory number / seed.",type=int)
parser.add_argument('-g', '--gpu', help="GPU device ID (omit flag for cpu)",default="-1")
parser.add_argument('-i', '--inherent', help="Record inherent states", default="false", action="store_true")
cmdargs = parser.parse_args()
#load other simulation parameters
basedir = cmdargs.basedir
with open(basedir+"/simulation_params.json", "r") as f:
  sp = json.load(f)
rec_period = sp["NVE_rec_period"]
analyzer_period = sp["analyzer_period"]
restart_period = sp["restart_period"]
if cmdargs.inherent:
    is_period = sp["IS_rec_period"]
Natoms = int(sp["N"])
seeddir = basedir+"/seed="+str(cmdargs.seed)
seed = cmdargs.seed
vel_seed = cmdargs.seed
kT = sp["T"]

nvedir = seeddir + "/NVE/"
os.makedirs(nvedir,exist_ok=True)
#are we using GPU or CPU?
if cmdargs.gpu == "-1":
    hoomd.context.initialize("")
elif int(cmdargs.gpu) >= 0:
    hoomd.context.initialize(" --mode=gpu  --gpu="+cmdargs.gpu)
else:
    print("Bad GPU number...")
    raise SystemExit

#this is kind of nonsense, since we don't want to set the time to zero if we are restarting...for now just never restart
system = hoomd.init.read_gsd(seeddir+"/restartNVT.gsd", seeddir+"/NVE/restart.gsd", time_step=0)

dt = 0.0025
limit_hours = 24*7-1

################################################################################
########## Set up the interactions #############################################
NeighborsListLJ = md.nlist.cell()
if sp["interaction_type"] == "LJ":
    myLjPair = module_MD_potentials.set_interaction_potential_LJ(NeighborsListLJ)
else:
    print("Invalid interaction")
    raise SystemExit
################################################################################
############ Physical integrations #############################################
md.integrate.mode_standard(dt=dt)
integrator_nve = md.integrate.nve(group=hoomd.group.all())
gsd_trajectory      = hoomd.dump.gsd(filename=seeddir+"/NVE/traj.gsd", group=hoomd.group.all(), period=rec_period     , phase=rec_period     , overwrite=False)
mit_multiple=int(max(rec_period, restart_period))
analyzerManyVariables_header     = "# we used (at least initially) dt="+str(dt)+"   See _notes.txt for more comments. "
analyzerManyVariables_quantities = ['temperature', 'pressure', 'potential_energy', 'kinetic_energy', 'momentum'] #, 'translao
analyzer_NVE_filename = seeddir+"/NVE/analyzer.dat"
analyzerManyVariables_NVE = hoomd.analyze.log(filename=analyzer_NVE_filename, \
            quantities=analyzerManyVariables_quantities, period=analyzer_period, \
            header_prefix = analyzerManyVariables_header+' this is the NVE part, as said in the title' , overwrite=False, phase=0)
limit_multiple=int(max(rec_period, restart_period))
hoomd.md.update.zero_momentum(int(1e8//Natoms), phase=0)

for batch in range(sp["tstepsNVE"]//is_period):
    #run NVE
    hoomd.run(is_period)

    #now turn off the usual file-saving to run FIRE, and save a snapshot to recover after FIRE
    analyzerManyVariables_NVE.disable()
    gsd_trajectory.disable()
    snap = system.take_snapshot()

    #enable FIRE and run   
    fire=hoomd.md.integrate.mode_minimize_fire(dt=0.0025, alpha_start=0.99, ftol=1e-5, Etol=1e-10, wtol=1e-5)
    while not(fire.has_converged()):
        hoomd.run(100,quiet=True)
       
    #save inherent structure to file
    hoomd.dump.gsd(filename=seeddir+"/NVE/IS/traj.gsd", overwrite=False, period=None, group=hoomd.group.all(), phase=-1)

    #now disable FIRE and turn back on the normal shit

    system.restore_snapshot(snap)
    md.integrate.mode_standard(dt=dt)  
    analyzerManyVariables_NVE.enable()
    gsd_trajectory.enable()

analyzerManyVariables_NVE.disable()
integrator_nve.disable()
