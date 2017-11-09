This folder contains the files used for TPS.

main.txt : lammps input script (parent code) for running TPS. Passed to lammps to execute the code.

run_traj.txt : lammps input script that runs the trajectory.

set_velocity.py & set_vel_subproc.py : python files that set the velocities of the atoms and rigid 
water molecules (corresponding to the correct linear and angular Boltzmann distributions)

0nacl_1800wat_*.xyz : sample initial configuration (.xyz files) for starting TPS