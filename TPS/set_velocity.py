
import random
import numpy as np
import set_vel_subproc
import sys



# order of arguments
# 0 - name of program (set_velocity.py)
# 1 - filename to read coordinates from
# 2 - shoot_new_old
# 3 - number of dimensions
# 4 - temp
# 5 - x box length
# 6 - y box length
# 7 - z box length
# 8 - half x box length
# 9 - half y box length
# 10 - half z box length
# 11 - number of milestones in the dump file		;; changed to be the total number of timesteps in the dump file
# 12 - trajname
# 13 - delta_tm_steps_shoot
# 14 - trajnumber

nghosts = 17
filename = str(sys.argv[1])
new_shoot_dump = int(sys.argv[2])			## 0 if shoot from old ;; 1 if shoot from new
D = int(sys.argv[3])
temp = float(sys.argv[4])
xyzbox = [float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7])]
halfxyzbox = [float(sys.argv[8]),float(sys.argv[9]),float(sys.argv[10])]
milestones_in_file = int(sys.argv[11])
trajname = str(sys.argv[12])
delta_tm_steps = float(sys.argv[13])
trajnumber	= int(sys.argv[14])
box_mass_filename = trajname + '_boxmass.dat'
bond_angle_filename = trajname + '_bond_angle.dat'


natoms = 7108
rigid_type_ind = [3]  					## atom id that indicates start of rigid body (LAMMPS requires it go O,H,H)
											## THIS PROGRAM ASSUMES ATOMS IN RIGID BODY ARE SEQUENTIAL!!
rigid_Nres = [3] 						## number of atoms in water rigid body 
n_rigid_types = len(rigid_type_ind)		## number of different type of rigid bodies
atommass = [22.99, 35.453, 15.9994, 1.008, 22.99, 35.453]					## mass of each atom in order of atom type
			## Na_x, Cl_x, Na_sol, Cl_sol, Ow, Hw, Na_kink, Cl_kink
beta2 = 1.0/(8.3144621*temp)			## mol/J	
beta3 = beta2 * 10**7					## for rotation part; in units of (mol fs^2)/(g A^2)





## open the file and read in the correct coordinates
shoot_config, types_list = set_vel_subproc.readLAMMPSxyz_shootconfig(filename,D,natoms,new_shoot_dump,milestones_in_file,delta_tm_steps,nghosts)

shoot_config_vel = np.copy(shoot_config)
## draw the velocities
vel_forw, shifted_crds = set_vel_subproc.draw_velocity(shoot_config_vel,D,beta2,beta3,atommass,natoms,rigid_type_ind,rigid_Nres,n_rigid_types,xyzbox,types_list,halfxyzbox)
vel_back = -1.0 * vel_forw

# read in default file and write the atom coordinates and velocities to it

# save velocity files for lammps to read in
set_vel_subproc.velxyz2lmps('input_data_forw_' + str(trajnumber) + '.dat',vel_forw,natoms,shoot_config,trajname,box_mass_filename,bond_angle_filename)
set_vel_subproc.velxyz2lmps('input_data_back_' + str(trajnumber) + '.dat',vel_back,natoms,shoot_config,trajname,box_mass_filename,bond_angle_filename)












