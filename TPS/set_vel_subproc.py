
import random
import numpy as np


	
def new_mitosis_inputfile(window_hi,window_low,mit_input,new_dir):
	mit_input[5,1] = window_low
	mit_input[6,1] = window_hi
	filename_mit_input = new_dir + 'inputs/mitosis_input.txt'
	np.savetxt(filename_mit_input,mit_input,fmt='%s',delimiter=',')
	
########################################################################################
##### TAKE SHOOTING POINT COORDINATES AND DRAWN VELOCITIES TO MAKE INPUT DATA FILE #####
########################################################################################
def velxyz2lmps(velocity_filename,velocities,natoms,xyzz,trajname,boxmass_filename,bond_angle_filename):
	#atoms = np.concatenate((template_mid,xyz),axis=1)
	filename = trajname + '_atinfo.dat'
	a = open(filename,'r')
	b = a.read()
	a.close()
	b = b.split()
	atom_default = np.array(b,dtype=float).reshape(-1,7)
	


	
	#vel_blank = np.array([[' ',' ',' '] for x in range(natoms)],dtype=str)
	velocities = np.round(velocities,decimals=8)
	velocities = velocities.astype('|S12')
	#vel_numb = [x for x in range(1,natoms+1)]
	#vel_numbers = np.int_(vel_numb).reshape(-1,1)
	#vel = np.concatenate((vel_numbers,velocities,vel_blank),axis=1)
	#blank = np.array([[' ',' ',' ',' ',' ',' ',' ']],dtype=str)
	#vel_header = np.array([['Velocities',' ',' ',' ',' ',' ',' ']],dtype=str)
	#atoms = np.concatenate((atom_default[:,0:4],xyz),axis=1)
	#atom_header = np.array([['Atoms',' ',' ',' ',' ',' ',' ']],dtype=str)
	#data4lmps = np.concatenate((blank,blank,atom_header,blank,atoms,blank,blank,vel_header,blank,vel),axis=0)
	#np.savetxt(velocity_filename,data4lmps,fmt='%s',delimiter=' ')
	
	boxmass_data = open(boxmass_filename,'r')
	c = boxmass_data.read()
	boxmass_data.close()
	
	bond_angle_data = open(bond_angle_filename,'r')
	d = bond_angle_data.read()
	bond_angle_data.close()
	
	input_data = open(velocity_filename,'w')
	input_data.write(c)
	
	input_data.write('\n\n\nAtoms\n\n')
	for i in range(natoms):
		input_data.write(str(atom_default[i,0]) + ' ' + str(atom_default[i,1]) + ' ' + str(atom_default[i,2]) + ' ' + str(atom_default[i,3]) + ' ' + str(xyzz[i,0]) + ' ' + str(xyzz[i,1]) + ' ' + str(xyzz[i,2]) + '\n')
	
	input_data.write('\n\n\nVelocities\n\n')
	for i in range(natoms):
		input_data.write(str(i+1) + ' ' + str(velocities[i,0]) + ' ' + str(velocities[i,1]) + ' ' + str(velocities[i,2]) + '\n')
	
	input_data.write('\n\n\n')
	input_data.write(d)
	
	input_data.close()
	
# ########################################################################################
	

#####################################################################
##### READ AN .XYZ FILE AND EXTRACT ATOM TYPE AND CONFIGURATION #####
#####################################################################
def readLAMMPSxyz_shootconfig(xyzfile,D,natoms,desired_milestone,milestones_in_file,delta_tm_steps,nghosts):
	# desired_milestone : the milestone number I want from the xyzfile 
	# milestones_in_file : number of milestones in the xyzfile ; NOT INCLUDING THE INITIAL CONFIG

	## read in the coordinate file 
	f = file(xyzfile,'r')
	coord = f.read()
	f.close()
	end_dump = str(natoms) + '\n' + 'Atoms.'
	
	
	## extract the coordinates from the desired dump 					
	header = 'Timestep: ' + str(desired_milestone) + '\n'

	i = coord.find(header,0)								## start searching from where last coordinate dump ended
	
	if desired_milestone != milestones_in_file:
		j = coord.find(end_dump,i)
		xyztmp = coord[(i+len(header)):j]
	else:
		end = len(coord)
		xyztmp = coord[(i+len(header)):end]

	## convert string of coordinates to an array which includes the atom type
	xyztmp = xyztmp.split()
	xyztmp = np.array(xyztmp,dtype=float).reshape(natoms,D+1)
	xyz = np.delete(xyztmp,0,1)

	#type_list = np.array(xyztmp[:,0],type=int)					## can just get this from the default input data file
	
	## convert first column to integers
	#print np.int_(xyz[:,0]).reshape(-1,1)
	#xyz[:,0] = np.int_(xyz[:,0]).reshape(-1,1)
	type_list = np.int_(xyztmp[:,0])#.reshape(-1,1)

	
	# now grab the updated ghost positions
	# ghost_file = 'ghost_' + xyzfile 
	# g = file(ghost_file,'r')
	# gcoord = g.read()
	# g.close()
	# aaa = int(desired_milestone/delta_tm_steps)
	# header = 'Timestep: ' + str(aaa) + '\n'
	# end_dump = str(nghosts) + '\n' + 'Atoms.'
	# i = gcoord.find(header)
	# if desired_milestone != milestones_in_file:
		# j = gcoord.find(end_dump,i)
		# gxyztmp = gcoord[(i+len(header)):j]
	# else:
		# end = len(gcoord)
		# gxyztmp = gcoord[(i+len(header)):end]
	# gxyztmp = gxyztmp.split()
	# gxyztmp = np.array(gxyztmp,dtype=float).reshape(-1,D+1)
	# gxyz = np.delete(gxyztmp,0,1)
	# ghost_start_index = len(xyz[:,0]) - len(gxyz[:,0])
	# xyz[ghost_start_index:,:] = gxyz
	
	
	return xyz, type_list
# #########################################################################################


#####################################################
##### GET INDICES FOR THE ENERGETIC INFORMATION #####
#####################################################
def thermo_info(thermo_output,Etot_label,PE_label,P_label):
	thermo_list = thermo_output.split()
	n_thermo_outputs = len(thermo_list)
	E_index = thermo_list.index(Etot_label)
	PE_index = thermo_list.index(PE_label)
	P_index = thermo_list.index(P_label)
	return n_thermo_outputs, E_index, PE_index, P_index
# ########################################################


###############################################################################
##### CHECK THAT KINKS WE DON'T "PULL" ON STAY THE SAME IN THE TRAJECTORY #####
###############################################################################
def check_other_kinks(q_nopull_kinked,q_nopull_neigh,n_m,maxq,minq):
	# maxq, minq : limits on how much OP for kinked ghosts can differ while still accepting move
	acc = True
	# i think i should really only need the max and min values of OP for kink and neighs
	
	## loop through all entries
	for i in range(n_m+1):
		if q_nopull_kinked[i,0] < minq or q_nopull_kinked[i,1] > maxq:			## if outside of the tolerance level
			acc = False
			break 
		elif q_nopull_neigh[i,0] != 0.0 or q_nopull_neigh[i,1] != 0.0:											## if an ion tries attaching to empty spots next to kinks
			acc = False 
			break 
	
	return acc
# ###############################################################################
	

###############################################################
##### READ THE ORDER PARAMETER VALUES FROM EACH MILESTONE #####
###############################################################
def read_q_file(q,qna,qcl,which_way,n_milestones,shoot_milestone_numb,trajnumber,qfile_header,qfile_footer,qfile_header_length,qfile_header_nopull,qfile_header_length_nopull,qfile_header_neigh,qfile_header_length_neigh):
	# n_milestones does not include the shooting point milestone
	# qfile_header : string to search for in the order parameter file that indicates beginning of output 
	# qfile_footer : string to search for that indicates end of output
	# q_nopull_kinked : array with the min and max OP value for other ghosts in kink sites 
	# q_nopull_neigh : array with the min and max OP value for other ghosts next to kink sites (neighs)

	## determine if for forward or backward trajectory
	if which_way == 'forward':
		add = True 
		opfilename = 'OPS_pull_' + str(trajnumber) + 'forw.txt'
		opfilename_nopull = 'OPS_nopull_' + str(trajnumber) + 'forw.txt'
		opfilename_neigh = 'OPS_neigh_' + str(trajnumber) + 'forw.txt'
	elif which_way == 'backward': 
		add = False
		opfilename = 'OPS_pull_' + str(trajnumber) + 'back.txt'
		opfilename_nopull = 'OPS_nopull_' + str(trajnumber) + 'back.txt'
		opfilename_neigh = 'OPS_neigh_' + str(trajnumber) + 'back.txt'
	
	## open the file-it will be in a .lammpstrj type of format
	opfile = open(opfilename,'r')
	s = opfile.read()
	opfile.close()
	opfile_nopull = open(opfilename_nopull,'r')
	t = opfile_nopull.read()
	opfile_nopull.close()
	opfile_neigh = open(opfilename_neigh,'r')
	u = opfile_neigh.read()
	opfile_neigh.close()
	
	qindex = shoot_milestone_numb 
	start = 0 
	stop = 0
	start_nopull = 0 
	stop_nopull = 0
	start_neigh = 0
	stop_neigh = 0
	acc = True
		
	## find the OP value in the file for each milestone 
	for i in range(n_milestones+1):
		start = s.find(qfile_header,stop)
		start_nopull = t.find(qfile_header_nopull,stop_nopull)
		start_neigh = u.find(qfile_header_neigh,stop_neigh)
		if i != n_milestones:
			stop = s.find(qfile_footer,start)
			stop_nopull = t.find(qfile_footer,start_nopull)
			stop_neigh = u.find(qfile_footer,start_neigh)
		else:
			stop = len(s)
			stop_nopull = len(t)
			stop_neigh = len(u)
		ops = s[(start+qfile_header_length):stop]
		ops = ops.split()
		ops_nopull = t[(start_nopull+qfile_header_length_nopull):stop_nopull]
		ops_nopull = ops_nopull.split()			
		ops_neigh = u[(start_neigh+qfile_header_length_neigh):stop_neigh]
		ops_neigh = ops_neigh.split()
		
		## put the data into lists 
		qna[qindex] = float(ops[0])
		qcl[qindex] = float(ops[1])
		q[qindex] = qna[qindex] + qcl[qindex]
		
		## check that other kink sites don't change
		for j in range(6):			## check kinked ions
			qnopull = float(ops_nopull[j])
			if qnopull == 0.0:						## if the ion has left the kink site : this is set in the cutoff for the compute
				acc = False								## going to reject trajectory since a kink ion tries leaving
				print 'Rejected b/c of ions in kinks. ',which_way,'. Milestone number ',i
				break
		if (False == acc):					## break out of going through milestones if will be rejecting
			break
			
		for j in range(8):			## check neighs
			qneigh = float(ops_neigh[j])
			if qneigh != 0.0:
				acc = False
				print 'Rejected b/c of neigh sites. ',which_way,'. Milestone number ',i
				break
		if (False == acc):
			break
	
		## adjust the milestone number accordingly
		if (add):
			qindex += 1 
		else:
			qindex -= 1 

	return q, qna, qcl, opfilename, opfilename_nopull, opfilename_neigh, acc
# ##############################################################


#####################################################################
##### TEST TO SEE OF AN ERROR OCCURRED DURING THE ATTEMPTED RUN #####
#####################################################################
def is_err_LAMMPSlog(log_filename):
	input = open(log_filename,'r')
	log = input.read()
	input.close()
	
	## check for an error indicator
	err = log.find('ERROR') 				## -1 if not in file 
	
	return err 
# #################################################################
	
	
############################################################
##### READ IN THE THERMODYNAMIC DATA FROM THE LOG FILE #####
############################################################
def LAMMPSlog(which_way,log_file_name,thermo_output,end_indicator,n_thermo_outputs,Etot_index,PE_index,P_index,eng,poteng,press,n_milestones,shoot_milestone_numb):
	## determine if for forward or backward trajectory
	if which_way == 'forward':
		add = True 
	elif which_way == 'backward': 
		add = False

	## read in the log file 
	input = open(log_file_name,'r')
	log = input.read()
	input.close()
	
	## find the region of thermo output, this first first region, which is what I want.
	i = log.find(thermo_output)
	j = log.find(end_indicator,i)

	## put the thermo data in array form 
	output = log[i+len(thermo_output):j]
	output = output.split()
	output = np.array(output,float).reshape(len(output)/n_thermo_outputs,n_thermo_outputs)
	
	## loop through and add data to lists of the thermo output
	for n in range(n_milestones+1):
		if (add):
			indx = shoot_milestone_numb + n 
		else:
			indx = shoot_milestone_numb - n 
		
		eng[indx] = output[n,Etot_index]
		poteng[indx] = output[n,PE_index]
		press[indx] = output[n,P_index]
	
	return eng, poteng, press
# ######################################################################
	
	
#################################################
##### READ IN VARIABLES FROM THE INPUT FILE #####
#################################################
def eps_input(input_file):
	from numpy import genfromtxt
	input = genfromtxt(input_file,dtype=None,delimiter=',')
	trajname=input[0,1]						## trajectory name
	inputtrajnumber=int(input[1,1])			## initial trajectory number (usually 0)
	trajnumberstart=int(input[2,1])			## first trajectory number to run 
	ntraj=int(input[3,1])					## number of trajectories to run 
	qlow=float(input[4,1])					## minimum order parameter value in this window 
	qhi=float(input[5,1])					## maximum order parameter value in this window 
	LAMMPSdir=input[6,1]					## directory in which lammps executable is located 
	n_atoms = int(input[7,1])				## number of atoms in the system
	n_bond = int(input[8,1])				## number of bonds in the system 
	n_ang = int(input[9,1])					## number of angles in the system 
	xlo = float(input[10,1])			
	xhi = float(input[11,1])
	ylo = float(input[12,1])
	yhi = float(input[13,1])
	zlo = float(input[14,1])
	zhi = float(input[15,1])
	return input,trajname,inputtrajnumber,trajnumberstart,ntraj,qlow,qhi,LAMMPSdir,n_atoms,n_bond,n_ang,xlo,xhi,ylo,yhi,zlo,zhi
# ########################################################################
	

#############################################################
##### READ IN DEFAULT INPUT SCRIPT AND SAVE TO VARIABLE #####
#############################################################
def LAMMPS_input_default(scriptname):
	## read in template file 
	template_file = open(scriptname,'r')		
	template_input_script = template_file.read()
	template_file.close()
	
	return template_input_script
# ###########################################################
	
	
#######################################################
##### MAKE THE INPUT SCRIPT FOR LAMMPS TRAJECTORY #####
#######################################################
def make_LAMMPS_input(forw_or_back,data_input_filename,trajname,trajnumber,temp,t_sim,dump_time,tstep,template_input_script):
	# forw_or_back is is string 'forw' or 'back' to indicate if for forward or backward trajectory 

	## define file names 
	input_file_name = 'input_' + str(trajnumber) + '_' + forw_or_back + '.txt'
	log_file_name = 'log_' + str(trajnumber) + '_' + forw_or_back + '.txt'
	output_xyz_name = trajname + '_' + str(trajnumber) + '_' + forw_or_back + '.xyz'
	prelim_xyz_name = trajname + '_' + str(trajnumber) + '_' + forw_or_back + '_prelim' + '.xyz'
	
	## draw random number for seed on langevin thermostatting
	seednumber = random.randint(1,99999)

	## write input script 
	input = open(input_file_name,'w')
	input.write('\n\nvariable     outxyz string ' + output_xyz_name + '\n')
	input.write('variable     input_file string ' + data_input_filename + '\n')
	input.write('variable     T equal ' + str(temp) + '\n')
	input.write('variable     seed equal ' + str(seednumber) + '\n')
	input.write('variable     tsim equal ' + str(int(t_sim)) + '\n')						## in number of timesteps
	input.write('variable     tstep equal ' + str(tstep) + '\n')
	input.write('variable     dump_steps equal ' + str(int(dump_time)) + '\n')
	input.write('variable     forw_back string ' + forw_or_back + '\n')
	input.write('variable     prelim_xyz string ' + prelim_xyz_name + '\n')
	input.write('variable     input_script string ' + input_file_name + '\n')
	input.write('variable     trajnumber string ' + str(trajnumber) + '\n\n')
	
	input.write(template_input_script)
	input.close()
	return input_file_name, log_file_name,output_xyz_name
# ###########################################################
	
	
################################################################
##### DRAW VELOCITIES FROM CORRECT BOLTZMANN DISTRIBUTIONS #####
################################################################
def draw_velocity(shoot_config,D,beta2,beta3,atommass,natoms,rigid_type_ind,rigid_Nres,n_rigid_types,xyzbox,types_list,halfxyzbox):
	# shoot_config : shooting point configuration; natom by 3 array
	# types_list : list of the atom types in the correct order with shoot_config
	# atommass : list of atom masses in order of atom type 
	# natoms : total number of atoms
	
	velocity = np.zeros((natoms,3))
	
	
	more_mols = True
	atom = 0
	rigid = -1
	while more_mols:
	
		## check if the next atom indicates the start of a rigid body 
		for i in range(n_rigid_types):
			if types_list[atom] == rigid_type_ind[i]:
				rigid += (i+1) 										## indicator of which rigid body type it is 
		
		## if not rigid, just do translational 
		if rigid == -1:
			type = types_list[atom] - 1
			velocity[atom,:] = set_trans_velocity(atommass[type],D,beta2)
			atom += 1			## up the atom counter
		else:
		## if rigid body, do translational and rotational 
		#	for i in range(n_rigid_types):
		#		if rigid == i:
			## make array of atom types and coords of rigid body 
			crds = shoot_config[atom:(atom+rigid_Nres[rigid]),:]
			types = types_list[atom:(atom+rigid_Nres[rigid])]
			
			## draw rotational velocity
			rotvel, bodymass, crds = set_rot_velocity(atommass,crds,beta3,rigid_Nres[rigid],D,xyzbox,types,halfxyzbox)
			shoot_config[atom:(atom+rigid_Nres[rigid]),:] = np.copy(crds)
			
			## draw translational velocity 
			transvel = set_trans_velocity(bodymass,D,beta2)
			
			## sum the velocity of each atom
			velocity[atom:(atom+rigid_Nres[rigid]),:] = (rotvel + transvel)
			
			## increase atom counter by number of atoms in rigid body
			atom += rigid_Nres[rigid]	
			
		rigid = -1
		if atom > (natoms-1):
			more_mols = False
	
	return velocity, shoot_config
# ########################################################################

	
########################################################################
##### DRAW RANDOM NUMBER TO SET TRANSLATIONAL VELOCITY OF MOLECULE #####
########################################################################
def set_trans_velocity(mass,D,beta2):
	## beta2 in mol/J units
	mass /= (1000.0)	# convert to kg/mol
	trans_stdev = np.sqrt(1/(mass*beta2))
	trans_vel = [0.00001 *random.gauss(0.0,trans_stdev) for x in range(D)]		## in A/fs b/c multiplied by 10**-5
	#print trans_vel
	## to convert to A/fs, multiply by 10**-5
	#trans_vel *= 0.00001 
	return trans_vel		
# ###################################################


#######################################################
##### SET ROTATIONAL VELOCITY OF A RIGID MOLECULE #####
#######################################################
def set_rot_velocity(atommass,crds,beta3,Nres,D,xyzbox,types,halfxyzbox):
	# atommass is list in order of atom type 
	# crds is Nres by 3 array  
	# beta3 is in proper units (mol fs^2)/(g A^2), will give velocities in A/fs
	
	## calculate center of mass and displacement of atoms from COM in xyz space 
	mass, com, deltaxyz, masses, crds = calc_com_deltaxyz(atommass,crds,Nres,D,halfxyzbox,types,xyzbox)
		# mass is the total mass of the molecule 
		# com is a 1 by D array of the center of mass 
		# deltaxyz is the displacement of each atom from the COM (Nres by D array)
		# masses is a list of the mass of each atom in order 
	
	## calculate moments of inertia, principal axes
	inertia_tensor, eigenvalues, eigenvectors = mom_of_inertia(deltaxyz,masses,Nres)
	
	## reverse third vector direction if needed to obtain right handed coordinate system
	third_eigenvector = np.cross(eigenvectors[:,0],eigenvectors[:,1])
	right_hand_eigenvectors = eigenvectors
	right_hand_eigenvectors[:,2] = third_eigenvector
	
	## calculate the conversion matrix to go from xyzspace to bodyspace
	xyz2uvw = np.transpose(right_hand_eigenvectors)
	#xyz2uvw = np.linalg.inv(right_hand_eigenvectors)	
	# test if this is just the transpose 
	#just_transpose = np.transpose(right_hand_eigenvectors)
	#print xyz2uvw - just_transpose
	
	## calculate coordinates in body space 
	delta_uvw = np.zeros((3,3))
	for j in range(Nres):
		delta_uvw[j,:] = np.dot(xyz2uvw,deltaxyz[j,:])	

	## construct vectors for rotations about u,v,w principal axes of rotation 
	rot_vectors = construct_rot_vectors(Nres,D,delta_uvw)
	
	## normalize distances for each atom in the rigid molecule 
		# I DON'T THINK I NEED THIS BECAUSE THIS JUST DIVIDES BY THE CORRECT ELEMENT OF radii
		# AND THEN I JUST MULTIPLY BY THAT SAME VALUE IN CALCULATING THE VELOCITY
	# final_rot_vectors = np.zeros_like(rot_vectors)
	# for j in range(Nres):
		# for k in range(D):
			# final_rot_vectors[Nres*j:(Nres*j+Nres),k] = rot_vectors[Nres*j:(Nres*j+Nres),k]/np.linalg.norm(rot_vectors[Nres*j:(Nres*j+Nres),k])
	
	## draw random numbers for the angular velocity and set velocity in body frame
	for j in range(3):
		rot_stdev = np.sqrt(1/(eigenvalues[j]*beta3))				# in units of 1/fs
		rand_num = random.gauss(0.0,rot_stdev)		
		rot_vectors[:,j] *= rand_num					# velocities now in A/fs 
		
	## convert velocities back to xyzspace 
	rotxyz_vectors = np.zeros_like(rot_vectors)
	for j in range(Nres):
		rotxyz_vectors[(D*j):(D*j+D),:] = np.dot(right_hand_eigenvectors,rot_vectors[(D*j):(D*j+D),:])
		
	## sum up contribution from each rotation to get net velocity
	net_rot_vel = np.sum(rotxyz_vectors,axis=1).reshape(Nres,D)		# sum across rows
		# net_rot_vel is Nres by D array where columns are x y z velocities caused by rotations 
		#				follows same order of Nres as originally did 
		
	return net_rot_vel, mass, crds
# ###################################################
	
	
#####################################################
##### CALCULATE COM AND DELTAXYZ FOR A MOLECULE #####
#####################################################
def calc_com_deltaxyz(atommass,crds,Nres,D,halfxyzbox,types,xyzbox):
	# atommass is list of masses in order of atom type
	# crds is Nres by 4 array with the first column corresponding to the atom type 
	# Nres is the number of atoms in the rigid body
	# halfxyzbox is the 
	
	## calculate the center of mass and mass of the molecule
	com = [0 for x in range(D)]
	masses = []
	mass = 0.0
	for i in range(Nres):
		type = types[i] - 1
		if i == 0:
			com += atommass[type] * crds[i,:]
		else:
			## calculate distance from atom 1 and shift in pbc if needed
			for j in range(D):
				delt = crds[i,j]-crds[0,j]
				if abs(delt) > halfxyzbox[j]:
					crds[i,j] -= (xyzbox[j]*int(delt/halfxyzbox[j]))
			## after shifting properly, add to COM calc
			com += atommass[type] * crds[i,:]	
		mass += atommass[type]
		masses.append(atommass[type])			## list of masses in correct order for further calculations
	com /= mass 								## com is now an array
	
	## calculate the x,y,z displacement of each atom from the COM
	deltaxyz = np.zeros((Nres,D))
	for i in range(Nres):
		deltaxyz[i,:] = crds[i,:] - com
		
	return mass,com,deltaxyz,masses,crds
# ######################################################


##########################################################
##### CALCULATE INERTIA TENSION, MOMENTS OF INERTIA, #####
##### AND PRINCIPAL AXES OF INERTIA					 #####
##########################################################
def mom_of_inertia(delta_xyz,masses,Nres):	
	inertia_tensor = np.zeros((3,3))
	Ixx = 0
	Iyy = 0
	Izz = 0
	Ixy = 0
	Iyz = 0
	Ixz = 0
	for j in range(Nres):
		Ixx += (delta_xyz[j,1]**2 + delta_xyz[j,2]**2) * masses[j]
		Iyy += (delta_xyz[j,0]**2 + delta_xyz[j,2]**2) * masses[j]
		Izz += (delta_xyz[j,0]**2 + delta_xyz[j,1]**2) * masses[j]
		Ixy -= delta_xyz[j,0] * delta_xyz[j,1] * masses[j]
		Iyz -= delta_xyz[j,1] * delta_xyz[j,2] * masses[j]
		Ixz -= delta_xyz[j,0] * delta_xyz[j,2] * masses[j]
	inertia_tensor[0,0] = Ixx
	inertia_tensor[1,1] = Iyy
	inertia_tensor[2,2] = Izz
	inertia_tensor[0,1] = Ixy
	inertia_tensor[1,0] = Ixy
	inertia_tensor[0,2] = Ixz
	inertia_tensor[2,0] = Ixz
	inertia_tensor[1,2] = Iyz
	inertia_tensor[2,1] = Iyz
	eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)	# eigenvectos is 3x3 matrix  with columns as eigenvectors
		## eigenvalues are the moments of inertia
		## eigenvectors are the principal axes of inertia 
	return inertia_tensor, eigenvalues, eigenvectors
# #####################################################################


##########################################################################
##### MAKE ROTATIONAL VECTORS. EACH VECTOR CORRESPONDS TO ROTATION   #####
##### ABOUT A DIFFERENT PRINCIPAL AXES. EVENTUALLY, WILL SUMM ACROSS #####
##### ROWS TO GET NET VELOCITY IN EACH DIRECTION					 #####
##########################################################################
def construct_rot_vectors(Nres,D,delta_uvw):
	vectors = np.zeros((Nres*D,3))
	#radii = np.zeros((Nres,3))				# array with how far away from principal axes of rotation
	if D != 3:
		print 'ERROR: CAN ONLY DO ROTATIONS IN THREE DIMENSIONS'
	else:
		for k in range(3):		
			for j in range(Nres):
				if k == 0:		## rotation about u axis										
					vectors[Nres*j+1,k] = delta_uvw[j,2]
					vectors[Nres*j+2,k] = -1*delta_uvw[j,1]
					#radii[j,k] = np.sqrt(delta_uvw[j,2]**2 + delta_uvw[j,1]**2)
				if k == (1):	## rotation about v axis
					vectors[Nres*j,k] = -1*delta_uvw[j,2]
					vectors[Nres*j+2,k] = delta_uvw[j,0]
					#radii[j,k] = np.sqrt(delta_uvw[j,2]**2 + delta_uvw[j,0]**2)
				if k == (2):	## rotation about w axis
					vectors[Nres*j,k] = delta_uvw[j,1]
					vectors[Nres*j+1,k] = -1*delta_uvw[j,0]
					#radii[j,k] = np.sqrt(delta_uvw[j,1]**2 + delta_uvw[j,0]**2)
	return vectors#, radii
# ############################################################################
	
	
	
	