export with_loc_j=1 # Column
export with_loc_k=1 # Exchange
export with_loc_f=1 # Fock
export with_dist=0
export with_sratio=0
export with_sosv=0
export with_fosv=0
export with_kosv=0
export with_t2=0
export with_fene=0
export with_fene_mat=1
export sym_fenemat=0
export t2_type=4
export nosv_fea=8
export int_type=mp2int
#export basis=631g
export basis=ccpvtz
export lmo_type=boys
#export lmo_type=pm
#export mol_system=butane
#export path_raw=/scratch_sh/ml_datasets/raw_dataset/$int_type/ml_features_"$int_type"_"$mol_system"_"$basis"_nonmbe.hdf5
#python3.9 ml_feature.py 
#ml_features_mp2int_water08_molpro_631g_pm.hdf5
#ml_features_mp2int_water16_molpro_631g_pm.hdf5
#ml_features_mp2int_water32_molpro_631g_pm.hdf5
#ml_features_mp2int_asp_protonated_molpro_631g_pm.hdf5
#ml_features_mp2int_water6_con_molpro_631g_pm.hdf5
#ml_features_mp2int_cam_water_molpro_631g_pm.hdf5
#for mol_system in asp_protonated_molpro ;do
for mol_system in dsgdb9nsd;do
	export mol_system=$mol_system
	echo $mol_system
	export path_raw=/scratch_sh/ml_datasets/raw_dataset/$int_type/ml_features_"$int_type"_"$mol_system"_"$basis"_"$lmo_type".hdf5
	python3.9 ml_feature.py
done
