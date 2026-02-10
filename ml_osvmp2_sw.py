from genericpath import isfile
import os
import sys
import time
import subprocess
import math
import scipy
from scipy.special import expit
import numpy as np
import h5py
#import psutil
import warnings
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, regularizers, metrics
from tensorflow.keras.layers import Lambda, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler#also MinMAx
from sklearn.decomposition import PCA
from joblib import dump, load

tf.keras.backend.set_floatx('float64')
hartee2kcalmol = 627.5

def str2list(istr, dtype="s"):
    if istr is None:
        return None
    elif type(istr) == list:
        return istr
    else:
        if dtype in ["s", str]:
            formater = str
        elif dtype in ["i", int]:
            formater = int
        elif dtype in ["f8", float]:
            formater = float
        return [formater(i) for i in istr.strip('][').split(',')]

def destination_generation(self):
    #system/descriptor/input
    paths_descriptor = {}
    for isystem in self.systems_test:
        path_descriptor = "%s/descriptors/%s/%s"%(self.main_dir, isystem, self.descriptor)
        paths_descriptor[isystem] = path_descriptor
    return paths_descriptor

def print_align(msg_list, align='l', align_1=None, indent=0, printout=False):
    if align_1 is None:
        align_1 = align
    align_list = []
    for align_i in [align_1, align]:
        align_format = []
        for i in list(align_i):
            if i == 'l':
                align_format.append('<')
            elif i == 'c':
                align_format.append('^')
            elif i == 'r':
                align_format.append('>')
        align_list.append(align_format)
    len_col = []
    for col_i in zip(*msg_list):
        len_col.append(max([len(str(i)) for i in col_i])+2)
    msg = ''
    for idx, msg_i in enumerate(msg_list):
        if idx == 0:
            align_i = align_list[0]
        else:
            align_i = align_list[1]
        msg += ' ' * indent
        msg += ''.join([('{:%s%d} '%(ali, li)).format(str(mi)) for ali, li, mi in zip(align_i, len_col, msg_i)])
        if idx != len(msg_list)-1:
            msg += '\n'
    if printout:
        print(msg)
    return msg + "\n"

def get_mem_aval():
    mem_info = subprocess.check_output("free", shell=True)
    return int(((mem_info.decode("utf-8").split("\n"))[1].split(" "))[-1])
'''def get_mem_aval(max_mem):
    pid = os.getpid()
    mem_dic = psutil.Process(int(pid)).memory_info()
    mem_used =  (mem_dic[0] - mem_dic[2] + mem_dic[2])*1e-6
    mem_aval = max_mem - mem_used
    return mem_aval '''

def list_from_environ(nsys, num, dtype="i"):
    if num is None:
        return num
    elif "[" in num:
        return str2list(num, dtype=dtype)
    else:
        if dtype in ["s", str]:
            formater = str
        elif dtype in ["i", int]:
            formater = int
        elif dtype in ["f8", float]:
            formater = float
        return [formater(num)] * nsys

def num_from_environ(num, dtype):
    if num is None:
        return None
    else:
        if dtype in ["i", int]:
            formater = int
        elif dtype in ["f8", float]:
            formater = float
        return formater(num)

def get_logger(logger):
    def neg_log(a):
        return np.log(-a)
    def log_back(a):
        return -math.e**(a)
    def sig_back(a):
        return -np.log(1/(a) - 1)
    if logger is None:
        logger_back = None
    if logger == 'log':
        logger = neg_log
        logger_back = log_back
    elif logger == 'sigmoid':
        logger = expit
        logger_back = sig_back
    elif logger == 'tanh':
        logger = np.tanh
        logger_back = np.arctanh
    return logger, logger_back

def get_scaler(scaler_name):
    if "log" in scaler_name:
        return get_logger(scaler_name)
    elif scaler_name == "std":
        return StandardScaler()
    elif scaler_name == "RobustScaler":
        return RobustScaler()
    elif scaler_name == "MinMax":
        return MinMaxScaler()
    elif scaler_name == "pca":
        return PCA(n_components='mle')


def get_scaler_dic(self):
    for scaler_type in self.scaler_dic.keys():
        scaler_name = self.scaler_dic[scaler_type]
        if scaler_name is not None:
            self.scaler_dic[scaler_type] = {}
            for ptype in self.pairtype_list:
                if "log" in scaler_type:
                    log_trans, log_back = get_logger(scaler_name)
                    self.scaler_dic[scaler_type][ptype] = log_trans
                    self.scaler_dic["%s_back"%scaler_type][ptype] = log_back
                else:      
                    if ("y_" in scaler_type) or (self.use_outcore is False):
                        if self.load_model is None:
                            self.scaler_dic[scaler_type][ptype] = get_scaler(scaler_name)
                        else:
                            file_scaler = '%s/my_scaler/%s/%s.bin'%(self.load_model, scaler_type, ptype)
                            if os.path.isfile(file_scaler):
                                self.scaler_dic[scaler_type][ptype] = load(file_scaler)

                    else:
                        self.scaler_dic[scaler_type][ptype] = []
                        for idx in range(self.nfea_dic[ptype]):
                            if self.load_model is None:
                                self.scaler_dic[scaler_type][ptype].append(get_scaler(scaler_name))
                            else:
                                file_scaler = f"{self.load_model}/my_scaler/{scaler_type}/{ptype}/{idx}.bin"
                                if os.path.isfile(file_scaler):
                                    self.scaler_dic[scaler_type][ptype] = load(file_scaler)

def save_scaler(self, scaler_type, ptype, fea_idx=None):
    dir_scaler = '%s/my_scaler/%s'%(self.output_dir, scaler_type)
    os.makedirs(dir_scaler, exist_ok=True)
    if fea_idx is None:
        file_scaler = f'{dir_scaler}/{ptype}.bin'
        scaler = self.scaler_dic[scaler_type][ptype]
    else:
        os.makedirs(f"{dir_scaler}/{ptype}", exist_ok=True)
        file_scaler = f'{dir_scaler}/{ptype}/{fea_idx}.bin'
        scaler = self.scaler_dic[scaler_type][ptype][fea_idx]
    dump(scaler, file_scaler, compress=True)

def scale_data(self, itype, ptype, idata, fit_y=False, fea_idx=None):
    if fit_y: 
        if (self.scaler_dic["y_log"] is not None) and \
            (itype != "test") and \
            (ptype in self.ptype_log):
            idata = self.scaler_dic["y_log"](idata)
        if (self.scaler_dic["y_scaler"] is not None) and \
            (itype != "test"):
            if itype == "train":
                self.scaler_dic["y_scaler"][ptype].fit(idata.reshape(-1,1))
                save_scaler(self, "y_scaler", ptype)
            idata = self.scaler_dic["y_scaler"][ptype].transform(idata.reshape(-1,1)).ravel()
        if self.interpolate:
            idata *= 0.9
            idata += 0.05
    else:
        if self.scaler_dic["x_scaler"] is not None:
            if itype == "train":
                self.scaler_dic["x_scaler"][ptype].fit(idata)
                save_scaler(self, "x_scaler", ptype, fea_idx)
            idata = self.scaler_dic["x_scaler"][ptype].transform(idata)
        if self.scaler_dic["pca_scaler"] is not None: #PCA does not function
            if itype == "train":
                self.scaler_dic["pca_scaler"][ptype].fit(idata)
                save_scaler(self, "pca_scaler", ptype, fea_idx)
            idata = self.scaler_dic["pca_scaler"][ptype].transform(idata)
            #idata_pca = self.scaler_dic["pca_scaler"][ptype].transform(idata)
            #idata = np.hstack((idata_pca, idata.reshape(-1,1)))
    
    return idata

class NN_MBE():
    def __init__(self):
        ######################################### NN skeleton #################################################
        self.interpolate = bool(int(os.environ.get('interpolate', 0))) # this controls the NN archetecture there is no need to save it.
        self.kernel_regularize = bool(int(os.environ.get('kernel_regularize', 0)))
        self.kernel_l2_value = float(os.environ.get('kernel_l2_value'))
        self.regularize = bool(int(os.environ.get('regularize', 0)))
        self.act_l1_value = float(os.environ.get('act_l1_value'))
        
        ####################################################################
        self.scaler_dic = {}
        self.scaler_dic["x_scaler"] = os.environ.get('x_scaler', None)
        self.scaler_dic["y_scaler"] = os.environ.get('y_scaler', None)
        self.scaler_dic["y_log"] = os.environ.get('y_log', None) #Only close off-diag2 and remote off-diag will be scaled
        self.ptype_log = ["offdiag_close2", "offdiag_remote"]
        use_pca = bool(int(os.environ.get('pca', 0)))
        if use_pca:
            self.scaler_dic["pca_scaler"] = "pca"
        else:
            self.scaler_dic["pca_scaler"] = None
        self.no_hidden_layers = int(os.environ.get('no_hidden_layers', 3))
        self.hidden_neu_list = list_from_environ(self.no_hidden_layers, os.environ.get('no_neuron_hidden_layer'))
        self.hidden_act_list = list_from_environ(self.no_hidden_layers, os.environ.get('activation_hidden_layer'), dtype=str)
        #log_transform = bool(int(os.environ.get('log',0)))
        self.dropout_list = str2list(os.environ.get('dropout_list'), "i") # really? float or int?
        self.batch_norm = bool(int(os.environ.get('batch_norm', 0)))
        self.no_epochs = int(os.environ.get('Epochs'))
        self.no_epochs_1 = int(os.environ.get('epo_1'))
        self.no_epochs_2 = int(os.environ.get('epo_2'))
        self.no_epochs_3 = int(os.environ.get('epo_3'))
        self.no_epochs_list = [self.no_epochs_1, self.no_epochs_2, self.no_epochs_3]
        self.learning_rate = float(os.environ.get('learning_rate'))
        self.decay =  float(os.environ.get('decay'))
        self.loss = os.environ.get('loss_function')
        ##############################################################################
        self.max_memory = int(os.environ.get('max_memory', 4000)) #MB, default 4GB
        self.descriptor = os.environ.get('descriptor')
        des_split = self.descriptor.split("_")
        self.basis = (des_split[-1].replace("_", "").replace("-", "")).lower()
        int_type = des_split[0]
        if des_split[1] in ["pm", 'boys']:
            lmo_type = des_split[1]
        else:
            lmo_type = None
        self.int_type = os.environ.get("int_type", int_type) #rhfint mp2int
        self.lmo_type = os.environ.get("lmo_type", lmo_type)
        self.qm_method = os.environ.get("qm_method", "osvmp2").lower()
        self.osv_tol = os.environ.get("osv_tol", "1e-4")        
        self.use_mbe = bool(int(os.environ.get('use_mbe', 1)))
        self.energy_decom = os.environ.get('energy_decom', None)
        self.use_osv = bool(int(os.environ.get('use_osv', 1)))
        self.use_outcore = bool(int(os.environ.get('use_outcore', 0)))
        self.save_model = bool(int(os.environ.get('save_model', 1)))
        self.load_model = os.environ.get('load_model')#Path to the directory
        if self.load_model is None:
            self.data_types   = ["train", "valid", "test"]
            self.systems_train = str2list(os.environ.get('systems_train', None))
            systems_test = str2list(os.environ.get('systems_test', None))
            systems_test_only = [i for i in systems_test if i not in self.systems_train]
            self.systems_test = self.systems_train + systems_test_only

            self.train_ratio = list_from_environ(len(self.systems_train), os.environ.get('no_train'), dtype=float)
            no_train = list_from_environ(len(self.systems_train), os.environ.get('no_train'))
            no_valid = list_from_environ(len(self.systems_train), os.environ.get('no_valid'))
            self.no_train = {}
            self.no_valid = {}
            for idx, isystem in enumerate(self.systems_train):
                self.no_train[isystem] = no_train[idx]
                self.no_valid[isystem] = no_valid[idx]
        else:
            self.data_types = ["test"]
            self.systems_train = []
            self.systems_test = str2list(os.environ.get('systems_test', None))
        dir_pair_ene = os.environ.get("dir_pair_ene", "/scratch_sh/ml_datasets/pair_energy/pene_split/%s"%self.int_type)
        dir_rhf_ene = os.environ.get("dir_rhf_ene", "/scratch_sh/ml_datasets/rhf_energy")
        self.file_pair_ene = {} 
        self.file_rhf_ene = {}
        for isystem in self.systems_test:
            self.file_pair_ene[isystem] = "%s/pene_split_%s_%s_%s_%s_%s_%s.hdf5"%(dir_pair_ene, self.int_type, isystem, 
                                                                               self.qm_method, self.osv_tol, self.basis,
                                                                               self.lmo_type)
            self.file_rhf_ene[isystem] = "%s/rhfe_%s_%s.hdf5"%(dir_rhf_ene, isystem, self.basis)
        self.pairtype_full = ["diag", "offdiag_close", "offdiag_remote"]
        self.pairtype_list = str2list(os.environ.get('pairtype_list', self.pairtype_full))
        #self.pairtype_list = ["diag", "offdiag_close", "offdiag_remote"]
        self.ptype_offdiagc = []

        self.start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.main_dir = os.environ.get('main_dir', ".")
        self.file_sample = os.environ.get('file_sample', None)
        self.output_dir = "%s/test/%s_%s_%s_%s"%(self.main_dir, self.start_time, self.descriptor, self.qm_method, self.osv_tol)
        if self.load_model is None:
            self.output_dir += "_ep%d"%self.no_epochs
            self.output_dir += "_tr%d"%no_train[0]
        if self.scaler_dic["x_scaler"] is not None:
            self.output_dir += "_x%s"%self.scaler_dic["x_scaler"]
        if self.scaler_dic["y_scaler"] is not None:
            self.output_dir += "_y%s"%self.scaler_dic["y_scaler"]
        if self.scaler_dic["pca_scaler"] is not None:
            self.output_dir += "_x%s"%self.scaler_dic["pca_scaler"]
        self.paths_descriptor = destination_generation(self)
        self.file_summary = "%s/summary.log"%self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)


        #Print the information
        if "osv"  in self.qm_method:
            if self.qm_method == "osvccsd-t":
                qm_method = "OSV-CCSD(T)"
            else:
                qm_method = "OSV-" + self.qm_method[3:].upper()
            qm_method = "%s (losv=%s)"%(qm_method, self.osv_tol)
        elif self.qm_method == "triple":
            qm_method = "perturbative triples"
        else:
            raise NotImplementedError("%s is not supported"%self.qm_method)

        print("Output: %s"%self.output_dir)
        msg = "\nInput information\n"
        msg += "-"*70 + "\n"
        msg_list = [["Features:", self.descriptor],
                    ["QM method:", qm_method],
                    ["Pair types", self.pairtype_list],
                    ["Training systems:", self.systems_train],
                    ["Testing systems:", self.systems_test],
                    ['Number of Hidden layers:', self.no_hidden_layers],
                    ['Number of Hidden neurons:', self.hidden_neu_list[0]],
                    ['Activation function:', self.hidden_act_list[0]],
                    ['Dropout_list:', self.dropout_list],
                    ['Batch normalization:', self.batch_norm],
                    ['Number of epochs:', self.no_epochs],
                    ['Learning rate:', self.learning_rate],
                    ['Decay:', self.decay],
                    ['Loss fucntion:', self.loss],
                    ['Input scaler:', self.scaler_dic["x_scaler"]],
                    ['Input PCA scaler:', self.scaler_dic["pca_scaler"]],
                    ['Output scaler:', self.scaler_dic["y_scaler"]],
                    ['Output log:', self.scaler_dic["y_log"]],
                    ['interpolation:', self.interpolate]]
        if self.kernel_regularize:
            msg_list.append(['kernel l2 reguration:', self.kernel_l2_value])
        if self.regularize:
            msg_list.append(['activity l1 reguration:', self.act_l1_value])
        msg_list = [imsg for imsg in msg_list if imsg[1] is not None]
        msg += print_align(msg_list, "ll")
        msg += "-"*70 + "\n"
        print(msg)
        with open(self.file_summary, "a") as f:
            f.write(msg)
        
            
        '''self.scaler_dic["x_scaler"] = get_scaler_dic(self, self.scaler_dic["x_scaler"], self.pairtype_list) #A dictionary of scalers
        self.scaler_dic["y_scaler"] = get_scaler_dic(self, self.scaler_dic["y_scaler"], self.pairtype_list)'''

            
        #self.scaler_dic["y_log"], self.scaler_dic["y_log_back"] = get_logger(self.scaler_dic["y_log"])


        
#read
    def train_test_split(self): 
        def get_data_mol(isystem, loc_dic, mols_close, mols_remote, all_dic):
            loc_dic[isystem] = {}
            for ptype in self.pairtype_list:
                loc_dic[isystem][ptype] = {}
                if ptype in ["diag", "offdiag_close"]:
                    for imol in mols_close:
                        loc_dic[isystem][ptype][imol] = [0, 0]
                elif mols_remote is not None:
                    for imol in mols_remote:
                        loc_dic[isystem][ptype][imol] = [0, 0]
            return loc_dic

        def get_data_loc(loc_dic, all_dic):
            for ptype in self.pairtype_list:
                idx0 = 0
                for isystem in loc_dic.keys():
                    for imol in loc_dic[isystem][ptype].keys():
                        idx1 = idx0 + all_dic[isystem][ptype][imol][0]
                        loc_dic[isystem][ptype][imol] = [idx0, idx1]
                        idx0 = idx1
            return loc_dic
        def prepare_data(all_dic):
            self.shape_dic = {}
            #for dic_idx, idic in enumerate([self.loc_train, self.loc_valid, self.loc_test]):
            #    itype = self.data_types [dic_idx]
            for itype in ["train", "valid"]:#self.data_types:
                self.shape_dic[itype] = {}
                for ptype in self.pairtype_list:
                    mol_dic = all_dic[self.systems_test[0]][ptype]
                    shape1 = mol_dic[list(mol_dic.keys())[0]][1]
                    self.shape_dic[itype][ptype] = [0, shape1]
                for isystem in self.loc_dic[itype].keys():
                    for ptype in self.pairtype_dic[isystem]:
                        for imol in self.loc_dic[itype][isystem][ptype].keys():
                            idx0, idx1 = self.loc_dic[itype][isystem][ptype][imol]
                            self.shape_dic[itype][ptype][0] += idx1 - idx0
                file_data = "%s/data_%s.hdf5"%(self.output_dir, itype)
                with h5py.File(file_data, "w") as fout:
                    for ptype in self.shape_dic[itype].keys():
                        fout.create_dataset(ptype, shape=self.shape_dic[itype][ptype], dtype="f8")
                        for isystem in self.loc_dic[itype].keys():
                            with h5py.File("%s/descriptors.hdf5"%self.paths_descriptor[isystem], "r") as ffea:
                                with h5py.File(self.file_pair_ene[isystem], "r") as fpene:
                                    for imol in self.loc_dic[itype][isystem][ptype].keys():
                                        idx0, idx1 = self.loc_dic[itype][isystem][ptype][imol]
                                        idata = ffea[ptype][imol][:]
                                        pene_list = fpene[imol][ptype][:].reshape(-1, 1)
                                        idata = np.hstack((idata, pene_list))
                                        fout[ptype][idx0:idx1] = idata
            if not (self.scaler_dic["x_scaler"] is None and 
                    self.scaler_dic["y_scaler"] is None and
                    self.scaler_dic["pca_scaler"] is None and
                    self.scaler_dic["y_log"] is None and
                    self.interpolate is None):
                #self.scal_fea_idx = {}
                self.nsamp_dic = {}
                self.pene_dic = {}
                self.feature_dic = {}
                for ptype in self.pairtype_list:
                    self.feature_dic[ptype] = {}
                    self.nsamp_dic[ptype] = {}
                    self.pene_dic[ptype] = {}
                    for itype in ["train", "valid"]:#self.data_types:
                        #itype = "train"
                        file_data = "%s/data_%s.hdf5"%(self.output_dir, itype)
                        file_data_scale = "%s/data_%s_scale.hdf5"%(self.output_dir, itype)
                        #Memory control
                        mem_avail = 0.5*get_mem_aval()#self.max_memory) #In MB
                        max_size = mem_avail/1e-6/8
                        with h5py.File(file_data, "r+") as fin:
                            #fit y
                            nsamp = fin[ptype].shape[0]
                            self.nsamp_dic[ptype][itype] = nsamp
                            y_list = scale_data(self, itype, ptype, fin[ptype][:, -1], fit_y=True)

                            #fit x
                            if self.use_outcore:
                                nfea = self.nfea_dic[ptype]
                                fin[ptype][:, -1] = y_list
                                for fidx in np.arange(nfea):
                                    idata = fin[ptype][:, fidx]
                                    idata = scale_data(self, itype, ptype, idata, fea_idx=fidx)
                                    fin[ptype][:, fidx] = idata
                            else:
                                idata = scale_data(self, itype, ptype, fin[ptype][:, :-1])
                                self.nfea_dic[ptype] = idata.shape[-1]
                                self.pene_dic[ptype][itype] = y_list
                                self.feature_dic[ptype][itype] = idata


        def get_mols_common():
            mols_dic = {}
            nfea_dic = {}
            for isystem in self.systems_test:
                mols_dic[isystem] = {}
                with h5py.File(f"{self.paths_descriptor[isystem]}/descriptors.hdf5", "r") as fdes:
                    for ptype in fdes.keys():
                        mols_dic[isystem][ptype] = {}
                    mols_des = list(fdes["diag"].keys())
                    with h5py.File(self.file_pair_ene[isystem], "r") as fpene:
                        mols_pene = list(fpene.keys())
                    mols_common = list(set(mols_des).intersection(mols_pene))
                    mols_common.sort()
                    mols_dic[isystem]["diag"] = mols_dic[isystem]["offdiag_close"] = mols_common
                    nfea_dic["diag"] = fdes["diag"][mols_des[0]].shape[-1]
                    nfea_dic["offdiag_close"] = fdes["offdiag_close"][mols_des[0]].shape[-1]
                    if "offdiag_remote" in mols_dic[isystem].keys():
                        mols_des_remote = list(fdes["offdiag_remote"].keys())
                        mols_dic[isystem]["offdiag_remote"] = list(set(mols_common).intersection(mols_des_remote))
                        mols_dic[isystem]["offdiag_remote"].sort()
                        nfea_dic["offdiag_remote"] = fdes["offdiag_remote"][mols_des_remote[0]].shape[-1]
            return mols_dic, nfea_dic
                
        #Features of all molecules are collected in a 2d (npair, nfeature) array
        #all_dic = {}
        self.mols_dic, self.nfea_dic = get_mols_common()
        get_scaler_dic(self)
        self.loc_dic = {}
        for itype in self.data_types + ["total"]:
            self.loc_dic[itype] = {}
        #self.loc_train = {}
        #self.loc_valid = {}
        #self.loc_test = {}
        self.pairtype_dic = {}
        pairtype_list = []
        fsamp = h5py.File("%s/mol_samp.hdf5"%self.output_dir, "w")
        for isystem in self.systems_test:
            ifolder = self.paths_descriptor[isystem]
            self.loc_dic["total"][isystem] = {}
            self.pairtype_dic[isystem] = []
            with h5py.File("%s/descriptors.hdf5"%ifolder, "r") as fdes:
                for ptype in fdes.keys():
                    if ptype in self.pairtype_list:
                        self.pairtype_dic[isystem].append(ptype)
                        pairtype_list.append(ptype)
                    self.loc_dic["total"][isystem][ptype] = {}
                    #for imol in fdes[ptype].keys():
                    for imol in self.mols_dic[isystem][ptype]:
                        nmol, nfea = fdes[ptype][imol].shape
                        self.loc_dic["total"][isystem][ptype][imol] = (nmol, nfea+1)
                        

            #Ramdonly pick training molecules
            mols_all = list(self.loc_dic["total"][isystem]["diag"].keys())
            mols_valid = mols_train = mols_test = None
            mols_train_remote = mols_valid_remote = mols_test_remote = None
            if isystem in self.systems_train:
                if self.file_sample is not None:
                    with h5py.File(self.file_sample, "r") as fsamp_chk:
                        mols_valid = [imol.decode("utf-8") for imol in fsamp_chk["%s/valid"%isystem][:]]
                        mols_train = [imol.decode("utf-8") for imol in fsamp_chk["%s/train"%isystem][:]]
                        mols_test = [imol.decode("utf-8") for imol in fsamp_chk["%s/test"%isystem][:]]
                else:
                    mols_train_val = np.random.choice(mols_all, self.no_train[isystem], replace=False)
                    mols_valid = mols_train_val[:self.no_valid[isystem]]
                    mols_train = mols_train_val[self.no_valid[isystem]:]
                    mols_test = list(set(mols_all) - set(mols_train_val))
                fsamp["%s/valid"%isystem] = np.asarray(mols_valid, dtype="S")
                fsamp["%s/train"%isystem] = np.asarray(mols_train, dtype="S")
                fsamp["%s/test"%isystem] = np.asarray(mols_test, dtype="S")
                if "offdiag_remote" in self.loc_dic["total"][isystem].keys():
                    mols_train_remote = list(set(self.loc_dic["total"][isystem]["offdiag_remote"]).intersection(mols_train))
                    mols_valid_remote = list(set(self.loc_dic["total"][isystem]["offdiag_remote"]).intersection(mols_valid))
            else:
                mols_test = mols_all
            if "offdiag_remote" in self.loc_dic["total"][isystem].keys():
                mols_test_remote = list(set(self.loc_dic["total"][isystem]["offdiag_remote"]).intersection(mols_test))
            
            for mol_list in [mols_valid, mols_test, mols_train_remote, mols_valid_remote, mols_test_remote]:
                if mol_list is not None:
                    mol_list.sort()
            #Get the locations of the features for each molecule
            if isystem in self.systems_train:
                self.loc_dic["train"] = get_data_mol(isystem, self.loc_dic["train"], mols_train, mols_train_remote, self.loc_dic["total"])
                self.loc_dic["valid"] = get_data_mol(isystem, self.loc_dic["valid"], mols_valid, mols_valid_remote, self.loc_dic["total"])
            self.loc_dic["test"] = get_data_mol(isystem, self.loc_dic["test"], mols_test, mols_test_remote, self.loc_dic["total"])
        fsamp.close()
        self.pairtype_list = sorted(list(set(pairtype_list)))
        for itype in self.data_types:
            get_data_loc(self.loc_dic[itype], self.loc_dic["total"])
        '''get_data_loc(self.loc_train, all_dic)
        get_data_loc(self.loc_valid, all_dic)
        get_data_loc(self.loc_test, all_dic)'''
        
        #Prepare training, validation and testing sets
        if self.load_model is None:
            prepare_data(self.loc_dic["total"])

        #Print out important information
        #test_label = ["training", "validation", "predction", "total"]
        test_label = self.data_types + ["total"]
        
        for isystem in self.loc_dic["total"].keys():
            msg = "%s\n"%isystem
            #msg_list = [["", "diag", "offdiag_close", "offdiag_remote"]]
            msg_list = []
            nmol_list = []
            npair_list = []
            nmol_dic = {}
            npair_dic = {}
            #for dic_idx, idic in enumerate([self.loc_train, self.loc_valid, self.loc_test, all_dic]):
            for itype in test_label:
                if isystem in self.loc_dic[itype].keys():
                    nmol_dic[itype] = {}
                    npair_dic[itype] = {}
                    idic = self.loc_dic[itype][isystem]
                    for ptype in self.pairtype_list:
                        if ptype not in idic.keys():
                            nmol = npair = 0
                        else:
                            mol_list = idic[ptype].keys()
                            nmol = len(mol_list)
                            npair = 0
                            for imol in mol_list:
                                if (itype == "total"):
                                    npair += idic[ptype][imol][0]
                                else:
                                    idx0, idx1 = idic[ptype][imol]
                                    npair += (idx1 - idx0)
                        nmol_list.append(nmol)
                        npair_list.append(npair)
                        nmol_dic[itype][ptype] = nmol
                        npair_dic[itype][ptype] = npair
            max_nmol = max(len(str(np.max(nmol_list))), 4)
            max_npair = max(len(str(np.max(npair_list))), 5)

            msg_list += [[""] + self.pairtype_list] 
            msg_list += [[""] + ["-"*(max((max_nmol+max_npair+2), len(ptype))) for ptype in self.pairtype_list]]
            msg_list += [[""] + [("{:>%ds}  {:>%ds}"%(max_nmol, max_npair)).format("nmol", "npair") for ptype in self.pairtype_list]]
            
            for itype in nmol_dic.keys():
                msg_i = [itype]
                for ptype in nmol_dic[itype].keys():
                    nmol = nmol_dic[itype][ptype]
                    npair = npair_dic[itype][ptype]
                    msg_i.append(("{:>%dd}  {:>%dd}"%(max_nmol, max_npair)).format(nmol, npair))
                msg_list.append(msg_i)
            msg += print_align(msg_list, "l"+"r"*(len(self.pairtype_list)))
            msg += "-"*52 + "\n"
            print(msg)
            with open(self.file_summary, "a") as f:
                f.write(msg)
    def get_ene_cal(self):
        ene_dic = {}
        for isystem in self.systems_test:
            ifolder = self.paths_descriptor[isystem]
            ene_dic[isystem] = {}
            with h5py.File("%s/descriptors.hdf5"%ifolder, "r") as f:
                for ptype in f.keys():
                    ene_dic[isystem][ptype] = {}
                    for imol in self.mols_dic[isystem][ptype]:
                        ene_dic[isystem][ptype][imol] = f[ptype][imol].shape

    def build(self, ptype):
        if self.load_model is not None:
            file_model = f"{self.load_model}/my_model/{ptype}.h5"
            if os.path.isfile(file_model):
                model = tf.keras.models.load_model(file_model)
            else:
                return None
        else:
            def scale_mae(y_true, y_pred):
                scale_factor = K.round(K.abs(tf.experimental.numpy.log10(K.abs(y_true)))) - 2
                scale_factor = K.switch(K.less(scale_factor, 0), K.zeros_like(scale_factor), scale_factor)
                err_scale = (y_pred - y_true) * 10**scale_factor
                #K.print_tensor(err_scale, message='err_scale = ')
                return K.mean(K.abs(err_scale), axis=-1)
            def log_err(y_true, y_pred):
                err_scale = np.log(y_true) - np.log(y_pred)
                return K.mean(K.abs(err_scale), axis=-1)
            self.optimizer = optimizers.Adam(learning_rate=self.learning_rate,decay=self.decay) # can change this later.
            model = Sequential()
            '''file_data = "%s/data_%s.hdf5"%(self.output_dir, self.data_types[0])
            with h5py.File(file_data, "r") as fdata:
                nsamp, nfea = fdata[ptype].shape
                input_dim = nfea - 1'''
            input_dim = self.nfea_dic[ptype]
            #input_dim = self.shape_dic["train"][ptype][1] - 1
            output_dim = 1
            for hidden_layer in range(self.no_hidden_layers):
                if hidden_layer == 0:
                    if self.regularize:
                        model.add(Dense(self.hidden_neu_list[0],input_shape=(input_dim,),activity_regularizer=regularizers.l1(self.act_l1_value)))
                    elif self.kernel_regularize:
                        model.add(Dense(self.hidden_neu_list[0],input_shape=(input_dim,),kernel_regularizer=regularizers.l2(self.kernel_l2_value)))

                    else:
                        model.add(Dense(self.hidden_neu_list[0],input_shape=(input_dim,)))
                    if self.batch_norm:
                        model.add(BatchNormalization())
                    model.add(Activation(self.hidden_act_list[0]))
                    if self.dropout_list is not None:
                        model.add(Dropout(self.dropout_list[0]))
                else:
                    if self.regularize:
                        model.add(Dense(self.hidden_neu_list[hidden_layer],activity_regularizer=regularizers.l1(self.act_l1_value)))
                    elif self.kernel_regularize:
                        model.add(Dense(self.hidden_neu_list[hidden_layer],kernel_regularizer=regularizers.l2(self.kernel_l2_value)))
                    else:
                        model.add(Dense(self.hidden_neu_list[hidden_layer]))
                    if self.batch_norm:
                        model.add(BatchNormalization())
                    model.add(Activation(self.hidden_act_list[hidden_layer]))
                    if self.dropout_list is not None:
                        model.add(Dropout(self.dropout_list[hidden_layer]))

            if self.interpolate:
                act = 'sigmoid'
                print('Use sigmoid in the output layer')
            else:
                act = 'linear'

            if self.batch_norm:
                if self.regularize:
                    model.add(Dense(output_dim,activity_regularizer=regularizers.l1(self.act_l1_value)))
                elif self.kernel_regularize:
                    model.add(Dense(output_dim,kernel_regularizer=regularizers.l2(self.kernel_l2_value)))
                else:
                    model.add(Dense(output_dim))

                model.add(BatchNormalization())
                model.add(Activation(act))
            else:
                if self.regularize:
                    model.add(Dense(output_dim,activation=act,activity_regularizer=regularizers.l1(self.act_l1_value))) #can I restrict the ouput range of this? 0-1, final one set a range
                elif self.kernel_regularize:
                    model.add(Dense(output_dim,activation=act,kernel_regularizer=regularizers.l2(self.kernel_l2_value)))
                else:
                    model.add(Dense(output_dim,activation=act))
            model.compile(loss="mae",metrics=[metrics.MeanAbsolutePercentageError(name="m%error")],optimizer=self.optimizer)
        model.summary()
        return model

    def fit(self, model, ptype, pidx):
        
        #Memory control is required
        checkpoint_dir = f'{self.output_dir}/checkpoints' 
        os.makedirs(checkpoint_dir,exist_ok=True)
        
        '''with h5py.File("%s/data_train.hdf5"%(self.output_dir), "r") as f:
            data_train = f[ptype][:]
        with h5py.File("%s/data_valid.hdf5"%(self.output_dir), "r") as f:
            data_valid = f[ptype][:]'''
        batch_size = 64
        nsamp_train = self.nsamp_dic[ptype]["train"]
        print("Batch size: %d"%batch_size)
        model_checkpoint = ModelCheckpoint(
            filepath = f'{checkpoint_dir}'+'/train_chk_%s.hdf5'%ptype,
            save_weights_only=False,
            monitor = 'val_loss',
            mode ='min',
            save_best_only=True,
            #save_freq = 10*int(nsamp_train//batch_size)
            period = 100
            )
        #
        data_valid = (self.feature_dic[ptype]["valid"], self.pene_dic[ptype]["valid"])
        model.fit(self.feature_dic[ptype]["train"],
                  self.pene_dic[ptype]["train"],
                  validation_data=data_valid,
                  batch_size=batch_size, 
                  epochs=self.no_epochs_list[pidx]) 
        if self.save_model:
            model.save("%s/my_model/%s.h5"%(self.output_dir, ptype))
        
    def predict(self, model, ptype):
        file_pene = open("%s/ene_pair_%s.log"%(self.output_dir, ptype), "w")
        file_pene_sumary = open("%s/ene_pair_summary_%s.log"%(self.output_dir, ptype), "w")
        self.ene_cal_dic[ptype] = {}
        self.ene_pred_dic[ptype] = {}
        for isystem in self.systems_test:
            self.ene_cal_dic[ptype][isystem] = {}
            self.ene_pred_dic[ptype][isystem] = {}
        #for idx_dic, loc_dic in enumerate([self.loc_train, self.loc_valid, self.loc_test]):
        for itype in self.data_types:
            #itype = self.data_types [idx_dic]
            #with h5py.File("%s/data_%s.hdf5"%(self.output_dir, itype), "r") as f:
                #for isystem in self.systems_test:
            for isystem in self.loc_dic[itype].keys():
                for imol in self.loc_dic[itype][isystem][ptype].keys():
                    idx0, idx1 = self.loc_dic[itype][isystem][ptype][imol]
                    if itype in ["train", "valid"]:
                        #idata = f[ptype][idx0:idx1]
                        ifea = self.feature_dic[ptype][itype][idx0:idx1]
                        ene_cal_pairs = self.pene_dic[ptype][itype][idx0:idx1]
                    else:
                        with h5py.File("%s/descriptors.hdf5"%self.paths_descriptor[isystem], "r") as ffea:
                            ifea = ffea[ptype][imol][:]
                        ifea = scale_data(self, itype, ptype, ifea)
                        with h5py.File(self.file_pair_ene[isystem], "r") as fpene:
                            ene_cal_pairs = fpene[imol][ptype][:]
                        
                    #ene_cal_pairs = idata[:,-1].ravel()
                    #ene_pred_pairs = model.predict(idata[:, :-1]).ravel()
                    ene_pred_pairs = model.predict(ifea).ravel()
                    if (self.scaler_dic["y_scaler"] is not None):
                        ene_pred_pairs = self.scaler_dic["y_scaler"][ptype].inverse_transform(ene_pred_pairs.reshape(-1, 1)).ravel()
                        if (itype != "test"):
                            ene_cal_pairs = self.scaler_dic["y_scaler"][ptype].inverse_transform(ene_cal_pairs.reshape(-1, 1)).ravel()
                    if (self.scaler_dic["y_log"] is not None) and (ptype in self.ptype_log):
                        ene_pred_pairs = [self.scaler_dic["y_log_back"](i) for i in ene_pred_pairs]
                        if (itype != "test"):
                            ene_cal_pairs = self.scaler_dic["y_log_back"](ene_cal_pairs)
                    
                    err_ene_pairs = ene_pred_pairs - ene_cal_pairs
                    ene_cal = np.sum(ene_cal_pairs)
                    ene_pred = np.sum(ene_pred_pairs)
                    err_ene = ene_pred - ene_cal
                    print("%s %s %s ene_cal: %.8f, ene_pred: %.8f, error: %s"%(itype, isystem, imol, ene_cal, ene_pred, "{:>12.4E}".format(err_ene)))

                    msg = "%s %s %s\n"%(itype, isystem, imol)
                    for ecal, epred, err in zip(ene_cal_pairs, ene_pred_pairs, err_ene_pairs):
                        msg += "%.8f  %.8f  %s\n"%(ecal, epred, "{:>12.4E}".format(err))
                    file_pene.write(msg)
                    pene_maxae = max(np.abs(err_ene_pairs))
                    pene_mae = np.mean(np.abs(err_ene_pairs))
                    pene_me = np.mean(err_ene_pairs)
                    if len(ene_cal_pairs) > 0:
                        pene_pearson = scipy.stats.pearsonr(ene_cal_pairs, ene_pred_pairs)[0]
                    else:
                        pene_pearson = 1
                    pene_summary = f"{itype} {isystem} {imol} maxae: %.4E, mae: %.4E, me: %.4E, pearson: %.4f\n"%(pene_maxae, pene_mae, pene_me, pene_pearson)
                    file_pene_sumary.write(pene_summary)
                    self.ene_cal_dic[ptype][isystem][imol] = ene_cal
                    self.ene_pred_dic[ptype][isystem][imol] = ene_pred
                    self.ene_cal_dic["total"][isystem][imol] += ene_cal
                    self.ene_pred_dic["total"][isystem][imol] += ene_pred
        file_pene.close()
        file_pene_sumary.close()
        msg = "\nErrors for %s (kcal/mol)\n"%ptype
        msg += "-"*73 + "\n"
        #for idx_dic, loc_dic in enumerate([self.loc_train, self.loc_valid, self.loc_test]):
        for itype in self.data_types:
            msg += "%s\n"%itype
            #msg_list = [["", "MaxE", "MAE", "Pearson"]]
            msg_list = [["", "MaxAE", "MAE", "ME", "MARE", "Pearson"]]
            for isystem in self.loc_dic[itype].keys():
                ene_cal = []
                ene_pred = []
                for imol in self.loc_dic[itype][isystem][ptype].keys():
                    ene_cal.append(self.ene_cal_dic[ptype][isystem][imol])
                    ene_pred.append(self.ene_pred_dic[ptype][isystem][imol])
                ene_cal = np.asarray(ene_cal)
                ene_pred = np.asarray(ene_pred)
                err_list = ene_cal - ene_pred
                maxae = max(np.abs(err_list))
                mae = np.mean(np.abs(err_list))
                me = np.mean(err_list)
                mare = np.mean(np.abs(ene_cal-(ene_pred+me)))
                if len(ene_cal) > 1:
                    pearson = scipy.stats.pearsonr(ene_cal, ene_pred)[0]
                else:
                    pearson = 1
                msg_list.append([isystem] + ["%.6f"%(num*hartee2kcalmol) for num in [maxae, mae, me, mare]] +  ["%.4f"%pearson])
                #msg_list.append([isystem, "%.6f"%(maxe*hartee2kcalmol), "%.6f"%(mae*hartee2kcalmol), "%.4f"%pearson])
            msg += print_align(msg_list, "lrrrrr")
            msg += "-"*73 + "\n"
        print(msg)
        with open(self.file_summary, "a") as f:
            f.write(msg)

    def kernel(self):

        ##########################################
        '''
        Diagonal, close off-diagonal and remote off-diagonal pairs are treated separately
        '''
        self.train_test_split()
        #self.build(loss,learning_rate,decay)
        self.ene_cal_dic = {}
        self.ene_pred_dic = {}
        
        for idic in [self.ene_cal_dic, self.ene_pred_dic]:
            idic["total"] = {}
            for isystem in self.systems_test:
                idic["total"][isystem] = {}
                #for loc_dic in [self.loc_train, self.loc_valid, self.loc_test]:
                for itype in self.data_types:
                    if isystem in self.loc_dic[itype].keys():
                        for ptype in self.loc_dic[itype][isystem].keys():
                            for imol in self.loc_dic[itype][isystem][ptype].keys():
                                idic["total"][isystem][imol] = 0.0

        for pidx, ptype in enumerate(self.pairtype_list):
            len_msg = 80
            msg = "\n" + "#"*len_msg + "\n"
            text = " Training on %s pairs "%ptype
            sub_len = (len_msg - len(text))//2
            msg += "#"*sub_len + text + "#"*(len_msg-len(text)-sub_len) + "\n"
            msg += "#"*len_msg + "\n"
            print(msg)

            #Initialize the model
            model = self.build(ptype)
            if model is None:
                print(f"There is no model for {ptype}")
                break

            #Training
            if self.load_model is None:
                self.fit(model, ptype, pidx) 

            #Prediction
            self.predict(model, ptype)

        msg_ene = ""
        msg_ene_withhf = ""
        msg = "\nTotal errors (kcal/mol)\n"
        msg += "-"*73 + "\n"
        #for idx_dic, loc_dic in enumerate([self.loc_train, self.loc_valid, self.loc_test]):
            #itype = self.data_types [idx_dic]
        for itype in self.data_types:
            msg += "%s\n"%itype
            msg_list = [["", "MaxAE", "MAE", "ME", "MARE", "Pearson"]]
            for isystem in self.loc_dic[itype].keys():
                if isystem in self.ene_cal_dic["total"].keys():
                    #file_hfe = "/scratch_sh/ml_datasets/rhf_energy/rhfe_%s_%s.hdf5"%(isystem, self.basis)
                    if os.path.isfile(self.file_rhf_ene[isystem]):
                        file_hfe = h5py.File(self.file_rhf_ene[isystem], "r")
                    else:
                        warnings.warn("%s cannot be found"%self.file_rhf_ene[isystem])
                        file_hfe = None
                    ecal_list = []
                    epred_list = []
                    err_list = []
                    if "diag" in self.pairtype_list:
                        ptype = "diag"
                    elif "offdiag_close" in self.pairtype_list:
                        ptype = "offdiag_close"
                    elif len(self.pairtype_list) == 1:
                        ptype = "offdiag_remote"
                    for imol in self.loc_dic[itype][isystem][ptype].keys():
                        if file_hfe is not None:
                            erhf = file_hfe[imol][0]
                        else:
                            erhf = 0.0
                        ecal = self.ene_cal_dic["total"][isystem][imol]
                        epred = self.ene_pred_dic["total"][isystem][imol]
                        err = ecal - epred
                        msg_ene += "%s  %s  %s  %.8f  %.8f  %s\n"%(itype, isystem, imol, ecal*hartee2kcalmol, epred*hartee2kcalmol, "{:>12.4E}".format(err*hartee2kcalmol))
                        msg_ene_withhf +=  "%s  %s  %s  %.8f  %.8f  %s\n"%(itype, isystem, imol, (ecal+erhf)*hartee2kcalmol, (epred+erhf)*hartee2kcalmol, "{:>12.4E}".format(err*hartee2kcalmol))
                        ecal_list.append(ecal)
                        epred_list.append(epred)
                        err_list.append(err)
                    ecal_list = np.asarray(ecal_list)
                    epred_list = np.asarray(epred_list)
                    err_list = np.asarray(err_list)
                    maxae = max(np.abs(err_list))
                    mae = np.mean(np.abs(err_list))
                    me = np.mean(err_list)
                    mare = np.mean(np.abs(ecal_list-(epred_list+me)))
                    if len(ecal_list) > 1:
                        pearson = scipy.stats.pearsonr(ecal_list, epred_list)[0]
                    else:
                        pearson = 1
                    msg_list.append([isystem] + ["%.6f"%(num*hartee2kcalmol) for num in [maxae, mae, me, mare]] +  ["%.4f"%pearson])
                    #msg_list.append([isystem, "%.6f"%(maxe*hartee2kcalmol), "%.6f"%(mae*hartee2kcalmol), "%.4f"%pearson])
                    if file_hfe is not None:
                        file_hfe.close()
            msg += print_align(msg_list, "lrrrrr")
            msg += "-"*73 + "\n"
        with open("%s/ene_total.log"%(self.output_dir), "w") as file_ene:
            file_ene.write(msg_ene)
        with open("%s/ene_total_withhf.log"%(self.output_dir), "w") as file_ene_withhf:
            file_ene_withhf.write(msg_ene_withhf)
        print(msg)
        with open(self.file_summary, "a") as f:
            f.write(msg)
nn_pair = NN_MBE()
nn_pair.kernel()
