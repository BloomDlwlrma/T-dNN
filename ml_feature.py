import os
import sys
import numbers
import numpy as np
import h5py
from functools import reduce
class ml_feature():
    def __init__(self):
        #Prepare pair energies
        #self.int_type = os.environ.get("int_type")
        #self.basis = (os.environ.get("basis","ccpvdz").replace("_", "").replace("-", "")).lower()
        
        # 初始化和环境变量读取-决定后续特征提取的开关
        self.with_loc_j = bool(int(os.environ.get("with_loc_j", 1)))
        self.with_loc_k = bool(int(os.environ.get("with_loc_k", 1)))
        self.with_loc_f = bool(int(os.environ.get("with_loc_f", 1)))
        self.with_sratio = bool(int(os.environ.get("with_sratio", 0)))
        self.with_t2 = bool(int(os.environ.get("with_t2", 0)))
        self.with_fene = bool(int(os.environ.get("with_fene", 0)))
        self.with_fene_mat = bool(int(os.environ.get("with_fene_mat", 1)))
        self.sym_fenemat = bool(int(os.environ.get("sym_fenemat", 1)))
        self.with_kosv = bool(int(os.environ.get("with_kosv", 0)))
        self.with_sosv = bool(int(os.environ.get("with_sosv", 0)))
        self.with_fosv = bool(int(os.environ.get("with_fosv", 0)))
        if (True in [self.with_t2, self.with_fene, self.with_fene_mat]):
            self.gen_t2 = True
        else:
            self.gen_t2 = False
        self.t2_type = int(os.environ.get("t2_type", 1)) #can be 1:(get_t2), 2(get_t2_qjl), 3(get_t2_wpn)
        self.nosv_fea = int(os.environ.get("nosv_fea", 8))
        # 路径解析和文件名拆分 从文件名提取元数据
            # int_type：积分类型（e.g., MP2 integrals）。
            # basis：基组（e.g., cc-pVTZ）。
            # lmo_type：轨道局域化方法。
            # mol_system：分子数据集名称。
        self.path_raw = os.environ.get("path_raw")
        file_raw = self.path_raw.split("/")[-1]
        fsplit = file_raw.split("_")
        if len(fsplit) > 5 and \
            fsplit[-1].split(".")[0] in ["pm", "boys"]: #Standard format 
            
            int_type = fsplit[2]
            basis = fsplit[-2]
            lmo_type = fsplit[-1].split(".")[0]
            mol_system = ""
            for idx, i in enumerate(fsplit[3:-3]):
                if idx == 0:
                    mol_system += i
                else:
                    mol_system += "_%s"%i
        else:
            int_type = basis = lmo_type = mol_system = None
        self.int_type = os.environ.get("int_type", int_type)
        self.basis = os.environ.get("basis", basis)
        self.lmo_type = os.environ.get("lmo_type", lmo_type)
        self.mol_system = os.environ.get("mol_system", mol_system)
        if None in [int_type, basis, lmo_type, mol_system]:
            raise OSError("%s not specified"%[i for i in [int_type, basis, lmo_type, mol_system] if i is None])
        # 输出路径构建
        self.path_des = "descriptors/%s/"%(self.mol_system)
        self.path_des += "%s_%s_%d"%(self.int_type, self.lmo_type, self.nosv_fea)
        # 特征字符串构建
        features = ""
        if self.with_loc_j:
            features += "_locj"
        if self.with_loc_k:
            features += "_lock"
        if self.with_loc_f:
            features += "_locf"
        if self.with_sratio:
            features += "_sratio"
        if self.with_sosv:
            features += "_sosv"
        if self.with_fosv:
            features += "_fosv"
        if self.with_kosv:
            features += "_kosv"
        if self.with_t2:
            features += "_t2"
        if self.with_fene:
            features += "_fene"
        if self.with_fene_mat:
            if self.sym_fenemat:
                features += "_fenemat_sym"
            else:
                features += "_fenemat"
        if self.gen_t2:
            if self.t2_type == 2:
                features += "_qjl"
            elif self.t2_type == 3:
                features += "_wpn"
            elif self.t2_type == 4:
                features += "_wpnv2"

        if features == "":
            sys.exit()
        else:
            self.path_des += features
        self.path_des += "_%s"%self.basis
        # 目录创建和最终初始化，用于后续特征分组；
        os.makedirs(self.path_des, exist_ok=True)
        self.pairtype_list = ["diag", "offdiag_close", "offdiag_remote"]

    def unpack_vec(self, vec, dim_list, pairlist, is_remote, nocc, nosv_list, four_block=False):
        
        def get_blocks(mat, pidx, ipair):
            i = ipair//nocc
            mat = mat.reshape(dim_list[pidx])
            if (four_block == False) or (is_remote[ipair]):
                new_block = mat[:self.nosv_fea, :self.nosv_fea]
            else:
                dim_new = 2 * self.nosv_fea
                new_block = np.zeros((dim_new, dim_new))
                new_block[:self.nosv_fea, :self.nosv_fea] = mat[:self.nosv_fea, :self.nosv_fea] #tl
                new_block[:self.nosv_fea, self.nosv_fea:] = mat[:nosv_list[i], nosv_list[i]:][:self.nosv_fea, :self.nosv_fea] #tr
                new_block[self.nosv_fea:, :self.nosv_fea] = mat[nosv_list[i]:, :nosv_list[i]][:self.nosv_fea, :self.nosv_fea] #bl
                new_block[self.nosv_fea:, self.nosv_fea:] = mat[nosv_list[i]:, nosv_list[i]:][:self.nosv_fea, :self.nosv_fea] #br
            return new_block
        mat_list = [None]*len(dim_list)
        idx0 = 0
        for pidx, ipair in enumerate(pairlist):
            idx1 = idx0 + np.product(dim_list[pidx])
            mat_list[pidx] = get_blocks(vec[idx0:idx1].reshape(dim_list[pidx]), pidx, ipair)
            idx0 = idx1
        return mat_list

    def get_t2(self, nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, kosv_list):
        def t2_pair(pidx, ipair):
            i = ipair//nocc
            j = ipair%nocc
            eij = loc_f[i*nocc+i] + loc_f[j*nocc+j]

            e_iu = np.diag(fosv_list[pairidx_list[i*nocc+i]])
            e_jv = np.diag(fosv_list[pairidx_list[j*nocc+j]])
            if is_remote[ipair]:
                return kosv_list[pidx]/(eij-e_iu.reshape(-1,1)-e_jv)
            else:
                t2_full = np.copy(kosv_list[pidx])
                t2_full[:self.nosv_fea, :self.nosv_fea] /= (eij-e_iu.reshape(-1,1)-e_iu) #ii
                t2_full[:self.nosv_fea, self.nosv_fea:] /= (eij-e_iu.reshape(-1,1)-e_jv) #ij
                t2_full[self.nosv_fea:, :self.nosv_fea] /= (eij-e_jv.reshape(-1,1)-e_iu) #ji
                t2_full[self.nosv_fea:, self.nosv_fea:] /= (eij-e_jv.reshape(-1,1)-e_jv) #jj
                return t2_full
        t2_list = [None]*len(pairlist)
        for pidx, ipair in enumerate(pairlist):
            t2_list[pidx] = t2_pair(pidx, ipair)
        return t2_list
    
    def get_t2_qjl(self, nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, kosv_list):
        def t2_pair(pidx_ij, ipair):
            i = ipair//nocc
            j = ipair%nocc
            pidx_ii = pairidx_list[i*nocc+i]
            pidx_jj = pairidx_list[j*nocc+j]
            eij = loc_f[i*nocc+i] + loc_f[j*nocc+j]
            if is_remote[ipair]:
                return kosv_list[pidx_ij]/(eij-2*fosv_list[pidx_ij])
            else:
                t2_full = np.copy(kosv_list[pidx_ij])
                t2_full[:self.nosv_fea, :self.nosv_fea] /= (eij-2*fosv_list[pidx_ii]) #ii
                t2_full[:self.nosv_fea, self.nosv_fea:] /= (eij-2*fosv_list[pidx_ij]) #ij
                t2_full[self.nosv_fea:, :self.nosv_fea] /= (eij-2*fosv_list[pidx_ij].T) #ji
                t2_full[self.nosv_fea:, self.nosv_fea:] /= (eij-2*fosv_list[pidx_jj]) #jj
                return t2_full
        t2_list = [None]*len(pairlist)
        for pidx, ipair in enumerate(pairlist):
            t2_list[pidx] = t2_pair(pidx, ipair)
        return t2_list
    def get_t2_wpn_v2(self, nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, sosv_list, kosv_list): 
        def t2_pair(pidx, ipair):
            i = ipair//nocc
            j = ipair%nocc
            eij = loc_f[i*nocc+i] + loc_f[j*nocc+j]

            e_iu = np.diag(fosv_list[pairidx_list[i*nocc+i]])
            e_jv = np.diag(fosv_list[pairidx_list[j*nocc+j]])
            if is_remote[ipair]:
                return kosv_list[pidx]/(eij-e_iu.reshape(-1,1)-e_jv) 
            else:
                t2_full = np.copy(kosv_list[pidx])
                t2_full[:self.nosv_fea, :self.nosv_fea] /= (eij-e_iu.reshape(-1,1)-e_iu) #ii
                t2_full[self.nosv_fea:, self.nosv_fea:] /= (eij-e_jv.reshape(-1,1)-e_jv) #jj

                if i!=j:
                    FTS = reduce(np.dot,[fosv_list[pidx],t2_full[self.nosv_fea:, :self.nosv_fea]/(eij-e_jv.reshape(-1,1)-e_iu),sosv_list[pidx]])
                    STF = reduce(np.dot,[sosv_list[pidx],t2_full[self.nosv_fea:, :self.nosv_fea]/(eij-e_jv.reshape(-1,1)-e_iu),fosv_list[pidx]])
                    STS= eij*reduce(np.dot,[sosv_list[pidx],t2_full[self.nosv_fea:, :self.nosv_fea]/(eij-e_jv.reshape(-1,1)-e_iu),sosv_list[pidx]])
                    # build advanced DC, EC
                    t2_full[:self.nosv_fea, self.nosv_fea:] = (t2_full[:self.nosv_fea, self.nosv_fea:] + FTS + STF - STS)/(eij-e_iu.reshape(-1,1)-e_jv) # DC
                    FTS = None
                    STF = None
                    STS = None
                    FTS = reduce(np.dot,[fosv_list[pidx].T,t2_full[:self.nosv_fea, self.nosv_fea:]/(eij-e_iu.reshape(-1,1)-e_jv),sosv_list[pidx].T])
                    STF = reduce(np.dot,[sosv_list[pidx].T,t2_full[:self.nosv_fea, self.nosv_fea:]/(eij-e_iu.reshape(-1,1)-e_jv),fosv_list[pidx].T])
                    STS= eij*reduce(np.dot,[sosv_list[pidx].T,t2_full[:self.nosv_fea, self.nosv_fea:]/(eij-e_iu.reshape(-1,1)-e_jv),sosv_list[pidx].T])

                    t2_full[self.nosv_fea:, :self.nosv_fea] = (t2_full[self.nosv_fea:, :self.nosv_fea] + FTS + STF - STS)/(eij-e_jv.reshape(-1,1)-e_iu)
                else:
                    t2_full[:self.nosv_fea, self.nosv_fea:] = t2_full[:self.nosv_fea, self.nosv_fea:]/(eij-e_iu.reshape(-1,1)-e_jv) # DC
                    t2_full[self.nosv_fea:, :self.nosv_fea] = t2_full[self.nosv_fea:, :self.nosv_fea]/(eij-e_jv.reshape(-1,1)-e_iu) # EC

                return t2_full
        t2_list = [None]*len(pairlist)
        for pidx, ipair in enumerate(pairlist):
            t2_list[pidx] = t2_pair(pidx, ipair)
        return t2_list
    def get_t2_wpn(self, nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, sosv_list, kosv_list): 
        def t2_pair(pidx, ipair): 
            i = ipair//nocc
            j = ipair%nocc
            eij = loc_f[i*nocc+i] + loc_f[j*nocc+j]

            e_iu = np.diag(fosv_list[pairidx_list[i*nocc+i]])
            e_jv = np.diag(fosv_list[pairidx_list[j*nocc+j]])
            if is_remote[ipair]:
                return kosv_list[pidx]/(eij-e_iu.reshape(-1,1)-e_jv) 
            else:
                t2_full = np.copy(kosv_list[pidx])
                t2_full[:self.nosv_fea, :self.nosv_fea] /= (eij-e_iu.reshape(-1,1)-e_iu) #ii
                t2_full[self.nosv_fea:, self.nosv_fea:] /= (eij-e_jv.reshape(-1,1)-e_jv) #jj

                FS = fosv_list[pidx]*sosv_list[pidx]
                S2 = eij*sosv_list[pidx]**2
                correction = 2*FS-S2
                # build advanced DC, EC
                t2_full[:self.nosv_fea, self.nosv_fea:] = (t2_full[:self.nosv_fea, self.nosv_fea:] + correction*(t2_full[self.nosv_fea:, :self.nosv_fea]/(eij-e_jv.reshape(-1,1)-e_iu)).T)/(eij-e_iu.reshape(-1,1)-e_jv)
                t2_full[self.nosv_fea:, :self.nosv_fea] = (t2_full[self.nosv_fea:, :self.nosv_fea] + (correction*t2_full[:self.nosv_fea, self.nosv_fea:]/(eij-e_iu.reshape(-1,1)-e_jv)).T)/(eij-e_jv.reshape(-1,1)-e_iu)

                return t2_full
        t2_list = [None]*len(pairlist)
        for pidx, ipair in enumerate(pairlist):
            t2_list[pidx] = t2_pair(pidx, ipair)
        return t2_list

    def get_fene(self, nocc, pairlist, is_remote, t2_list, kosv_list): 
        def symmetrize(mat,remote=False): 
            id_upper = np.triu_indices(self.nosv_fea)
            id_nodiag = np.triu_indices(self.nosv_fea,k=1)
            def sym_type1(mat):
                mat_sym_1 = 0.5*(mat+mat.T)

                mat_sym_2 = 0.5*np.abs((mat-mat.T))
                mat = np.append(mat_sym_1[id_upper],mat_sym_2[id_nodiag])
                mat = mat.reshape(self.nosv_fea,self.nosv_fea)
                return mat
            def sym_type2(mat1,mat2):
                mat1s = 0.5*(mat1+mat2.T)
                mat2s = 0.5*np.abs((mat1-mat2.T))
                return mat1s,mat2s
            if remote:
                mat = sym_type1(mat)
                return mat
            if isinstance(mat,list): 
                mat1,mat2 = mat
                mat1,mat2 = sym_type2(mat1,mat2)
                return mat1, mat2
            else: 
                mat = sym_type1(mat)
                return mat

        fene_mat = [None] * len(t2_list)
        fene = [None] * len(t2_list)
        for pidx, ipair in enumerate(pairlist):
            i = ipair//nocc
            j = ipair%nocc
            if is_remote[ipair]: 
                fene_mat[pidx] = (2 * t2_list[pidx]) * kosv_list[pidx]
            else:
                fene_mat[pidx] = (2 * t2_list[pidx] - t2_list[pidx].T) * kosv_list[pidx]
            fene[pidx] = np.sum(fene_mat[pidx])
            if i != j:
                fene[pidx] *= 2
            if i==j: 
                id_upper = np.triu_indices(self.nosv_fea) 
                fene_mat[pidx] = fene_mat[pidx][:self.nosv_fea,:self.nosv_fea][id_upper] 
            if self.sym_fenemat: 
                if i != j:
                    if is_remote[ipair]:
                        fene_mat[pidx] = symmetrize(fene_mat[pidx],remote=True)
                    else:
                        fene_mat[pidx][:self.nosv_fea,self.nosv_fea:] = symmetrize(fene_mat[pidx][:self.nosv_fea,self.nosv_fea:]) # DC
                        fene_mat[pidx][self.nosv_fea:,:self.nosv_fea] = symmetrize(fene_mat[pidx][self.nosv_fea:,:self.nosv_fea]) # EC
                        fene_mat[pidx][:self.nosv_fea,:self.nosv_fea],fene_mat[pidx][self.nosv_fea:,self.nosv_fea:] = symmetrize([fene_mat[pidx][:self.nosv_fea,:self.nosv_fea],fene_mat[pidx][self.nosv_fea:,self.nosv_fea:]])
        return fene_mat, fene

    def add_feature(self, fea_dic, feature_list, nocc, pairlist, is_remote, 
                     pairidx_list=None, with_ii_jj=False, exclude_diag=False): 
        def flatten(val): 
            if isinstance(val, numbers.Number):
                return val
            else:
                return val.ravel()
        pidx_diag = pidx_offdiagc = pidx_offdiagr = 0
        for pidx, ipair in enumerate(pairlist):
            i = ipair//nocc
            j = ipair%nocc
            if pairidx_list is not None:
                idx = pidx
            else:
                idx = ipair
            feature = flatten(feature_list[idx])
            if i == j: 
                if exclude_diag == False:
                    fea_dic["diag"][pidx_diag] = np.append(fea_dic["diag"][pidx_diag], feature)
                pidx_diag += 1
            else: 
                if with_ii_jj: 
                    pair_diag = [i*nocc+i, j*nocc+j] 
                    if pairidx_list is not None:
                        idx_diag = np.append(pairidx_list[pair_diag[0]],pairidx_list[pair_diag[1]])
                    else:
                        idx_diag = pair_diag
                    #print(idx_diag)
                    feature_ii_jj = feature_list[idx_diag]

                    if self.sym_fenemat:
                        sym1 = 1/2*(feature_ii_jj[0]+feature_ii_jj[1])
                        sym2 = 1/2*np.abs(feature_ii_jj[0]-feature_ii_jj[1])
                        feature_ii_jj = np.asarray([sym1,sym2])

                    if is_remote[ipair]:
                        fea_dic["offdiag_remote"][pidx_offdiagr] = np.append(fea_dic["offdiag_remote"][pidx_offdiagr], feature_ii_jj)
                    else:
                        fea_dic["offdiag_close"][pidx_offdiagc] = np.append(fea_dic["offdiag_close"][pidx_offdiagc], feature_ii_jj)

                if is_remote[ipair]:
                    fea_dic["offdiag_remote"][pidx_offdiagr] = np.append(fea_dic["offdiag_remote"][pidx_offdiagr], feature)
                    pidx_offdiagr += 1
                else:
                    fea_dic["offdiag_close"][pidx_offdiagc] = np.append(fea_dic["offdiag_close"][pidx_offdiagc], feature)
                    pidx_offdiagc += 1
                
        return fea_dic
    
    # 特征分组基于分子对类型，将每个分子的特征分配到三个类别。分组在 kernel 方法中初始化，并在 add_feature 中填充。
    def kernel(self):
        with h5py.File("%s/descriptors.hdf5"%self.path_des, "w") as fdes:
            with h5py.File(self.path_raw, "r") as fraw: # self.path_raw = os.environ.get("path_raw")
                print(fraw.keys())
                for imol in fraw.keys():
                    if imol == 'water02':
                        print("Yes")
                    pairlist = fraw[imol]["pairlist"][:]
                    pairlist_remote = fraw[imol]["pairlist_screened"][:]
                    nocc = fraw[imol]["nocc"][0]
                    '''nosv = fraw[imol]["nosv(orb_list)"][:]
                    orb_list = fraw[imol]["orb_list"][:]
                    nosv_list = [None]*nocc
                    for idx, i in enumerate(orb_list):
                        nosv_list[i] = nosv[idx]'''

                    is_remote = [False] * (nocc**2)
                    for ipair in pairlist_remote:
                        is_remote[ipair] = True

                    #Initialize the feature dictionary 分组初始化
                    sf_dim = fraw[imol]["sf_osv_dim(pairlist)"][:]
                    nosv_list = [None]*nocc
                    
                    fea_dic = {}
                    for ptype in self.pairtype_list:
                        fea_dic[ptype] = []
                    pairidx_list = [None] * (nocc**2)
                    for pidx, ipair in enumerate(pairlist):
                        i = ipair//nocc
                        j = ipair%nocc
                        # print("i j", i, j)
                        nosv_list[i], nosv_list[j] = sf_dim[pidx]
                        pairidx_list[ipair] = pidx
                        if i == j:
                            fea_dic["diag"].append([])
                        elif is_remote[ipair]:
                            fea_dic["offdiag_remote"].append([])
                        else:
                            fea_dic["offdiag_close"].append([])

                    print([nosv for nosv in nosv_list if nosv is not None])
                    if imol == 'water02':
                            print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0])
                    if self.with_sratio:
                        s_ratio = fraw[imol]["s_ratio(nocc,nocc)"][:].ravel()
                        fea_dic = self.add_feature(fea_dic, s_ratio, nocc, pairlist, is_remote)
                        if imol == 'water02':
                            len_fea = len(fea_dic["diag"])
                            
                            print("Length of feature", len_fea)
                            print("Length of feature list 0, ", fea_dic["diag"][0].size)
                            print("Content of feature list 0, ", fea_dic["diag"][0])
                            print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)

                    if self.with_loc_j:
                        loc_j = fraw[imol]["Coulomb(pairlist)"][:]
                        fea_dic = self.add_feature(fea_dic, loc_j, nocc, pairlist, is_remote, pairidx_list, with_ii_jj=True)
                        if imol == 'water02':
                            print("with_loc_j")
                            print("Type of feature", type(fea_dic["diag"]))
                            print("Type of feature list 0", type(fea_dic["diag"][0]))
                            print("Length of feature list 0, ", fea_dic["diag"][0].size)
                            print("Content of feature list 0, ", fea_dic["diag"][0])
                            print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)
                    if self.with_loc_k:
                        loc_k = fraw[imol]["Exchange(pairlist)"][:]
                        if self.with_loc_j: 
                            with_ii_jj = False
                            exclude_diag = True
                        else:
                            with_ii_jj = True
                            exclude_diag = False

                        fea_dic = self.add_feature(fea_dic, loc_k, nocc, pairlist, is_remote, pairidx_list, with_ii_jj=with_ii_jj, exclude_diag=exclude_diag)
                        if imol == 'water02':
                            print("with_loc_k")
                            print("Length of feature list 0, ", fea_dic["diag"][0].size)
                            # print("Content of feature list 0, ", fea_dic["diag"][0])
                            print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)
                    if self.with_loc_f or self.gen_t2:
                        loc_f = fraw[imol]["loc_fock(nocc,nocc)"][:].ravel()
                        print("loc_f, type, shape", type(loc_f), loc_f.shape)
                        if self.with_loc_f:
                            fea_dic = self.add_feature(fea_dic, loc_f, nocc, pairlist, is_remote, with_ii_jj=True)
                            if imol == 'water02':
                                print("with_loc_f")
                                print("Length of feature list 0, ", fea_dic["diag"][0].size)
                                print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)
                    #matrix
                    if self.with_sosv or (self.gen_t2 and self.t2_type == 3 or self.gen_t2 and self.t2_type ==4):
                        smat_osv = fraw[imol]["Smat_osv(pairlist)"][:]
                        sosv_dim = fraw[imol]["sf_osv_dim(pairlist)"][:]
                        sosv_list = self.unpack_vec(smat_osv, sosv_dim, pairlist, is_remote, nocc, nosv_list)
                        if self.with_sosv:
                            fea_dic = self.add_feature(fea_dic, sosv_list, nocc, pairlist, is_remote, pairidx_list)
                    if self.with_fosv or self.gen_t2:
                        fmat_osv = fraw[imol]["Fmat_osv(pairlist)"][:]
                        fosv_dim = fraw[imol]["sf_osv_dim(pairlist)"][:]
                        fosv_list = self.unpack_vec(fmat_osv, fosv_dim, pairlist, is_remote, nocc, nosv_list)
                        if (self.with_fosv):
                            fea_dic = self.add_feature(fea_dic, fosv_list, nocc, pairlist, is_remote, pairidx_list)

                    if self.with_kosv or self.gen_t2:
                        kmat_osv = fraw[imol]["Kmat_osv(pairlist)"][:]
                        kosv_dim = fraw[imol]["kmat_osv_dim(pairlist)"][:]
                        kosv_list = self.unpack_vec(kmat_osv, kosv_dim, pairlist, is_remote, nocc, nosv_list, four_block=True)
                        if self.with_kosv:
                            fea_dic = self.add_feature(fea_dic, kosv_list, nocc, pairlist, is_remote, pairidx_list)
                    
                    if self.gen_t2:
                        if self.t2_type == 2:
                            t2_list = self.get_t2_qjl(nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, kosv_list)
                        elif self.t2_type == 3:
                            t2_list = self.get_t2_wpn(nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, sosv_list, kosv_list)
                        elif self.t2_type == 4:
                            t2_list = self.get_t2_wpn_v2(nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, sosv_list, kosv_list)
                        else:
                            t2_list = self.get_t2(nocc, pairlist, pairidx_list, is_remote, loc_f, fosv_list, kosv_list)
                        
                        if self.with_t2:
                            fea_dic = self.add_feature(fea_dic, t2_list, nocc, pairlist, is_remote, pairidx_list)
                            if imol == 'water02':
                                print("with_t2")
                                print("Length of feature list 0, ", fea_dic["diag"][0].size)
                                print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)
                        if self.with_fene_mat or self.with_fene:
                            fenemat_list, fene_list = self.get_fene(nocc, pairlist, is_remote, t2_list, kosv_list)
                            if self.with_fene_mat:
                                fea_dic = self.add_feature(fea_dic, fenemat_list, nocc, pairlist, is_remote, pairidx_list)
                                if imol == 'water02':
                                    print("with_fene_mat")
                                    print("Length of feature list 0, ", fea_dic["diag"][0].size)
                                    print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)
                            if self.with_fene:
                                fea_dic = self.add_feature(fea_dic, fene_list, nocc, pairlist, is_remote, pairidx_list)
                                if imol == 'water02':
                                    print("with_fene")
                                    print("Length of feature list 0, ", fea_dic["diag"][0].size)
                                    print("Length of feature list 0, off_diag ", fea_dic["offdiag_close"][0].size)

                    for ptype in fea_dic.keys():
                        if fea_dic[ptype] != []:
                            ifea = np.asarray(fea_dic[ptype])
                            fdes["%s/%s"%(ptype, imol)] = ifea
                            print(ptype, ifea.shape)
                    
my_feature = ml_feature()     
my_feature.kernel()
