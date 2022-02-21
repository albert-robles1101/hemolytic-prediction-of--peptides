#Cálculo de los descriptores físico - químicos de peptidos de HAPPEN y HemoPi
#30 de mayo de 2021

import pandas as pd
import os
import numpy as np
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
os. chdir("C:/Users/alber/Documents/")
def calc_PFQ(dataset): 
    #dataset es un parametro tipo string que especifica el nombre del archivo .xlsx
    #de peptidos
    db = pd.read_excel(dataset+'.xlsx')
    db_seq = np.array(db.iloc[:,2])

    pepdesc_total = PeptideDescriptor(db_seq, 'eisenberg')
    globdesc_total = GlobalDescriptor(db_seq)
    pepdesc_total.load_scale('eisenberg')
    pepdesc_total.calculate_global()  # calculate global Eisenberg hydrophobicity
    pepdesc_total.calculate_moment(append=True)
    pepdesc_total.load_scale('gravy')  # load GRAVY scale
    pepdesc_total.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
    pepdesc_total.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment
    pepdesc_total.load_scale('z3')  # load old Z scale
    pepdesc_total.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)
    pepdesc_total.load_scale('z5')  # load old Z scale
    pepdesc_total.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)
    pepdesc_total.load_scale('AASI')
    pepdesc_total.calculate_global(append=True)  # calculate global AASI index
    pepdesc_total.calculate_moment(append=True)  # calculate AASI index moment
    pepdesc_total.load_scale('ABHPRK')
    pepdesc_total.calculate_global(append=True)  # calculate ABHPRK feature 
    pepdesc_total.load_scale('argos')
    pepdesc_total.calculate_global(append=True)  # calculate global argos index
    pepdesc_total.calculate_moment(append=True)  # calculate argos index moment
    pepdesc_total.load_scale('bulkiness')
    pepdesc_total.calculate_global(append=True)  # calculate global bulkiness index
    pepdesc_total.calculate_moment(append=True)  # calculate bulkiness index moment
    pepdesc_total.load_scale('charge_phys')
    pepdesc_total.calculate_global(append=True)  # calculate global charge_phys index
    pepdesc_total.load_scale('charge_acid')
    pepdesc_total.calculate_global(append=True)  # calculate global charge_acid index
    pepdesc_total.load_scale('Ez')
    pepdesc_total.calculate_global(append=True)  # calculate global energies of insertion of amino acid side chains into lipid bilayers index
    pepdesc_total.load_scale('flexibility')
    pepdesc_total.calculate_global(append=True)  # calculate global flexibility scale
    pepdesc_total.calculate_moment(append=True)  # calculate flexibility moment
    pepdesc_total.load_scale('grantham')
    pepdesc_total.calculate_global(append=True)  # calculate global amino acid side chain composition, polarity and molecular volume
    pepdesc_total.load_scale('hopp-woods')
    pepdesc_total.calculate_global(append=True)  # calculate global Hopp-Woods hydrophobicity scale
    pepdesc_total.calculate_moment(append=True)  # calculate Hopp-Woods hydrophobicity moment
    pepdesc_total.load_scale('ISAECI')
    pepdesc_total.calculate_global(append=True) # calculate global ISAECI (Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains) index
    pepdesc_total.load_scale('janin')
    pepdesc_total.calculate_global(append=True)  # calculate global Janin hydrophobicity scale
    pepdesc_total.calculate_moment(append=True)  # calculate Janin hydrophobicity moment
    pepdesc_total.load_scale('kytedoolittle')
    pepdesc_total.calculate_global(append=True)  # calculate global Kyte & Doolittle hydrophobicity scale
    pepdesc_total.calculate_moment(append=True)  # calculate Kyte & Doolittle hydrophobicity moment
    pepdesc_total.load_scale('levitt_alpha')
    pepdesc_total.calculate_global(append=True)  # calculate global Levitt alpha-helix propensity scale
    pepdesc_total.calculate_moment(append=True)  # calculate Levitt alpha-helix propensity moment
    pepdesc_total.load_scale('MSS')
    pepdesc_total.calculate_global(append=True)  # calculate global MSS index, graph-theoretical index that reflects topological shape and size of amino acid side chains
    pepdesc_total.calculate_moment(append=True)  # calculate MSS moment
    pepdesc_total.load_scale('MSW')
    pepdesc_total.calculate_global(append=True)  # calculate global MSW scale, Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids
    pepdesc_total.load_scale('pepArc')
    pepdesc_total.calculate_global(append=True) # calculate global pepArc, modlabs pharmacophoric feature scale, dimensions are: hydrophobicity, polarity, positive charge, negative charge, proline.
    pepdesc_total.load_scale('pepcats')
    pepdesc_total.calculate_global(append=True) # calculate global pepcats, modlabs pharmacophoric feature based PEPCATS scale
    pepdesc_total.load_scale('polarity')
    pepdesc_total.calculate_global(append=True)  # calculate global AA polarity 
    pepdesc_total.calculate_moment(append=True)  # calculate AA polarity moment
    pepdesc_total.load_scale('PPCALI')
    pepdesc_total.calculate_global(append=True)  # calculate global modlabs inhouse scale derived from a PCA of 143 amino acid property scales
    pepdesc_total.load_scale('refractivity')
    pepdesc_total.calculate_global(append=True) # calculate global relative AA refractivity
    pepdesc_total.calculate_moment(append=True) # calculate relative AA refractivity moment
    pepdesc_total.load_scale('t_scale')
    pepdesc_total.calculate_global(append=True) # calculate global t scale, A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program
    pepdesc_total.load_scale('TM_tend')
    pepdesc_total.calculate_global(append=True) # calculate global Amino acid transmembrane propensity scale
    pepdesc_total.calculate_moment(append=True) # calculate Amino acid transmembrane propensity scale moment
    col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3, Z5_1,Z5_2,Z5_3, Z5_4,Z5_5,S_AASI, uS_AASI, modlas_ABHPRK, H_argos, uH_argos, B_Builkiness, uB_Builkiness, charge_phys, charge_acid, Ez, flexibility, u_flexibility, Grantham, H_HoppWoods, uH-HoppWoods, ISAECI, H_Janin, uH_Janin, H_KyteDoolittle, uH_KyteDoolittle, F_Levitt, uF_Levitt, MSS_shape, u_MSS_shape, MSW, pepArc, pepcats, polarity, u_polarity, PPCALI, refractivity, u_refractivity, t_scale, TM_tend, u_TM_tend'
    pepdesc_total.save_descriptor('C:/Users/alber/Documents/pep_descriptors_'+ dataset + '.csv', header=col_names1)

    globdesc_total.length()  # sequence length
    globdesc_total.boman_index(append=True)  # Boman index
    globdesc_total.aromaticity(append=True)  # global aromaticity
    globdesc_total.aliphatic_index(append=True)  # aliphatic index
    globdesc_total.instability_index(append=True)  # instability index
    globdesc_total.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
    globdesc_total.calculate_MW(amide=False, append=True)  # molecular weight
    globdesc_total.isoelectric_point(amide=False,append=True) # isoelectric point
    globdesc_total.hydrophobic_ratio(append=True)
#globdesc_totalAPD.charge_density(ph=7.4, amide=False, append=True) # charge density (does not work)

    col_names2 = 'ID, Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex, NetCharge, MW, IsoelectricPoint, HydrophobicRatio'
    globdesc_total.save_descriptor('C:/Users/alber/Documents/global_descriptors_'+ dataset + '.csv', header=col_names2)
    


#############################################################################
calc_PFQ('HAPPENN_7_35_dataset')

calc_PFQ('HemoPi1_train_7_35_dataset')
calc_PFQ('HemoPi2_train_7_35_dataset')
calc_PFQ('HemoPi3_train_7_35_dataset')

calc_PFQ('HemoPi1_test_7_35_dataset')
calc_PFQ('HemoPi2_test_7_35_dataset')
calc_PFQ('HemoPi3_test_7_35_dataset')
