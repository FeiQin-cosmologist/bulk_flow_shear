#==============================================================================
#
#                                INTRODUCTION
#
#------------------------------------------------------------------------------


# python /Users/fei/WSP/Scie/Proj5/Prog/Part2/1\ 2MTF/7\ BQ\ wMLE\ 2MTF.py

# calculate cosmic flows of the CF4 tully-fisher durvey.



import math
import numpy as np
from scipy.linalg import lapack
from CosmoFunc import *
import matplotlib.pyplot as plt
from BF_OptMC  import *






#==============================================================================
#
#                                     CODE
#
#------------------------------------------------------------------------------
# 1. INITIAL SETTING:

    



OmegaM= 0.3121  
OmegaA= 1.0-OmegaM  
Hub   = 100.

mkSET   ='logd'
Fit_tech='etaMLE'
CFtype  ='BQ'
QzzType ='fix'
NmcSamp =[800,2500]
 

input_dir_mock   = '/Users/fei/WSP/Scie/Proj7/Data/Orig/SDSS/SDSSmocks/SDSS_'
input_dir_mockbc = '/Users/fei/WSP/Scie/Proj7/Data/Orig/SDSS/SDSSmocks/SDSS_'
output_dir       = '/Users/fei/WSP/Scie/Proj7/Data/Prod/SDSS/SDSS_'















print(mkSET,Fit_tech,CFtype,QzzType )
# 2: calculate and read in the true bulk flow which is known from similations--
Imock=120
Jmock=8
#cutval=mkSET * Imock
Nmock=Jmock*Imock
Ux_true=np.zeros( Nmock ) ; Uy_true=np.zeros( Nmock ) ; Uz_true=np.zeros( Nmock )
if(CFtype=='BQ'):
    Qxx_true= np.empty(Nmock) ; Qxy_true= np.empty(Nmock) ; Qxz_true= np.empty(Nmock) ; Qyy_true= np.empty(Nmock) ; Qyz_true= np.empty(Nmock) ; Qzz_true= np.empty(Nmock)
i = 0
for i_mock in range(Imock):
  i_mock=i_mock+19000+136  #+ cutval
  for j_mock in range(Jmock):
    input_dir = input_dir_mock+str(i_mock)+'.'+str(j_mock)
    infil= np.loadtxt(input_dir) ; vx = infil[:,13] ; vy = infil[:,14] ; vz = infil[:,15]
    if(CFtype=='BF'):
        Ux_true[i]  = np.sum(vx)/np.sum(np.shape(vx))  ;    Uy_true[i]  = np.sum(vy)/np.sum(np.shape(vy))  ;   Uz_true[i]  = np.sum(vz)/np.sum(np.shape(vz))
    if(CFtype=='BQ'):
        ra = infil[:,1] ; dec = infil[:,2] ; cz = infil[:,4]
        hatr  = hatr_Fun(ra,dec) ; 
        DistsR= DRcon(cz/LightSpeed,'z2d',OmegaM,OmegaA,Hub)
        '''
        Ux_true[i]  = np.sum(vx)/np.sum(np.shape(vx))  ;    Uy_true[i]  = np.sum(vy)/np.sum(np.shape(vy))  ;   Uz_true[i]  = np.sum(vz)/np.sum(np.shape(vz))
        hatr  = hatr.T
        Vel=np.sqrt(vx**2+vy**2+vz**2) ; ra_V = 180.0/math.pi*np.arctan2(vy, vx)+180.0 ; dec_V= 180.0/math.pi*np.arcsin(vz/Vel)
        hatv=np.array([[0.]*3]*np.sum(np.shape(ra))) ; hatv[:,0]=np.cos(dec_V/180.0*math.pi)*np.cos(ra_V/180.0*math.pi);hatv[:,1]=np.cos(dec_V/180.0*math.pi)*np.sin(ra_V/180.0*math.pi);hatv[:,2]=np.sin(dec_V/180.0*math.pi)
        # you should divide s*r_i*r_j by r: s*r_i*r_j/r to get the quadrupole term Qua:
        Qua=np.zeros((3,3)) ; HI=np.zeros((3,3))
        Qua[0,0]=np.sum(Vel*hatv[:,0]*hatr[:,0]/DistsR)/np.sum(np.shape(Vel)) ; Qua[1,1]=np.sum(Vel*hatv[:,1]*hatr[:,1]/DistsR)/np.sum(np.shape(Vel))
        Qua[2,2]=np.sum(Vel*hatv[:,2]*hatr[:,2]/DistsR)/np.sum(np.shape(Vel)) ; Qua[0,1]=np.sum(Vel*hatv[:,0]*hatr[:,1]/DistsR)/np.sum(np.shape(Vel))
        Qua[0,2]=np.sum(Vel*hatv[:,0]*hatr[:,2]/DistsR)/np.sum(np.shape(Vel)) ; Qua[1,2]=np.sum(Vel*hatv[:,1]*hatr[:,2]/DistsR)/np.sum(np.shape(Vel))
        Qua[1,0]=Qua[0,1] ; Qua[2,0]=Qua[0,2] ; Qua[2,1]=Qua[1,2]
        # remove the trace part of Qua to get the trace less Qij:
        HI[0,0]=1./3.*np.trace(Qua) ; HI[0,1]=0. ; HI[0,2]=0. ; HI[1,0]=0.
        HI[1,1]=1./3.*np.trace(Qua) ; HI[1,2]=0. ; HI[2,0]=0. ; HI[2,1]=0.
        HI[2,2]=1./3.*np.trace(Qua) ; Qij=Qua-HI   
        Qxx_true[i]=Qij[0,0] ; Qxy_true[i]=-Qij[0,1] ; Qxz_true[i]=-Qij[0,2] ; Qyy_true[i]=-Qij[1,1] ; Qyz_true[i]=-Qij[1,2] ; Qzz_true[i]=Qij[2,2] 
        '''
        ERRs2= 1.*np.ones(len(ra))#sigma_Vpec*sigma_Vpec+sigma_star*sigma_star
        Awei,AIJ=Aij_inv(ERRs2,DistsR,hatr) 
        Vel=vx*hatr[0,:] + vy*hatr[1,:] + vz*hatr[2,:]    
        Qua=np.zeros((3,3)) ; HI=np.zeros((3,3))
        Qua[0,0]=np.sum(Awei[3,:]*Vel); Qua[1,1]=np.sum(Awei[6,:]*Vel)
        Qua[2,2]=np.sum(Awei[8,:]*Vel) ; Qua[0,1]=np.sum(Awei[4,:]*Vel)
        Qua[0,2]=np.sum(Awei[5,:]*Vel) ; Qua[1,2]=np.sum(Awei[7,:]*Vel) 
        Qua[1,0]=Qua[0,1] ; Qua[2,0]=Qua[0,2] ; Qua[2,1]=Qua[1,2]
        # remove the trace part of Qua to get the trace less Qij:
        HI[0,0]=1./3.*np.trace(Qua) ; HI[0,1]=0. ; HI[0,2]=0. ; HI[1,0]=0.
        HI[1,1]=1./3.*np.trace(Qua) ; HI[1,2]=0. ; HI[2,0]=0. ; HI[2,1]=0.
        HI[2,2]=1./3.*np.trace(Qua) ; Qij=Qua-HI   
        Qxx_true[i]=Qij[0,0] ; Qxy_true[i]=Qij[0,1] ; Qxz_true[i]=Qij[0,2] ; 
        Qyy_true[i]=Qij[1,1] ; Qyz_true[i]=Qij[1,2] ; Qzz_true[i]=Qij[2,2] 
        Ux_true[i]=np.sum(Awei[0,:]*Vel) ; 
        Uy_true[i]=np.sum(Awei[1,:]*Vel) ;  
        Uz_true[i]=np.sum(Awei[2,:]*Vel) ;
        
    i = i + 1


















#=================================      MLE   =================================
# 3. MLE: 
print(Fit_tech,CFtype,QzzType)
# 3.1 the output of optimization and MCMC
B_opt=np.zeros((3,Nmock))  ;  Sv_opt=np.zeros(Nmock)  ;  B_MC =np.zeros((3,Nmock))  ;  Sv_MC =np.zeros(Nmock)
# 3.2 some arrays needed to calculate Chi^2
U_data = np.empty(3*Nmock);U_true = np.empty(3*Nmock);U_cov = np.empty((Nmock,3,3))
if(CFtype=='BQ'):
  if(QzzType== 'fix'):  
    Q_opt=np.zeros((5,Nmock))  ;  Q_MC =np.zeros((5,Nmock))  ;  eQzz=np.zeros(Nmock)
    Q_data = np.empty(5*Nmock) ;  Q_true = np.empty(5*Nmock) ;  Q_cov = np.empty((Nmock,5,5))
    Tot_data = np.empty(8*Nmock) ;  Tot_true = np.empty(8*Nmock) ;  Tot_cov= np.empty((Nmock,8,8))
  if(QzzType== 'free'):
    Q_opt=np.zeros((6,Nmock))  ;  Q_MC =np.zeros((6,Nmock))  ;  eQzz=np.zeros(Nmock)
    Q_data = np.empty(6*Nmock) ;  Q_true = np.empty(6*Nmock) ;  Q_cov = np.empty((Nmock,6,6))
    Tot_data = np.empty(9*Nmock) ;  Tot_true = np.empty(9*Nmock) ;  Tot_cov= np.empty((Nmock,9,9))  
# 3.3 read in the mocks or survey data
i = 0
mockID = np.zeros(Nmock)
for i_mock in range(Imock):
  i_mock=i_mock+19000 +136#+ cutval
  for j_mock in range(Jmock):  
    print(str(i_mock)+'.'+str(j_mock) )
    input_dir = input_dir_mockbc+str(i_mock)+'.'+str(j_mock) 
    mockID[i] = float(str(i_mock)+'.'+str(j_mock))
    infile=np.loadtxt(input_dir)
    ra_sdssm      = infile[:,1] ; dec_sdssm      = infile[:,2] ; z_sdssm = infile[:,4]/LightSpeed ;
    alpha_sdssm   = infile[:,20]
    if(mkSET=='logd'):
        logd_sdssm    = infile[:,18]; elogd_sdssm   = infile[:,19] 
    if(mkSET=='logdcorr'):
        logd_sdssm    = infile[:,21]; elogd_sdssm   = infile[:,22]
    if(mkSET=='logdt'):
        logd_sdssm    = infile[:,17]; elogd_sdssm   = infile[:,19]    
# 3.4 calculate cosmicflow using opt and mcmc:
    if(Fit_tech=='qMLE' ):        
        if(CFtype=='BF'):
            sigma_star_adjustor=10000.0
            B_opt[:,i],Sv_opt[i]                    = Opt_qMLE(ra_sdssm,dec_sdssm,Vpec_bc_sdssm,eVpec_bc_sdssm,nv_sdssm,delt_sdssm,z_sdssm,Hub,sigma_star_adjustor,CFtype,QzzType)        
            B_MC[:,i] ,Sv_MC[i], Um_Cov,SX,SY,SZ,SS = MC_qMLE( ra_sdssm,dec_sdssm,Vpec_bc_sdssm,eVpec_bc_sdssm,nv_sdssm,delt_sdssm,z_sdssm,Hub,NmcSamp,sigma_star_adjustor,CFtype,QzzType)
        if(CFtype=='BQ'):
            sigma_star_adjustor=100000.0
            B_opt[:,i],Q_opt[:,i],Sv_opt[i]         = Opt_qMLE(ra_sdssm,dec_sdssm,Vpec_bc_sdssm,eVpec_bc_sdssm,nv_sdssm,delt_sdssm,z_sdssm,Hub,sigma_star_adjustor,CFtype,QzzType) 
            if(QzzType== 'fix'):    
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,eQzz[i],SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,SS = MC_qMLE( ra_sdssm,dec_sdssm,Vpec_bc_sdssm,eVpec_bc_sdssm,nv_sdssm,delt_sdssm,z_sdssm,Hub,NmcSamp,sigma_star_adjustor,CFtype,QzzType) 
            if(QzzType== 'free'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,Szz,SS     = MC_qMLE( ra_sdssm,dec_sdssm,Vpec_bc_sdssm,eVpec_bc_sdssm,nv_sdssm,delt_sdssm,z_sdssm,Hub,NmcSamp,sigma_star_adjustor,CFtype,QzzType)         
    if(Fit_tech=='etaMLE' ):  
        if(CFtype=='BF'):
            B_opt[:,i],Sv_opt[i]                    = Opt_etaMLE(ra_sdssm,dec_sdssm,z_sdssm,logd_sdssm,elogd_sdssm,False,OmegaM,OmegaA,Hub,CFtype,QzzType,False) 
            B_MC[:,i] ,Sv_MC[i], Um_Cov,SX,SY,SZ,SS = MC_etaMLE( ra_sdssm,dec_sdssm,z_sdssm,logd_sdssm,elogd_sdssm,False,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,False) 
        if(CFtype=='BQ'):
            B_opt[:,i],Q_opt[:,i],Sv_opt[i] = Opt_etaMLE(ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,False,OmegaM,OmegaA,Hub,CFtype,QzzType,False) 
            if(QzzType== 'fix'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,eQzz[i],SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,SS = MC_etaMLE( ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,False,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,False)    
            if(QzzType== 'free'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,Szz,SS = MC_etaMLE( ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,False,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,False)    
    if(Fit_tech=='sketaMLE' ):  
        if(CFtype=='BF'):
            B_opt[:,i],Sv_opt[i]                    = Opt_etaMLE(ra_sdssm,dec_sdssm,z_sdssm,logd_sdssm,elogd_sdssm,alpha_sdssm,OmegaM,OmegaA,Hub,CFtype,QzzType,True) 
            B_MC[:,i] ,Sv_MC[i], Um_Cov,SX,SY,SZ,SS = MC_etaMLE( ra_sdssm,dec_sdssm,z_sdssm,logd_sdssm,elogd_sdssm,alpha_sdssm,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,True) 
        if(CFtype=='BQ'):
            B_opt[:,i],Q_opt[:,i],Sv_opt[i] = Opt_etaMLE(ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,alpha_sdssm,OmegaM,OmegaA,Hub,CFtype,QzzType,True) 
            if(QzzType== 'fix'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,eQzz[i],SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,SS = MC_etaMLE( ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,alpha_sdssm,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,True)    
            if(QzzType== 'free'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,Szz,SS = MC_etaMLE( ra_sdssm ,dec_sdssm ,z_sdssm ,logd_sdssm ,elogd_sdssm ,alpha_sdssm,OmegaM,OmegaA,Hub,NmcSamp,CFtype,QzzType,True)    
    if(Fit_tech=='wMLE' ): 
        Vpec_sdssm = Vpec_Fun_wat(z_sdssm,logd_sdssm,OmegaM,OmegaA, Hub)
        eVpec_sdssm= Vpec_Fun_wat(z_sdssm,elogd_sdssm,OmegaM,OmegaA, Hub)
        if(CFtype=='BF'):
            B_opt[:,i],Sv_opt[i]                    = Opt_wMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm,z_sdssm,Hub,CFtype,QzzType)        
            B_MC[:,i] ,Sv_MC[i], Um_Cov,SX,SY,SZ,SS = MC_wMLE( ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm,z_sdssm,Hub,NmcSamp,CFtype,QzzType)
        if(CFtype=='BQ'):
            B_opt[:,i],Q_opt[:,i],Sv_opt[i] = Opt_wMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm,z_sdssm,Hub,CFtype,QzzType) 
            if(QzzType== 'fix'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,eQzz[i],SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,SS = MC_wMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm,z_sdssm,Hub,NmcSamp,CFtype,QzzType) 
            if(QzzType== 'free'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,Szz,SS = MC_wMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm,z_sdssm,Hub,NmcSamp,CFtype,QzzType) 
    if(Fit_tech=='tMLE' ): 
        Vpec_sdssm = Vpec_Fun_tra(z_sdssm,logd_sdssm,OmegaM,OmegaA, Hub) 
        eVpec_sdssm= Vpec_Fun_wat(z_sdssm,elogd_sdssm,OmegaM,OmegaA, Hub)#0.177*z_sdssm*LightSpeed
        if(CFtype=='BF'):
            B_opt[:,i],Sv_opt[i]                    = Opt_tMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm ,z_sdssm,Hub,CFtype,QzzType)        
            B_MC[:,i] ,Sv_MC[i], Um_Cov,SX,SY,SZ,SS = MC_tMLE( ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm ,z_sdssm,Hub,NmcSamp,CFtype,QzzType)
        if(CFtype=='BQ'):
            B_opt[:,i],Q_opt[:,i],Sv_opt[i] = Opt_tMLE(ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm ,z_sdssm,Hub,CFtype,QzzType) 
            if(QzzType== 'fix'):    
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,eQzz[i],SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,SS = MC_tMLE( ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm ,z_sdssm,Hub,NmcSamp,CFtype,QzzType)
            if(QzzType== 'free'):
                B_MC[:,i] ,Q_MC[:,i] ,Sv_MC[i], Um_Cov,Qm_cov,Totm_cov,SX,SY,SZ,Sxx,Sxy,Sxz,Syy,Syz,Szz,SS = MC_tMLE( ra_sdssm,dec_sdssm,Vpec_sdssm,eVpec_sdssm ,z_sdssm,Hub,NmcSamp,CFtype,QzzType)
# 3.5 stor the output into arrays
    if(CFtype=='BF')or((CFtype=='BQ')):    
        U_cov[i]       = Um_Cov
        U_true[3*i]    = Ux_true[i]  ; U_true[3*i+1]   = Uy_true[i]  ; U_true[3*i+2]   = Uz_true[i]   
        U_data[3*i]    = B_opt[0,i]  ; U_data[3*i+1]   = B_opt[1,i]  ; U_data[3*i+2]   = B_opt[2,i]
    if(CFtype=='BQ'):
      if(QzzType== 'fix'):                
        Q_cov[i]       = Qm_cov
        Q_true[5*i]    = Qxx_true[i] ; Q_true[5*i+1]   = Qxy_true[i] ; Q_true[5*i+2]   = Qxz_true[i]
        Q_true[5*i+3]  = Qyy_true[i] ; Q_true[5*i+4]   = Qyz_true[i]   
        Q_data[5*i]    = Q_opt[0,i]  ; Q_data[5*i+1]   = Q_opt[1,i]  ; Q_data[5*i+2]   = Q_opt[2,i]
        Q_data[5*i+3]  = Q_opt[3,i]  ; Q_data[5*i+4]   = Q_opt[4,i] 
        Tot_cov[i]     = Totm_cov
        Tot_true[8*i]  = Ux_true[i]  ; Tot_true[8*i+1] = Uy_true[i]  ; Tot_true[8*i+2] = Uz_true[i]  
        Tot_true[8*i+3]= Qxx_true[i] ; Tot_true[8*i+4] = Qxy_true[i] ; Tot_true[8*i+5] = Qxz_true[i]
        Tot_true[8*i+6]= Qyy_true[i] ; Tot_true[8*i+7] = Qyz_true[i] 
        Tot_data[8*i]  = B_opt[0,i]  ; Tot_data[8*i+1] = B_opt[1,i]  ; Tot_data[8*i+2] = B_opt[2,i]  
        Tot_data[8*i+3]= Q_opt[0,i]  ; Tot_data[8*i+4] = Q_opt[1,i]  ; Tot_data[8*i+5] = Q_opt[2,i]
        Tot_data[8*i+6]= Q_opt[3,i]  ; Tot_data[8*i+7] = Q_opt[4,i]
      if(QzzType== 'free'):
        Q_cov[i]       = Qm_cov
        Q_true[6*i]    = Qxx_true[i] ; Q_true[6*i+1]   = Qxy_true[i] ; Q_true[6*i+2]   = Qxz_true[i]
        Q_true[6*i+3]  = Qyy_true[i] ; Q_true[6*i+4]   = Qyz_true[i] ; Q_true[6*i+5]   = Qzz_true[i]   
        Q_data[6*i]    = Q_opt[0,i]  ; Q_data[6*i+1]   = Q_opt[1,i]  ; Q_data[6*i+2]   = Q_opt[2,i]
        Q_data[6*i+3]  = Q_opt[3,i]  ; Q_data[6*i+4]   = Q_opt[4,i]  ; Q_data[6*i+5]   = Q_opt[5,i]
        Tot_cov[i]     = Totm_cov
        Tot_true[9*i]  = Ux_true[i]  ; Tot_true[9*i+1] = Uy_true[i]  ; Tot_true[9*i+2] = Uz_true[i]  
        Tot_true[9*i+3]= Qxx_true[i] ; Tot_true[9*i+4] = Qxy_true[i] ; Tot_true[9*i+5] = Qxz_true[i]
        Tot_true[9*i+6]= Qyy_true[i] ; Tot_true[9*i+7] = Qyz_true[i] ; Tot_true[9*i+8] = Qzz_true[i]
        Tot_data[9*i]  = B_opt[0,i]  ; Tot_data[9*i+1] = B_opt[1,i]  ; Tot_data[9*i+2] = B_opt[2,i]  
        Tot_data[9*i+3]= Q_opt[0,i]  ; Tot_data[9*i+4] = Q_opt[1,i]  ; Tot_data[9*i+5] = Q_opt[2,i]
        Tot_data[9*i+6]= Q_opt[3,i]  ; Tot_data[9*i+7] = Q_opt[4,i]  ; Tot_data[9*i+8] = Q_opt[5,i]                
    i = i + 1
# 3.6 Varify Optmization = MCMC results
plt.figure(1,figsize=(9,6))
plt.scatter( B_opt[0,:],B_MC[0,:],s=50,marker="o",c='royalblue')
plt.scatter( B_opt[1,:],B_MC[1,:],s=50,marker="D",c='g')
plt.scatter( B_opt[2,:],B_MC[2,:],s=50,marker="*",c='crimson')
plt.plot([-900, 900], [-900, 900], ls="--", c=".3");plt.xlim(-900,900) ; plt.ylim(-900,900)
plt.xlabel('$B_{opt}~[ km~s^{-1}]$',fontsize=22);plt.ylabel('$B_{MCMC}~[ km~s^{-1}]$',fontsize=22);  
plt.xticks(fontsize=16);  plt.yticks(fontsize=16); plt.grid(True) ; plt.title('Varify Optmization=MCMC-----%s'% Fit_tech,fontsize=22)
if(CFtype=='BQ'):
  plt.figure(2,figsize=(9,6))
  plt.scatter( Q_opt[0,:],Q_MC[0,:],s=50,marker="o",c='royalblue')
  plt.scatter( Q_opt[1,:],Q_MC[1,:],s=50,marker="o",c='y')
  plt.scatter( Q_opt[2,:],Q_MC[2,:],s=50,marker="D",c='g')
  plt.scatter( Q_opt[3,:],Q_MC[3,:],s=50,marker="o",c='k')
  plt.scatter( Q_opt[4,:],Q_MC[4,:],s=50,marker="*",c='crimson')
  if(QzzType== 'free'):plt.scatter( Q_opt[5,:],Q_MC[5,:],s=50,marker="*",c='crimson')
  plt.plot([-10, 10], [-10, 10], ls="--", c=".3");plt.xlim(-10,10) ; plt.ylim(-10,10)
  plt.xlabel('$Q_{opt}$',fontsize=22);plt.ylabel('$Q_{MCMC}$',fontsize=22);  
  plt.xticks(fontsize=16);plt.yticks(fontsize=16); plt.grid(True) ; plt.title('Varify Optmization=MCMC-----%s'% Fit_tech,fontsize=22)
plt.figure(3,figsize=(9,6))
plt.scatter( np.abs(Sv_opt),Sv_MC,s=50,marker="o",c='royalblue')
plt.plot([min([min(np.abs(Sv_opt)),min(Sv_MC)]), max([max(np.abs(Sv_opt)),max(Sv_MC)]) ], [min([min(np.abs(Sv_opt)),min(Sv_MC)]), max([max(np.abs(Sv_opt)),max(Sv_MC)])], ls="--", c=".3");plt.xlim(min([min(np.abs(Sv_opt)),min(Sv_MC)]),max([max(np.abs(Sv_opt)),max(Sv_MC)])) ; plt.ylim(min([min(np.abs(Sv_opt)),min(Sv_MC)]),max([max(np.abs(Sv_opt)),max(Sv_MC)]))
plt.xlabel('$\sigma_{opt}~[ km~s^{-1}]$',fontsize=22);plt.ylabel('$\sigma_{MCMC}~[ km~s^{-1}]$',fontsize=22);  
plt.xticks(fontsize=16);  plt.yticks(fontsize=16); plt.grid(True) ; plt.title('Varify Optmization=MCMC-----%s'% Fit_tech,fontsize=22)
plt.show()    
#============================   THE END OF  etaMLE   ==========================




















# 4. Calculate the chi-squared-------------------------------------------------
# 4.1 chi-squared of B:
Cov_all = np.zeros((3*Nmock,3*Nmock))
for i in range(Nmock):
    Cov_all[3*i:3*(i+1),3*i:3*(i+1)] = U_cov[i]
pivots   = np.zeros(3*Nmock, np.intc)
identity = np.eye(3*Nmock)
U_cov_lu, pivots, U_cov_inv, info = lapack.dgesv(Cov_all, identity)
chi_squared = 0.0
for i in range(3*Nmock):
    chi_squared += (U_data[i]-U_true[i])*np.sum(U_cov_inv[i,0:]*(U_data-U_true))
print( ' ')
print( 'Python emcee BKF Chi^2=', chi_squared/(3*Nmock-1.0))
# erros;
PyerrB=np.zeros((3,Nmock))
for i in range((Nmock)):
    PyerrB[0,i]=np.sqrt(Cov_all[i*3,i*3])
    PyerrB[1,i]=np.sqrt(Cov_all[i*3+1,i*3+1])
    PyerrB[2,i]=np.sqrt(Cov_all[i*3+2,i*3+2])
# 4.2 chi-squared of shear moments:
if(CFtype=='BQ'):
  if(QzzType== 'fix'):
    Cov_all  = np.zeros((5*Nmock,5*Nmock))
    for i in range(Nmock):
        Cov_all[5*i:5*(i+1),5*i:5*(i+1)] = Q_cov[i]
    pivots = np.zeros(5*Nmock, np.intc)
    identity = np.eye(5*Nmock)
    Q_cov_lu, pivots, Q_cov_inv, info = lapack.dgesv(Cov_all, identity)
    chi_squared = 0.0
    for i in range(5*Nmock):
        chi_squared += (Q_data[i]-Q_true[i])*np.sum(Q_cov_inv[i,0:]*(Q_data-Q_true))
    print( ' ')
    print( 'Python emcee QUA Chi^2=', chi_squared/(5*Nmock-1.0))
    # erros;
    PyerrQ=np.zeros( (5,Nmock))
    for i in range(Nmock):
        PyerrQ[0,i]=np.sqrt(Cov_all[i*5,i*5])
        PyerrQ[1,i]=np.sqrt(Cov_all[i*5+1,i*5+1])
        PyerrQ[2,i]=np.sqrt(Cov_all[i*5+2,i*5+2])
        PyerrQ[3,i]=np.sqrt(Cov_all[i*5+3,i*5+3])
        PyerrQ[4,i]=np.sqrt(Cov_all[i*5+4,i*5+4])
# 4.3 chi-squared of total BQ:
    Cov_all  = np.zeros((8*Nmock,8*Nmock))
    for i in range(Nmock):
        Cov_all[8*i:8*(i+1),8*i:8*(i+1)] = Tot_cov[i]
    pivots = np.zeros(8*Nmock, np.intc)
    identity = np.eye(8*Nmock)
    Tot_cov_lu, pivots, Tot_cov_inv, info = lapack.dgesv(Cov_all, identity)
    chi_squared = 0.0
    for i in range(8*Nmock):
        chi_squared += (Tot_data[i]-Tot_true[i])*np.sum(Tot_cov_inv[i,0:]*(Tot_data-Tot_true))
    print(' ')
    print('Python emcee Tot=[B,Q] Chi^2=', chi_squared/(8*Nmock-1.0))
  if(QzzType== 'free'):
    Cov_all  = np.zeros((6*Nmock,6*Nmock))
    for i in range(Nmock):
        Cov_all[6*i:6*(i+1),6*i:6*(i+1)] = Q_cov[i]
    pivots = np.zeros(6*Nmock, np.intc)
    identity = np.eye(6*Nmock)
    Q_cov_lu, pivots, Q_cov_inv, info = lapack.dgesv(Cov_all, identity)
    chi_squared = 0.0
    for i in range(6*Nmock):
        chi_squared += (Q_data[i]-Q_true[i])*np.sum(Q_cov_inv[i,0:]*(Q_data-Q_true))
    print( ' ')
    print( 'Python emcee QUA Chi^2=', chi_squared/(6*Nmock-1.0))
    # erros;
    PyerrQ=np.zeros( (6,Nmock))
    for i in range(Nmock):
        PyerrQ[0,i]=np.sqrt(Cov_all[i*6,i*6])
        PyerrQ[1,i]=np.sqrt(Cov_all[i*6+1,i*6+1])
        PyerrQ[2,i]=np.sqrt(Cov_all[i*6+2,i*6+2])
        PyerrQ[3,i]=np.sqrt(Cov_all[i*6+3,i*6+3])
        PyerrQ[4,i]=np.sqrt(Cov_all[i*6+4,i*6+4])
        PyerrQ[5,i]=np.sqrt(Cov_all[i*6+5,i*6+5])
    #chi-squared of total BQ:
    Cov_all  = np.zeros((9*Nmock,9*Nmock))
    for i in range(Nmock):
        Cov_all[9*i:9*(i+1),9*i:9*(i+1)] = Tot_cov[i]
    pivots = np.zeros(9*Nmock, np.intc)
    identity = np.eye(9*Nmock)
    Tot_cov_lu, pivots, Tot_cov_inv, info = lapack.dgesv(Cov_all, identity)
    chi_squared = 0.0
    for i in range(9*Nmock):
        chi_squared += (Tot_data[i]-Tot_true[i])*np.sum(Tot_cov_inv[i,0:]*(Tot_data-Tot_true))
    print(' ')
    print('Python emcee Tot=[B,Q] Chi^2=', chi_squared/(8*Nmock-1.0))
















# 5. plots:--------------------------------------------------------------------
plt.figure(4,figsize=(9,6))
plt.scatter( Ux_true,B_opt[0,:],s=20,marker="o",c='royalblue')
plt.errorbar(Ux_true,B_opt[0,:],PyerrB[0],linestyle='',color='royalblue',marker="o",label='$B_x$')
plt.scatter( Uy_true,B_opt[1,:],s=20,marker="D",c='g')
plt.errorbar(Uy_true,B_opt[1,:],PyerrB[1],linestyle='',color='g',marker="D",label='$B_y$')
plt.scatter( Uz_true,B_opt[2,:],s=80,marker="*",c='crimson')
plt.errorbar(Uz_true,B_opt[2,:],PyerrB[2],linestyle='',color='crimson',marker="*",label='$B_z$')
plt.plot([-1000, 1000], [-1000, 1000], ls="--", c=".3");plt.xlim(-900,900);plt.ylim(-900,900);plt.legend(fontsize=20)
plt.ylabel('$B( '+Fit_tech+')~[ km~s^{-1}]$',fontsize=22);
if(Fit_tech=='etaMLE'):plt.ylabel('$B(\eta MLE)~[ km~s^{-1}]$',fontsize=22);
if(Fit_tech=='sketaMLE'):plt.ylabel('$Q(skew\eta MLE)~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);
plt.xlabel('$B_{true}~[ km~s^{-1}]$',fontsize=22);plt.xticks(fontsize=16);plt.yticks(fontsize=16);plt.grid(True)
plt.savefig('/Users/fei/WSP/Scie/Proj7/Result/SDSS/SDSS_B_'+Fit_tech+CFtype+QzzType+mkSET+'.pdf', dpi=150, bbox_inches='tight')

if(CFtype=='BQ'):     
    plt.figure(6,figsize=(9,10))
    ax=plt.subplot(211)
    plt.scatter( Qxx_true,Q_opt[0,:],s=20,marker="o",c='royalblue',label='$Q_{xx}$')
    plt.errorbar(Qxx_true,Q_opt[0,:],PyerrQ[0],linestyle='',color='royalblue',marker="o")
    plt.scatter( Qyy_true,Q_opt[3,:],s=20,marker="D",c='g',label='$Q_{yy}$')
    plt.errorbar(Qyy_true,Q_opt[3,:],PyerrQ[3],linestyle='',color='g',marker="D")
    if(QzzType== 'fix'):
        plt.scatter( Qzz_true,-Q_opt[0,:]-Q_opt[3,:],s=80,marker="*",c='crimson',label='$Q_{zz}$')
        plt.errorbar(Qzz_true,-Q_opt[0,:]-Q_opt[3,:],eQzz,linestyle='',color='crimson',marker="*")
    if(QzzType== 'free'): 
        plt.scatter( Qzz_true,Q_opt[5,:],s=80,marker="*",c='crimson',label='$Q_{zz}$')
        plt.errorbar(Qzz_true,Q_opt[5,:],PyerrQ[5],linestyle='',color='crimson',marker="*")
    plt.plot([-7, 7], [-7, 7], ls="--", c=".3") ; plt.xlim(-7, 7);plt.ylim(-7, 7);plt.legend(fontsize=20)
    plt.ylabel('$Q( '+Fit_tech+')~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);
    if(Fit_tech=='etaMLE'):plt.ylabel('$Q(\eta MLE)~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);
    if(Fit_tech=='sketaMLE'):plt.ylabel('$Q(skew\eta MLE)~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);
    plt.setp(ax.get_xticklabels(), visible=False)  ; plt.yticks(fontsize=16); plt.grid(True)
    ax=plt.subplot(212)#--------- 
    plt.scatter( Qxy_true,Q_opt[1,:],s=20,marker="o",c='indigo',label='$Q_{xy}$')
    plt.errorbar(Qxy_true,Q_opt[1,:],PyerrQ[1],linestyle='',color='indigo',marker="o")
    plt.scatter( Qxz_true,Q_opt[2,:],s=20,marker="D",c='purple',label='$Q_{xz}$')
    plt.errorbar(Qxz_true,Q_opt[2,:],PyerrQ[2],linestyle='',color='purple',marker="D")
    plt.scatter( Qyz_true,Q_opt[4,:],s=80,marker="*",c='magenta',label='$Q_{yz}$')
    plt.errorbar(Qyz_true,Q_opt[4,:],PyerrQ[4],linestyle='',color='magenta',marker="*")
    plt.plot([-7, 7], [-7, 7], ls="--", c=".3") ; plt.xlim(-7, 7) ; plt.ylim(-7, 7) ; plt.legend(fontsize=20)
    plt.ylabel('$Q( '+Fit_tech+')~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);plt.xlabel('$Q_{true}~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=22);
    if(Fit_tech=='etaMLE'):plt.ylabel('$Q(\eta MLE)~~[ h~km~s^{-1}~Mpc^{-1} ]$',fontsize=19);
    plt.xticks(fontsize=16);  plt.yticks(fontsize=16); plt.grid(True)
    plt.subplots_adjust(left=0.125,bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.02)
    plt.savefig('/Users/fei/WSP/Scie/Proj7/Result/SDSS/SDSS_Q_'+Fit_tech+CFtype+QzzType+mkSET+'.pdf', dpi=150, bbox_inches='tight')







 
 
 
 
 
 
 
 
 
 
# 6. out put:------------------------------------------------------------------
if(CFtype=='BF'):
  outfile = open(output_dir+Fit_tech+CFtype+mkSET+'.txt', 'w')
  outfile.write("#  mockID    Bx      By     Bz     errBx     errBy     errBz     Bx_true      By_true     Bz_true \n")
  for i in range(Nmock):
    outfile.write("%9.1lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  \n"
                  %(mockID[i],B_opt[0,i],  B_opt[1,i],   B_opt[2,i],   PyerrB[0,i], PyerrB[1,i],  PyerrB[2,i],
                    Ux_true[i],  Uy_true[i],   Uz_true[i]))
if(CFtype=='BQ'):
    
  outfile = open(output_dir+Fit_tech+CFtype+QzzType+mkSET+'.txt', 'w')
  outfile.write("#  mockID    Bx      By     Bz     errBx     errBy     errBz     Bx_true      By_true     Bz_true \n")
  for i in range(Nmock):
    outfile.write("%9.1lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  \n"
                  %(mockID[i],B_opt[0,i],  B_opt[1,i],   B_opt[2,i],   PyerrB[0,i], PyerrB[1,i],  PyerrB[2,i],
                    Ux_true[i],  Uy_true[i],   Uz_true[i]))    
    
  outfile = open(output_dir+Fit_tech+CFtype+QzzType+mkSET+'.txt', 'w')
  outfile.write("#  mockID    Bx      By     Bz     errBx     errBy     errBz    Qxx      Qxy    Qxz     Qyy     Qyz    Qzz    errQxx      errQxy    errQxz     errQyy     errQyz    errQzz     Bx_true      By_true     Bz_true    Qxx_true      Qxy_true    Qxz_true     Qyy_true     Qyz_true    Qzz_true \n")  
  for i in range(Nmock):
    if(QzzType== 'fix'):  
      outfile.write("%9.1lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n"
                  %(mockID[i],B_opt[0,i],  B_opt[1,i],   B_opt[2,i],   PyerrB[0,i], PyerrB[1,i],  PyerrB[2,i],                 
                    Q_opt[0,i],  Q_opt[1,i],   Q_opt[2,i],   Q_opt[3,i],  Q_opt[4,i],   -Q_opt[0,i]-Q_opt[3,i],PyerrQ[0,i],PyerrQ[1,i],PyerrQ[2,i],PyerrQ[3,i],PyerrQ[4,i],eQzz[i],
                    Ux_true[i],  Uy_true[i],   Uz_true[i],   Qxx_true[i], Qxy_true[i],  Qxz_true[i], Qyy_true[i],Qyz_true[i],Qzz_true[i]))
    if(QzzType== 'free'):  
      outfile.write("%9.1lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n"
                  %(mockID[i],B_opt[0,i],  B_opt[1,i],   B_opt[2,i],   PyerrB[0,i], PyerrB[1,i],  PyerrB[2,i],                 
                    Q_opt[0,i],  Q_opt[1,i],   Q_opt[2,i],   Q_opt[3,i],  Q_opt[4,i],   Q_opt[5,i], PyerrQ[0,i],PyerrQ[1,i],PyerrQ[2,i],PyerrQ[3,i],PyerrQ[4,i],PyerrQ[5,i],
                    Ux_true[i],  Uy_true[i],   Uz_true[i],   Qxx_true[i], Qxy_true[i],  Qxz_true[i], Qyy_true[i],Qyz_true[i],Qzz_true[i]))

outfile.close()
 


plt.show()
