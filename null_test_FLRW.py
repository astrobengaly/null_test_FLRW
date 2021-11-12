def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys

from gapp import gp, dgp, covariance
import pickle
from numpy import array,concatenate,loadtxt,savetxt,zeros
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.integrate import quad

data1=sys.argv[1]
npts1=int(sys.argv[2])
npts2=int(sys.argv[3])
data2=sys.argv[4]
#prior=sys.argv[5]

# comoving radial distance integrand for a CPL-like model
def hz_model(z,omegaM,omegaX,h,w0,w1):
    return h*np.sqrt(omegaM*(1.+z)**3. + omegaX*(1.+z)**(3.*(1.+w0+w1))*np.exp(-3.*w1*z/(1.+z))) 

# comoving radial distance integrand for a CPL-like model
def integrand(z,omegaM,omegaX,h,w0,w1):
    return 1./np.sqrt(omegaM*(1.+z)**3. + omegaX*(1.+z)**(3.*(1.+w0+w1))*np.exp(-3.*w1*z/(1.+z))) 

# comoving radial distance assuming the above integrand    
def dz_model(z,omegaM,omegaX,h,w0,w1):
    aux=quad(integrand, 0., z, args=(omegaM,omegaX,h,w0,w1))
    aux=aux[0]
    DH = (2998./h)
    if (1.-omegaM+omegaX == 0.):
        return DH*aux
    if (1.-omegaM+omegaX > 0.):
        return DH*np.sinh(aux*np.sqrt(1.-omegaM+omegaX))/(np.sqrt(1.-omegaM+omegaX))
    if (1.-omegaM+omegaX < 0.):
        return DH*np.sin(aux*np.sqrt(abs(1.-omegaM+omegaX)))/(np.sqrt(abs(1.-omegaM+omegaX)))

if __name__=="__main__":

        # ========= LOADING AND PRE-PROCESSING COSMOLOGICAL DATA
        (z1,hz,errhz,hid) = loadtxt(data1+'_'+str(npts1)+'+'+str(npts2)+'pts.dat',unpack='True')
        (z2,dz,errdz) = loadtxt(data2+'.dat',unpack='True')
        
        # defining the speed of light
        c = 2.998e5
                
        # ===================== GAUSSIAN PROCESS RECONSTRUCTION
                
        # defining the redshift range and the number of bins of the GP reconstruction
        zmin = 0.01
        zmax = 2.50
        nbins = 1000
      
        # initial values of the hyperparameters - optional
        #initheta = [0., 0.]

        # GP reconstruction - Cosmic chronometer measurements
        g = gp.GaussianProcess(z1,hz,errhz,covfunction=covariance.SquaredExponential,cXstar=(zmin,zmax,nbins))
        dg = dgp.DGaussianProcess(z1,hz,errhz,covfunction=covariance.SquaredExponential,cXstar=(zmin,zmax,nbins))
        (rec,theta) = g.gp(thetatrain='False')
        (drec,dtheta) = dg.dgp(thetatrain='False')
        (d2rec,d2theta) = dg.d2gp(thetatrain='False')
        
        # calculate covariances between h, h' and h'' at points Zstar.
        fcov = dg.f_covariances(fclist=[0,1])
        
        # getting the reconstructed quantities and associating them to new variables
        n_start = 0
        z_rec = rec[n_start:,0]
        ez_rec = rec[n_start:,1]
        errez_rec = rec[n_start:,2]    
        dez_rec = drec[n_start:,1]
        errdez_rec = drec[n_start:,2]
        errezdez_rec = fcov[n_start:,:,]
        
        #--------------------------------------------------------------------------------------
        
        # GP reconstruction - cosmological distance measurements 
        
        g = gp.GaussianProcess(z2,dz,errdz,covfunction=covariance.SquaredExponential,cXstar=(zmin,zmax,nbins))
        dg = dgp.DGaussianProcess(z2,dz,errdz,covfunction=covariance.SquaredExponential,cXstar=(zmin,zmax,nbins))
        (rec1,theta1) = g.gp(thetatrain='False')
        (drec1,dtheta1) = dg.dgp(thetatrain='False')
        (d2rec1,d2theta1) = dg.d2gp(thetatrain='False')
        
        # calculate covariances between d, d' and d'' at points Zstar.
        fcov1 = dg.f_covariances(fclist=[0,1])
        
        # getting the reconstructed quantities and associating them to new variables
        n_start = 0
        daz_rec = rec1[n_start:,1]
        errdaz_rec = rec1[n_start:,2]     
        ddaz_rec = drec1[n_start:,1]
        errddaz_rec = drec1[n_start:,2]
        d2daz_rec = d2rec1[n_start:,1]
        errd2daz_rec = d2rec1[n_start:,2]
        errdazddaz_rec = fcov1[n_start:,:,]
            
        # ----------- null test for FLRW deviations 
        # ----------- following Maartens 2011 (M11) and Arjona & Nesseris (AN21)
        
        # using the trapezoide sum rule to get D(z) from the H(z) data points
        sumdz = 0.
        sumerrdz = 0.
        dz_arr = []
        npts = len(z1)
        for i in range(npts-1):
            sumdz += (z1[i+1]-z1[i])*( (1./hz[i+1]) + (1./hz[i]) )
            sumerrdz += (z1[i+1]-z1[i])*np.sqrt( (errhz[i+1]/(hz[i+1]**4.)) + (errhz[i]/(hz[i]**4.)) )
            dz_arr.append([sumdz, sumerrdz])
            
        # defining dz and errdz from the dz array
        dz_arr = (c/2.)*np.array(dz_arr)
        dz = dz_arr[:,0]
        errdz = dz_arr[:,1]
        
        # using the trapezoide sum rule to get D(z) from the reconstructed H(z) 
        sumdz_rec = 0.
        sumerrdz_rec = 0.
        dz_rec_arr = []
        for i in range(nbins-1):
            sumdz_rec += (z_rec[i+1]-z_rec[i])*( (1./ez_rec[i+1]) + (1./ez_rec[i]) )
            sumerrdz_rec += (z_rec[i+1]-z_rec[i])*np.sqrt( (errez_rec[i+1]/(ez_rec[i+1]**4.)) + (errez_rec[i]/(ez_rec[i]**4.)) )
            dz_rec_arr.append([sumdz_rec, sumerrdz_rec])
        
        # defining dz_rec and errdz_rec from the dz_rec array
        dz_rec_arr = (c/2.)*np.array(dz_rec_arr)
        dz_rec = dz_rec_arr[:,0]
        errdz_rec = dz_rec_arr[:,1]
            
        # zetaz test following AN21:
        zetaz_arr = []
        for i in range(nbins-1):
            zetaz = 1. - (((1.+z_rec[i])*daz_rec[i])/(dz_rec[i]))
            zerretaz = np.sqrt( ( ((1.+z_rec[i])/(dz_rec[i]))*errdaz_rec[i] )**2. + ( (((1.+z_rec[i])*daz_rec[i])/(dz_rec[i]**2.))*errdz_rec[i] )**2. )
            zetaz_arr.append([zetaz,zerretaz])
        
        # same as above, but for zetaz
        zetaz_arr = np.array(zetaz_arr)
        zetaz = zetaz_arr[:,0]
        zerretaz = zetaz_arr[:,1]
        
        ## debug
        #for i in range(nbins-1):
            #print z_rec[i], daz_rec[i], errdaz_rec[i], dz_rec[i], errdz_rec[i], zetaz[i], zerretaz[i]
        
        # saving results
        filename_out = 'rec_zetaz_hz_daz_bao'
        savetxt(filename_out+'.dat',np.transpose([z_rec[0:nbins-1], daz_rec[0:nbins-1], errdaz_rec[0:nbins-1], dz_rec, errdz_rec, zetaz, zerretaz]))
        
        filename_out1 = 'dz_cc_'+str(npts1)+'+'+str(npts2)+'pts'
        savetxt(filename_out1+'.dat',np.transpose([z1[0:npts-1], dz, errdz]))
            
        ## ========================= PLOTTING RESULTS
                        
        # reloading results
        (z_rec1,daz_rec,errdaz_rec,dz_rec,errdz_rec,zetaz,errzetaz) = loadtxt(filename_out+'.dat',unpack='True')
        
        # latex rendering text fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  
        
        # --------------------------------------------------------------

        # plotting the DA(z) results
        fig, ax1 = plt.subplots(figsize = (15., 10.))
                
        ## Define axes
        ax1.set_xlabel(r"$z$", fontsize=22)
        ax1.set_ylabel(r"$D_\mathrm{A}(z)$ (Mpc)", fontsize=22)
        plt.xlim(zmin,zmax+0.01)
        #plt.ylim(-1.,1.)
        #ax.set_yscale('log')
        for t in ax1.get_xticklabels(): t.set_fontsize(22)
        for t in ax1.get_yticklabels(): t.set_fontsize(22)

        ax1.fill_between(z_rec1, daz_rec+1.*errdaz_rec, daz_rec-1.*errdaz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
        ax1.fill_between(z_rec1, daz_rec+2.*errdaz_rec, daz_rec-2.*errdaz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
        ax1.fill_between(z_rec1, daz_rec+3.*errdaz_rec, daz_rec-3.*errdaz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
        plt.legend((r"$1\sigma$", "$2\sigma$", "$3\sigma$"), fontsize='22', loc='lower right')
        plt.show()
            
        #saving the plot
        fig.savefig('rec_daz_transv_bao.png')

        # --------------------------------------------------------------
        
        # plotting the DC(z) results 
        fig, ax2 = plt.subplots(figsize = (15., 10.))
                
        ## Define axes
        ax2.set_xlabel(r"$z$", fontsize=22)
        ax2.set_ylabel(r"$D_\mathrm{C}(z)$ (Mpc)", fontsize=22)
        plt.xlim(zmin,zmax+0.01)
        #plt.ylim(-1.,1.)
        #ax.set_yscale('log')
        for t in ax2.get_xticklabels(): t.set_fontsize(22)
        for t in ax2.get_yticklabels(): t.set_fontsize(22)

        ax2.fill_between(z_rec1, dz_rec+1.*errdz_rec, dz_rec-1.*errdz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
        ax2.fill_between(z_rec1, dz_rec+2.*errdz_rec, dz_rec-2.*errdz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
        ax2.fill_between(z_rec1, dz_rec+3.*errdz_rec, dz_rec-3.*errdz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
        plt.legend((r"$1\sigma$", "$2\sigma$", "$3\sigma$"), fontsize='22', loc='lower right')
        plt.show()
            
        #saving the plot
        fig.savefig('rec_dz_radial_bao.png')
        
        # -------------------------------------------------------------
                
        # plotting the zetaz results
        fig, ax3 = plt.subplots(figsize = (15., 10.))
                
        ## Define axes
        ax3.set_xlabel(r"$z$", fontsize=22)
        ax3.set_ylabel(r"$\zeta(z)$", fontsize=22)
        plt.xlim(zmin,zmax+0.01)
        plt.ylim(-1.,1.)
        #ax.set_yscale('log')
        for t in ax3.get_xticklabels(): t.set_fontsize(22)
        for t in ax3.get_yticklabels(): t.set_fontsize(22)

        plt.axhline(0., color='black')
        ax3.fill_between(z_rec1, zetaz+1.*errzetaz, zetaz-1.*errzetaz, facecolor='#808080', alpha=0.80, interpolate=True)
        ax3.fill_between(z_rec1, zetaz+2.*errzetaz, zetaz-2.*errzetaz, facecolor='#808080', alpha=0.50, interpolate=True)
        ax3.fill_between(z_rec1, zetaz+3.*errzetaz, zetaz-3.*errzetaz, facecolor='#808080', alpha=0.30, interpolate=True)
        plt.legend((r"FLRW", "$1\sigma$", "$2\sigma$", "$3\sigma$"), fontsize='22', loc='lower right')
        #plt.show()
            
        #saving the plot
        fig.savefig(filename_out+'.png')