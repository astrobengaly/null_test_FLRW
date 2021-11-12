# null_test_FLRW
Scripts and cosmological data for performing a null test of the FLRW assumption - that is, the Universe is statistically isotropic and homogeneous at large scales - based on Maartens 2011 (arXiv:1104.1300) and Arjona & Nesseris 2021 (arXiv:2103.06789). This test is hereafter referred as zeta(z). 

We use observational measurements of radial [H(z)] and transverse [DA(z)] BAO modes in order to test their mutual consistency along the redshift. Any significant inconsistency between them would lead to a departure of the FLRW assumption, thus a breakdown one of the foundations of the standard cosmological paradigm - the LCDM model. This analysis is carried out using a non-parametric reconstruction method based on Gaussian Processes, as in the GaPP package (see GaPP repository and references therein for more info). Therefore, minimal assumptions on the material content of the Universe are made. 

The code runs as a script such as 
python null_test_FLRW.py hz_cc 0 18 daz_transv_bao 
where hz_cc represents the radial BAO data points, 0 and 18 correspond to the number of galaxy age and radial bao measurements in the sample (so 0 means no galaxy age data are included), and daz_transv_bao contains the transverse BAO measurements.  

The outputs consist on three plots, as follow: 
- The reconstructed comoving distance curve DC(z) obtained from integrating the reconstructed H(z) curve from GaPP using a trapezoid sum rule
- The reconstructed angular diameter distance DA(z) from the transverse BAO mode measurements
- The null FLRW test zeta(z) result. 
