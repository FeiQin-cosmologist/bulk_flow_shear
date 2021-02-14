# Cosmological Equations
import math
from scipy import integrate
from scipy import special

# Speed of light in km/s
LightSpeed = 299792.458

# Calculates H(z)/H0
def Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    fz = ((1.0+redshift)**(3*(1.0+w0+wa*ap)))*math.exp(-3*wa*(redshift/(1.0+redshift)))
    omega_k = 1.0-omega_m-omega_lambda-omega_rad
    return math.sqrt(omega_rad*(1.0+redshift)**4+omega_m*(1.0+redshift)**3+omega_k*(1.0+redshift)**2+omega_lambda*fz)

# The Lookback Time Integrand
def LookbackTimeIntegrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return 1.0/((1.0+redshift)*Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap))

# Calculates the Lookback Time in s
def LookbackTime(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    kmtoMpc = 3.08567758149137E19
    return (kmtoMpc/Hubble_Constant)*integrate.quad(LookbackTimeIntegrand, 0.0, redshift, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]

# The Comoving Distance Integrand
def DistDcIntegrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return 1.0/Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)

# The Comoving Distance in Mpc
def DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (LightSpeed/Hubble_Constant)*integrate.quad(DistDcIntegrand, 0.0, redshift, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]

# The Transverse Comoving Distance in Mpc
def DistDm(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    omega_k = 1.0-omega_m-omega_lambda-omega_rad
    dc = DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap)
    if (omega_k > 0):
        prefac = (Hubble_Constant*math.sqrt(omega_k))/LightSpeed
        return math.sinh(dc*prefac)/prefac
    elif (omega_k < 0):
        prefac = (Hubble_Constant*math.sqrt(math.fabs(omega_k)))/LightSpeed
        return math.sin(dc*prefac)/prefac
    else: 
        return dc

# The Angular Diameter Distance in Mpc
def DistDa(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return DistDm(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap)/(1.0+redshift)

# The Luminosity Distance in Mpc
def DistDl(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (1.0+redshift)*DistDm(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap)

# The Linear Growth Factor Integrand assuming GR
def GrowthFactorGRIntegrand(scale_factor, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    redshift = (1.0/scale_factor)-1.0
    return 1.0/(scale_factor*Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap))**3

# The Linear Growth Factor assuming GR
def GrowthFactorGR(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    prefac = Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)
    scale_factor = 1.0/(1.0+redshift)
    return prefac*integrate.quad(GrowthFactorGRIntegrand, 0.0, scale_factor, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]

# The Linear Growth Factor Integrand for an arbitrary value of gamma
def GrowthFactorGammaIntegrand(scale_factor, gamma, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    redshift = (1.0/scale_factor)-1.0
    return (Omega_m_z(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)**gamma)/scale_factor

# The Linear Growth Factor for an arbitrary value of gamma
def GrowthFactorGamma(gamma, redshift_low, redshift_high, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    scale_factor_low = 1.0/(1.0+redshift_low)
    scale_factor_high = 1.0/(1.0+redshift_high)
    return math.exp(integrate.quad(GrowthFactorGammaIntegrand, scale_factor_low, scale_factor_high, args=(gamma, omega_m, omega_lambda, omega_rad, w0, wa, ap))[0])
    
# Calculate the BAO scaling parameter alpha
def Alpha(Da, HubbleParam, rdrag, Da_fid, HubbleParam_fid, rdrag_fid):
    return (Da/Da_fid)**(2.0/3.0)*(HubbleParam_fid/HubbleParam)**(1.0/3.0)*(rdrag_fid/rdrag)

# Calculate the BAO scaling parameter epsilon
def Epsilon(Da, HubbleParam, Da_fid, HubbleParam_fid):
    return ((Da_fid/Da)**(1.0/3.0)*(HubbleParam_fid/HubbleParam)**(1.0/3.0))-1.0

# Omega_M at a given redshift
def Omega_m_z(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return (omega_m*(1.0+redshift)**3)/(Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)**2)

# Calculates the Volume Averaged Angular Diameter Distance
def DistDv(redshift, Da, HubbleParam):
    return (((1.0+redshift)**2*LightSpeed*redshift*Da**2)/HubbleParam)**(1.0/3.0)

# Calculates the Alcock-Paczynski Parameter F
def AP_F(redshift, Da, HubbleParam):
    return ((1.0+redshift)*Da*HubbleParam)/LightSpeed

# Calculates the growth factor, EXACTLY as used in PICOLA, which is normalised in some weird way
def GrowthFactorPICOLA(redshift, omega_m):
    scale_factor = 1.0/(1.0+redshift)
    x = omega_m/(scale_factor**3*(1.0-omega_m))
    if (math.fabs(x-1.0) < 1.0E-3):
        x -= 1.0
        hyperP = 0.8595967680646080 - 0.10165999125204040*x + 0.025791094277821357*x**2 - 0.008194025861121475*x**3 + 0.0029076305993447644*x**4 - 0.00110254263871597610*x**5 + 0.00043707304964624546*x**6 - 0.000178888996468783100*x**7
        hyperM = 1.1765206505266006 + 0.15846194123099624*x - 0.014200487494738975*x**2 + 0.002801728034399257*x**3 - 0.0007268267888593511*x**4 + 0.00021801569226706922*x**5 - 0.00007163321597397065*x**6 + 0.000025063737576245116*x**7
    else:
        if (x < 1.0):
            hyperP = special.hyp2f1( 1.0/2.0, 2.0/3.0, 5.0/3.0, -x)
            hyperM = special.hyp2f1(-1.0/2.0, 2.0/3.0, 5.0/3.0, -x)
        else:
            x=1.0/x;
            if ((x < 1.0) and (x > 1.0/30.0)):
                hyperP  = 4.0*math.sqrt(x)*special.hyp2f1(-1.0/6.0, 1.0/2.0, 5.0/6.0, -x) - 3.4494794123063873799*x**(2.0/3.0)
                hyperM  = (4.0/(7.0*math.sqrt(x)))*special.hyp2f1(-7.0/6.0, -1.0/2.0, -1.0/6.0, -x) - 1.4783483195598803057*x**(2.0/3.0)
            elif (x <= 1.0/30.0):
                hyperP =                                3.9999999999999996*x**0.5 - 3.4494794123063865*x**0.66666666666666666 + 0.39999999999999990*x**1.5 - 0.136363636363636350*x**2.5 + 0.073529411764705870*x**3.5 - 0.047554347826086950*x**4.5 + 0.033943965517241374*x**5.5 - 0.0257812500000000000*x**6.5  + 0.0204363567073170720*x**7.5 - 0.0167132438497340400*x**8.5 + 0.0139977797022405640*x**9.5 - 0.0119455628475900410*x**10.5 + 0.0103500366210937500*x**11.5 - 0.00908057790406992600*x**12.5
                hyperM = 0.5714285714285715000/x**0.5 + 2.0000000000000010*x**0.5 - 1.4783483195598794*x**0.66666666666666666 + 0.10000000000000002*x**1.5 - 0.022727272727272735*x**2.5 + 0.009191176470588237*x**3.5 - 0.004755434782608697*x**4.5 + 0.002828663793103449*x**5.5 - 0.0018415178571428578*x**6.5  + 0.0012772722942073172*x**7.5 - 0.0009285135472074472*x**8.5 + 0.0006998889851120285*x**9.5 - 0.0005429801294359111*x**10.5 + 0.0004312515258789064*x**11.5 - 0.00034925299631038194*x**12.5
            else:
                hyperP = 0.0
                hyperM = 0.0
    
    if (scale_factor > 0.2):
        return 3.4494794123063873799*(1.0/omega_m-1.0)**0.666666666666666666666666666*math.sqrt(1.0+(1.0/scale_factor**3-1.0)*omega_m) + (hyperP*(4*scale_factor**3*(omega_m-1.0)-omega_m)-7.0*scale_factor**3*hyperM*(omega_m-1.0))/(scale_factor**5*(omega_m-1.0)-scale_factor**2*omega_m)
    else:
        return (scale_factor*(1.0-omega_m)**1.5*(1291467969*scale_factor**12*(omega_m-1.0)**4 + 1956769650*scale_factor**9*(omega_m-1.0)**3*omega_m + 8000000000*scale_factor**3*(omega_m-1.0)*omega_m**3 + 37490640625*omega_m**4))/(1.5625e10*omega_m**5);

# Calculates the growth rate f, EXACTLY as used in PICOLA (which may not be quite the same as the usual methods)
def GrowthRatePICOLA(redshift, omega_m):
    scale_factor = 1.0/(1.0+redshift)
    prefac = 1.0/(scale_factor**2*Ez(redshift, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0))
    part1 = (6.0*(1.0-omega_m)**(1.5))/GrowthFactorPICOLA(redshift, omega_m)
    part2 = ((3.0*omega_m)/(2.0*scale_factor))#*GrowthFactorPICOLA(redshift,omega_m)/GrowthFactorPICOLA(0.0,omega_m)
    return prefac*(part1-part2) 

# The numerator integrand for calculating Zeff
def ZeffConstNIntegrand1(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap)**2*redshift)/Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)

# The denominator integrand for calculating Zeff
def ZeffConstNIntegrand2(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap)**2)/Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)

# Calculate the effective redshift for constant n(z)
def ZeffConstN(redshift_low, redshift_high, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    numerator = integrate.quad(ZeffConstNIntegrand1, redshift_low, redshift_high, args=(omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap))[0]
    denominator = integrate.quad(ZeffConstNIntegrand2, redshift_low, redshift_high, args=(omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap))[0]
    return numerator/denominator



