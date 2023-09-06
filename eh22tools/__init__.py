import numpy as np

# Version 1.3 - February 2023
# AUTHOR: Roberta C. Hamme (University of Victoria)
# This software is available from http://www.cambridge.org/emerson-hamme
# as part of Chemical Oceanography: Element Fluxes in the Sea (2022)
# by Steven R. Emerson and Roberta C. Hamme
# This software is licenced under Apache License 2.0
# It is provided "as is" without warranty of any kind.

def dens0(S,T):
    """
    Calculates density of seawater at atmospheric pressure and a
    hydrostatic pressure of 0 dbar (surface).

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]

    Returns
    -------
    density : array_like
           density of seawater [kg m^-3]

    Usage
    --------
    >>> import eh22tools as eh
    >>> density = eh.dens0(S,T)
    if S and T are not singular they must have same dimensions

    References
    ----------
    .. [1] Unesco 1983. Algorithms for computation of fundamental properties of
       seawater, 1983. Unesco Tech. Pap. in Mar. Sci., No. 44, 53 pp.

    .. [2] Millero, F.J. and  Poisson, A.
       International one-atmosphere equation of state of seawater.
       Deep-Sea Res. 1981. Vol28A(6) pp625-629.

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('dens0: S & T must have same dimensions or be singular')

    # Convert temperature on ITS-90 scale to IPTS-68 scale for use with density equations
    T = 1.00024 * T

    # Calculate density of pure water
    a0 = 999.842594
    a1 =   6.793952e-2
    a2 =  -9.095290e-3
    a3 =   1.001685e-4
    a4 =  -1.120083e-6
    a5 =   6.536332e-9
    dens_0sal = a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4 + a5*T**5;

    # Correct density for salinity
    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9
    c0 = -5.72466e-3
    c1 =  1.0227e-4
    c2 = -1.6546e-6
    d0 = 4.8314e-4
    return(dens_0sal + (b0 + b1*T + b2*T**2 + b3*T**3 + b4*T**4)*S + (c0 + c1*T + c2*T**2)*S**1.5 + d0*S**2)


def dynvisc(S,T):
    """
    Calculates dynamic viscosity of seawater at the given salinity and
    temperature of the water and a hydrostatic pressure of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]

    Returns
    -------
    dyn_viscosity : array_like
           dynamic viscosity of seawater [Pa s = kg m^-1 s^-1]

    Usage
    --------
    >>> import eh22tools as eh
    >>> dyn_viscosity = eh.dynvisc(S,T)
    if S and T are not singular they must have same dimensions

    References
    ----------
    .. [1] Sharqawy, M. H., J. H. Lienhard, and S. M. Zubair (2010) The
       thermophysical properties of seawater: a review of existing
       correlations and data, Desalination and Water Treatment, 16, 354-380.

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('dynvisc: S & T must have same dimensions or be singular')

    # Dynamic viscosity of pure water in atm
    dyn_viscosity_0sal = 4.2844e-5 + (0.157*(T + 64.993)**2 - 91.296)**-1

    # Correct dynamic viscosity for salinity
    A = 1.541 + 1.998e-2*T - 9.52e-5*T**2
    B = 7.974 - 7.561e-2*T + 4.724e-4*T**2
    return(dyn_viscosity_0sal * (1 + A*S/1000 + B*(S/1000)**2))


def kinvisc(S,T):
    """
    Calculates kinematic viscosity of seawater at the given salinity and
    temperature of the water and a hydrostatic pressure of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]

    Returns
    -------
    kin_viscosity : array_like
           kinematic viscosity of seawater [m^2 s^-1]

    Usage
    --------
    >>> import eh22tools as eh
    >>> kin_viscosity = eh.kinvisc(S,T)
    if S and T are not singular they must have same dimensions

    References
    ----------
    .. [1] Sharqawy, M. H., J. H. Lienhard, and S. M. Zubair (2010) The
       thermophysical properties of seawater: a review of existing
       correlations and data, Desalination and Water Treatment, 16, 354-380.

    .. [2] Unesco 1983. Algorithms for computation of fundamental properties of
       seawater, 1983. Unesco Tech. Pap. in Mar. Sci., No. 44, 53 pp.

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)


    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('kinvisc: S & T must have same dimensions or be singular')

    dyn_viscosity = dynvisc(S,T)
    density = dens0(S,T)
    return(dyn_viscosity / density)


def vpress(S,T,*,units='atm'):
    """
    Calculates the vapor pressure of seawater at the given salinity and temperature
    of the water and a hydrostatic pressure of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]

    Kwargs
    ----------
    units = desired output units formatted as a character string
            'atm' is default if no value supplied
            possible values: 'atm','Pa','hPa','mbar','Torr','mm-Hg','psi'

    Returns
    -------
    vapor_press : array_like
           vapor pressure of seawater [units from kwarg, atm is default]

    Usage
    --------
    >>> import eh22tools as eh
    >>> vapor_press = eh.vpress(S,T) # using default units of atm
    >>> vapor_press = eh.vpress(S,T,units='Pa') # using alterative units, example for Pa
    if S and T are not singular they must have same dimensions

    References
    ----------
    .. [1] Guide to Best Practices for Ocean CO2 Measurements
       Dickson, A.G., C.L. Sabine, J.R. Christian (Eds.) 2007
       PICES Special Publication 3, 191pp.
       Chapter 5: Physical and thermodynamic data

    .. [2] Wagner, W., A. Pruss (2002) The IAPWS formulation 1995
       for the thermodynamic properties of ordinary water substance for
       general and scientific use, J. Phs. Chem. Ref. Data, 31, 387-535.

    .. [3] Millero, F.J. (1974) Seawater as a multicomponent electrolyte
       solution, pp.3-80.  In: The Sea, Vol. 5, E.D. Goldberg Ed.

    """

    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('vpress: S & T must have same dimensions or be singular')

    # check that units is one of the supported values
    if not (units in {'atm','Pa','hPa','mbar','Torr','mm-Hg','psi'}):
        raise TypeError('vpress: Expected input parameter units to match one of these values:\n''atm'',''Pa'',''hPa'',''mbar'',''Torr'',''mm-Hg'', or ''psi''')

    # calculate scaled temperatures
    TK = T+273.15
    Tmod = 1-TK/647.096

    # Calculate value of Wagner and Pruss polynomial
    Wagner = -7.85951783*Tmod +1.84408259*Tmod**1.5 -11.7866497*Tmod**3 +22.6807411*Tmod**3.5 -15.9618719*Tmod**4 +1.80122502*Tmod**7.5

    # Vapor pressure of pure water in atm
    vapor_0sal_atm = np.exp(Wagner * 647.096 / TK) * 217.75

    # Correct vapor pressure for salinity
    molality = 31.998 * S /(1e3-1.005*S)
    osmotic_coef = 0.90799 -0.08992*(0.5*molality) +0.18458*(0.5*molality)**2 -0.07395*(0.5*molality)**3 -0.00221*(0.5*molality)**4
    vapor_press_atm = vapor_0sal_atm * np.exp(-0.018 * osmotic_coef * molality)

    # Convert to desired units
    if units == 'atm':
        vapor_press = vapor_press_atm
    elif units == 'Pa':
        vapor_press = vapor_press_atm * 101325
    elif units == 'hPa':
        vapor_press = vapor_press_atm * 1013.25
    elif units == 'mbar':
        vapor_press = vapor_press_atm * 1013.25
    elif units == 'Torr':
        vapor_press = vapor_press_atm * 760
    elif units == 'mm-Hg':
        vapor_press = vapor_press_atm * 760
    elif units == 'psi':
        vapor_press = vapor_press_atm * 14.6959

    return(vapor_press)


def gasdiff(S,T,gas):
    """
    Calculates the molecular diffusion coefficient of a gas in seawater
    at the given salinity and temperature of the water and a hydrostatic
    pressure of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    gas : string (single value only)
           gas for which to calculate diffusion coefficient
           possible values: 'N2','O2','Ar','CO2','Ne','He','CH4','Kr','Xe','CFC-12','CFC-11','SF6','DMS','Rn'

    Returns
    -------
    gas_diffusion_coef : array_like
           molecular diffusion coefficient of the gas [cm^2 s^-1]

    Usage
    --------
    >>> import eh22tools as eh
    >>> gas_diffusion_coef = eh.gasdiff(S,T,'Ne') # example for Ne
    if S and T are not singular they must have same dimensions
    gas must be a single string

    References
    ----------
    .. [1] Appendix E, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press
       and references therein

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('gasdiff: S & T must have same dimensions or be singular')

    # check that gas is one of the supported values
    if not (gas in {'N2','O2','Ar','CO2','Ne','He','CH4','Kr','Xe','CFC-12','CFC-11','SF6','DMS','Rn'}):
        raise TypeError('gasdiff: Expected input parameter gas to match one of these values:\n''N2'',''O2'',''Ar'',''CO2'',''Ne'',''He'',''CH4'',''Kr'',''Xe'',''CFC-12'',''CFC-11'',''SF6'',''DMS'',''Rn''')

    # Set constants for calculation
    if gas == 'N2':
        A = 3412e-5
        Ea = 18.50e3
    elif gas == 'O2':
        A = 4286e-5
        Ea = 18.70e3
    elif gas == 'Ar':
        A = 2227e-5
        Ea = 16.68e3
    elif gas == 'CO2':
        A = 5019e-5
        Ea = 19.51e3
    elif gas == 'Ne':
        A = 1608e-5
        Ea = 14.84e3
    elif gas == 'He':
        A = 818e-5
        Ea = 11.70e3
    elif gas == 'CH4':
        A = 3047e-5
        Ea = 18.36e3
    elif gas == 'Kr':
        A = 6393e-5
        Ea = 20.20e3
    elif gas == 'Xe':
        A = 9007e-5
        Ea = 21.61e3
    elif gas == 'CFC-12':
        A = 3600e-5
        Ea = 20.1e3
    elif gas == 'CFC-11':
        A = 1500e-5
        Ea = 18.1e3
    elif gas == 'SF6':
        A = 2900e-5
        Ea = 19.3e3
    elif gas == 'DMS':
        A = 2000e-5
        Ea = 18.1e3
    elif gas == 'Rn':
        A = 15877e-5
        Ea = 23.26e3


    # Molecular diffusion coefficient of gas in pure water
    TK = T+273.15
    gas_diffusion_coef_0sal = A * np.exp(-Ea/(8.314462*TK))

    # Correct molecular diffusion coefficient for salinity
    return(gas_diffusion_coef_0sal * (1-0.049*S/35.5))


def schmidt(S,T,gas):
    """
    Calculates the Schmidt number of a gas in seawater at the given
    salinity and temperature of the water and a hydrostatic pressure
    of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    gas : string (single value only)
           gas for which to calculate Schmidt number
           possible values: 'N2','O2','Ar','CO2','Ne','He','CH4','Kr','Xe','CFC-12','CFC-11','SF6','DMS','Rn'

    Returns
    -------
    gas_Schmidt_number : array_like
           Schmidt number of the gas [unitless]

    Usage
    --------
    >>> import eh22tools as eh
    >>> gas_Schmidt_number = eh.schmidt(S,T,'Ne') # example for Ne
    if S and T are not singular they must have same dimensions
    gas must be a single string

    References
    ----------
    .. [1] Appendix E, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press
       and references therein

    .. [2] Sharqawy, M. H., J. H. Lienhard, and S. M. Zubair (2010) The
       thermophysical properties of seawater: a review of existing
       correlations and data, Desalination and Water Treatment, 16, 354-380.

    .. [3] Unesco 1983. Algorithms for computation of fundamental properties of
       seawater, 1983. Unesco Tech. Pap. in Mar. Sci., No. 44, 53 pp.

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('schmidt: S & T must have same dimensions or be singular')

    # check that gas is one of the supported values
    if not (gas in {'N2','O2','Ar','CO2','Ne','He','CH4','Kr','Xe','CFC-12','CFC-11','SF6','DMS','Rn'}):
        raise TypeError('schmidt: Expected input parameter gas to match one of these values:\n''N2'',''O2'',''Ar'',''CO2'',''Ne'',''He'',''CH4'',''Kr'',''Xe'',''CFC-12'',''CFC-11'',''SF6'',''DMS'',''Rn''')

    gas_diffusion_coef = gasdiff(S,T,gas)
    kin_viscosity = kinvisc(S,T)
    return(1e4 * kin_viscosity / gas_diffusion_coef)


def gassat(S,T,gas,units='umol/kg',molfract=None):
    """
    Calculates the concentration of a gas in seawater at equilibrium with
    the atmosphere at the given salinity and temperature of the water and
    a hydrostatic pressure of 0 dbar (surface).   For gases with variable
    atmospheric concentration, the user may supply the mole fraction.

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    gas : string (single value only)
           gas for which to calculate equilibrium concentration
           possible values: 'N2','O2','Ar','CO2','Ne','He','CH4','Kr','N2O','Xe','CFC-12','CFC-11','SF6'

    Kwargs
    ----------
    units = desired output units formatted as a character string
        'umol/kg' is default if no value supplied
        possible values: 'mol/kg','mmol/kg','umol/kg','nmol/kg','pmol/kg',
           'fmol/kg','mol/m3','mmol/m3','umol/m3','nmol/m3','pmol/m3','fmol/m3'
    molfract = for gases with variable atmospheric concentrations
           (CH4, N2O, CFC-12, CFC-11, and SF6), the user may provide a dry
           atmospheric mole fraction (units of mol-gas / mol-atm);
        for CO2, provide the atmospheric fugacity (atm);
        any provided value is ignored for other gases.
        default for gases with variable concentration is the NOAA Annual
           Greenhouse Gas Index if no value supplied (see Table D.1 in
           Emerson and Hamme for all default mole fractions used)

    Returns
    -------
    gas_equil_conc : array_like
           equilibrium concentration of the gas in seawater [units from kwarg, umol/kg is default]

    Usage
    --------
    >>> import eh22tools as eh
    >>> gas_equil_conc = eh.gassat(S,T,'O2') # example for O2 using default units of atm
    >>> gas_equil_conc = eh.gassat(S,T,'Xe',units='nmol/kg') # example for Xe using units of nmol/kg
    >>> gas_equil_conc = eh.gassat(S,T,'N2O',molfract=3.50e-7) # example for N2O using alternate mole fraction
    if S and T are not singular they must have same dimensions
    gas and units must be a single string, molfract must be a single number

    References
    ----------
    .. [1] Appendix E, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press
       and references therein

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('gassat: S & T must have same dimensions or be singular')

    # check that units is one of the supported values
    if not (units in {'mol/kg','mmol/kg','umol/kg','nmol/kg','pmol/kg','fmol/kg','mol/m3','mmol/m3','umol/m3','nmol/m3','pmol/m3','fmol/m3'}):
        raise TypeError('gassat: Expected input parameter units to match one of these values:\n''mol/kg'',''mmol/kg'',''umol/kg'',''nmol/kg'',''pmol/kg'',''fmol/kg'',''mol/m3'',''mmol/m3'',''umol/m3'',''nmol/m3'',''pmol/m3'',''fmol/m3''')

    # check that gas is one of the supported values
    if not (gas in {'N2','O2','Ar','CO2','Ne','He','CH4','Kr','N2O','Xe','CFC-12','CFC-11','SF6'}):
        raise TypeError('gassat: Expected input parameter gas to match one of these values:\n''N2'',''O2'',''Ar'',''CO2'',''Ne'',''He'',''CH4'',''Kr'',''N2O'',''Xe'',''CFC-12'',''CFC-11'',''SF6''')

    # set molfract values if needed and check size if supplied
    if molfract == None:
        if gas == 'CO2':
            molfract = 4.10e-4
        elif gas == 'CH4':
            molfract = 1.87e-6
        elif gas == 'N2O':
            molfract = 3.32e-7
        elif gas == 'CFC-12':
            molfract = 5.01e-10
        elif gas == 'CFC-11':
            molfract = 2.26e-10
        elif gas == 'SF6':
            molfract = 9.96e-12
    elif np.asanyarray(molfract).size > 1:
        raise TypeError('gassat: molfract must be singular')

    # calculate scaled temperature
    TS = np.log((298.15-T)/(273.15+T))
    TK = T+273.15

    # perform calculation for units of umol/kg
    if gas == 'N2':
        gas_equil_conc_umolkg = np.exp(6.42931 + 2.92704*TS + 4.32531*TS**2 + 4.69149*TS**3 + S*(-7.44129e-3 - 8.02566e-3*TS  -1.46775e-2*TS**2))
    elif gas == 'O2':
        gas_equil_conc_umolkg = np.exp(5.80871 + 3.20291*TS + 4.17887*TS**2 + 5.10006*TS**3 - 9.86643e-2*TS**4 + 3.80369*TS**5 + S*(-7.01577e-3 - 7.70028e-3*TS - 1.13864e-2*TS**2 - 9.51519e-3*TS**3) - 2.75915e-7*S**2)
    elif gas == 'Ar':
        gas_equil_conc_umolkg = np.exp(2.79150 + 3.17609*TS + 4.13116*TS**2 + 4.90379*TS**3 + S*(-6.96233e-3 - 7.66670e-3*TS - 1.16888e-2*TS**2))
    elif gas == 'CO2':
        henrys_law_const = henryconst(S,T,'CO2')
        gas_equil_conc_umolkg = molfract * henrys_law_const * 1e6
    elif gas == 'Ne':
        gas_equil_conc_umolkg = np.exp(2.18156 + 1.29108*TS + 2.12504*TS**2 + S*(-5.94737e-3 - 5.13896e-3*TS))*1e-3
    elif gas == 'He':
        gas_equil_conc_umolkg = np.exp(-178.1424 + 217.5991*(100/TK) + 140.7506*np.log(TK/100) - 23.01954*(TK/100) + S*(-0.038129 + 0.019190*(TK/100) - 0.0026898*(TK/100)**2) -2.55e-6*S**2)*1e6
    elif gas == 'CH4':
        gas_equil_conc_umolkg = np.exp(np.log(molfract) - 417.5053 + 599.8626*(100/TK) + 380.3636*np.log(TK/100) - 62.0764*(TK/100) + S*(-0.064236 + 0.034980*(TK/100) - 0.0052732*(TK/100)**2))*1e-3
    elif gas == 'Kr':
        gas_equil_conc_umolkg = np.exp(-112.6840 + 153.5817*(100/TK) + 74.4690*np.log(TK/100) - 10.0189*(TK/100) + S*(-0.011213 - 0.001844*(TK/100) + 0.0011201*(TK/100)**2))/0.0223518
    elif gas == 'N2O':
        gas_equil_conc_umolkg = np.exp(np.log(molfract) - 168.2459 + 226.0894*(100/TK) + 93.2817*np.log(TK/100) - 1.48693*(TK/100)**2 + S*(-0.060361 + 0.033765*(TK/100) - 0.0051862*(TK/100)**2))*1e6
    elif gas == 'Xe':
        gas_equil_conc_umolkg = np.exp(-224.5100 + 292.8234*(100/TK) + 157.6127*np.log(TK/100) - 22.66895*(TK/100) + S*(-0.084915 + 0.047996*(TK/100) - 0.0073595*(TK/100)**2) +6.69e-6*S**2)*1e6
    elif gas == 'CFC-12':
        gas_equil_conc_umolkg = np.exp(np.log(molfract)-220.2120 + 301.8695*(100/TK)+114.8533*np.log(TK/100)-1.39165*(TK/100)**2 + S*(-0.147718 + 0.093175*(TK/100)-0.0157340*(TK/100)**2)) * 1e6
    elif gas == 'CFC-11':
        gas_equil_conc_umolkg = np.exp(np.log(molfract) - 232.0411 + 322.5546*(100/TK) + 120.4956*np.log(TK/100) - 1.39165*(TK/100)**2 + S*(-0.146531 + 0.093621*(TK/100) - 0.0160693*(TK/100)**2))*1e6
    elif gas == 'SF6':
        gas_equil_conc_umolkg = np.exp(np.log(molfract) - 82.1639 + 120.152*(100/TK) + 30.6372*np.log(TK/100) + S*(0.0293201 - 0.0351974*(TK/100) + 0.00740056*(TK/100)**2))*1e6

    # convert units if needed
    if units == 'umol/kg':
        gas_equil_conc = gas_equil_conc_umolkg
    elif units == 'mol/kg':
        gas_equil_conc = gas_equil_conc_umolkg * 1e-6
    elif units == 'mmol/kg':
        gas_equil_conc = gas_equil_conc_umolkg * 1e-3
    elif units == 'nmol/kg':
        gas_equil_conc = gas_equil_conc_umolkg * 1e3
    elif units == 'pmol/kg':
        gas_equil_conc = gas_equil_conc_umolkg * 1e6
    elif units == 'fmol/kg':
        gas_equil_conc = gas_equil_conc_umolkg * 1e9
    elif units == 'mol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * 1e-6 * dens0(S,T)
    elif units == 'mmol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * 1e-3 * dens0(S,T)
    elif units == 'umol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * dens0(S,T)
    elif units == 'nmol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * 1e3 * dens0(S,T)
    elif units == 'pmol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * 1e6 * dens0(S,T)
    elif units == 'fmol/m3':
        gas_equil_conc = gas_equil_conc_umolkg * 1e9 * dens0(S,T)

    return(gas_equil_conc)


def henryconst(S,T,gas):
    """
    Calculates the Henry's Law constant of a gas in seawater at the given
    salinity and temperature of the water and a hydrostatic pressure
    of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    gas : string (single value only)
           gas for which to calculate Henry's Law constant
           possible values: 'N2','O2','Ar','CO2','Ne','He','CH4','Kr','N2O','Xe','CFC-12','CFC-11','SF6','DMS'

    Returns
    -------
    henrys_law_const : array_like
           Henry's Law constant for the gas in seawater [mol kg^-1 atm^-1]

    Usage
    --------
    >>> import eh22tools as eh
    >>> henrys_law_const = eh.henryconst(S,T,'Ne') # example for Ne
    if S and T are not singular they must have same dimensions
    gas must be a single string

    References
    ----------
    .. [1] Appendix E, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press
       and references therein

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('henryconst: S & T must have same dimensions or be singular')

    # check that gas is one of the supported values
    if not (gas in {'N2','O2','Ar','CO2','Ne','He','CH4','Kr','N2O','Xe','CFC-12','CFC-11','SF6','DMS'}):
        raise TypeError('henryconst: Expected input parameter gas to match one of these values:\n''N2'',''O2'',''Ar'',''CO2'',''Ne'',''He'',''CH4'',''Kr'',''N2O'',''Xe'',''CFC-12'',''CFC-11'',''SF6'',''DMS''')

    # calculate scaled temperature
    TS = np.log((298.15-T)/(273.15+T))
    TK = T+273.15

    # perform calculation, mainly converting equil conc to Henry's law const
    if gas == 'N2':
        molfract = 7.8084e-1
        henrys_law_const = gassat(S,T,'N2',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'O2':
        molfract = 2.0946e-1
        henrys_law_const = gassat(S,T,'O2',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'Ar':
        molfract = 9.34e-3
        henrys_law_const = gassat(S,T,'Ar',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'CO2':
        henrys_law_const = np.exp(-60.2409 + 9345.17/TK + 23.3585*np.log(TK/100) + S*(0.023517 - 0.00023656*TK + 0.0047036*(TK/100)**2))
    elif gas == 'Ne':
        molfract = 1.818e-5
        henrys_law_const = gassat(S,T,'Ne',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'He':
        molfract = 5.24e-6
        henrys_law_const = gassat(S,T,'He',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'CH4':
        gas_equil_conc_nmolkg_1atm = np.exp(-417.5053 + 599.8626*(100/TK) + 380.3636*np.log(TK/100) - 62.0764*(TK/100)+ S*(-0.064236 + 0.034980*(TK/100) - 0.0052732*(TK/100)**2))
        henrys_law_const = gas_equil_conc_nmolkg_1atm * 1e-9 / (1-vpress(S,T))
    elif gas == 'Kr':
        molfract = 1.14e-6
        henrys_law_const = gassat(S,T,'Kr',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'N2O':
        gas_equil_conc_molkg_1atm = np.exp(-168.2459 + 226.0894*(100/TK) + 93.2817*np.log(TK/100) - 1.48693*(TK/100)**2+ S*(-0.060361 + 0.033765*(TK/100) - 0.0051862*(TK/100)**2))
        henrys_law_const = gas_equil_conc_molkg_1atm / (1-vpress(S,T))
    elif gas == 'Xe':
        molfract = 8.7e-8
        henrys_law_const = gassat(S,T,'Xe',units='mol/kg') / ((1-vpress(S,T)) * molfract)
    elif gas == 'CFC-12':
        gas_equil_conc_molkg_1atm = np.exp(-220.2120 + 301.8695*(100/TK)+114.8533*np.log(TK/100)-1.39165*(TK/100)**2 + S*(-0.147718 + 0.093175*(TK/100)-0.0157340*(TK/100)**2))
        henrys_law_const = gas_equil_conc_molkg_1atm / (1-vpress(S,T))
    elif gas == 'CFC-11':
        gas_equil_conc_molkg_1atm = np.exp(-232.0411 + 322.5546*(100/TK) + 120.4956*np.log(TK/100) - 1.39165*(TK/100)**2 + S*(-0.146531 + 0.093621*(TK/100) - 0.0160693*(TK/100)**2))
        henrys_law_const = gas_equil_conc_molkg_1atm / (1-vpress(S,T))
    elif gas == 'SF6':
        gas_equil_conc_molkg_1atm = np.exp(-82.1639 + 120.152*(100/TK) + 30.6372*np.log(TK/100) + S*(0.0293201 - 0.0351974*(TK/100) + 0.00740056*(TK/100)**2))
        henrys_law_const = gas_equil_conc_molkg_1atm / (1-vpress(S,T))
    elif gas == 'DMS':
        henrys_law_0sal = np.exp(3463 / TK - 12.20)
        henrys_law_345sal = np.exp(3547 / TK - 12.64)
        henrys_law_const = ((henrys_law_345sal - henrys_law_0sal) / 34.5 * S + henrys_law_0sal) / dens0(S,T) * 1000

    return(henrys_law_const)


def dissconst(S,T,P,reaction):
    """
    Calculates the equilibrium constants for the dissociation reactions of
    the carbonate and borate buffer systems, and the solubility products
    for aragonite and calcite, at the given salinity, temperature, and
    hydrostatic pressure of the water.  These constants are based on the
    "total" pH scale.

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    P : array_like
            pressure [dbar (decibars), use 0 for surface]
            Note that pressure in dbar is ~1-2% greater than depth in m.
    reaction : string (single value only)
           reaction for which to calculate the dissociation constant
       use the following to specify:
           'K1' = first dissociation constant for carbonic acid (CO2 + H2O <--> HCO3- + H+)
           'K2' = second dissociation constant for carbonic acid (HCO3- <--> CO3-- + H+)
           'KB' = dissociation constant for boric acid (B(OH)3 + H2O <--> B(OH)4- + H+)
           'KW' = dissociation constant for water (H2O <--> OH- + H+)
           'Kcalcite' = solubility product for calcite (CaCO3 <--> Ca++ + CO3--)
           'Karagonite' = solubility product for aragonite (CaCO3 <--> Ca++ + CO3--)

    Returns
    -------
    dissociation_const : array_like
           equilibrium constant or solubility product for the dissociation reaction
           [mol kg^-1 for K1, K2, KB; mol^2 kg^-2 for KW, Kcalcite, Karagonite]


    Usage
    --------
    >>> import eh22tools as eh
    >>> gas_equil_conc = eh.dissconst(S,T,P,'K1') # example for K1 reaction
    if S, T, and P are not singular they must have same dimensions
    reaction must be a single string

    References
    ----------
    .. [1] Appendices F & G, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press
       and references therein

    """

    S = np.asanyarray(S)
    T = np.asanyarray(T)
    P = np.asanyarray(P)

    # check S,T,P dimensions and verify they have the same shape or are singular
    if ((S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T))) or ((S.size>1) and (P.size>1) and (np.shape(S)!=np.shape(P))) or ((T.size>1) and (P.size>1) and (np.shape(T)!=np.shape(P))):
        raise TypeError('dissconst: S, T, & P must have same dimensions or be singular')

    # check that reaction is one of the supported values
    if not (reaction in {'K1','K2','KB','KW','Kcalcite','Karagonite'}):
        raise TypeError('dissconst: Expected input parameter reaction to match one of these values:\n''K1'',''K2'',''KB'',''KW'',''Kcalcite'',''Karagonite''')

    # calculate temperature in Kelvin
    TK = T+273.15
    # gas constant (cm^3 bar^-1 mol^-1 K^-1)
    R = 83.1446
    # convert pressure from dbar to bar
    P = P * 0.1

    # perform calculation for specified reaction
    if reaction == 'K1':
        dissociation_const_0dbar = 10**(-(3633.86/TK - 61.2172 + 9.6777*np.log(TK) - 0.011555*S + 0.0001152*S**2))
        return(dissociation_const_0dbar * np.exp((-(-25.5 + 0.1271*T) + 0.5*(-3.08e-3 + 8.77e-5*T)*P)*P/(R*TK)))
    elif reaction == 'K2':
        dissociation_const_0dbar = 10**(-(471.78/TK + 25.929 - 3.16967*np.log(TK) - 0.01781*S + 0.0001122*S**2))
        return(dissociation_const_0dbar * np.exp((-(-15.82 - 0.0219*T) + 0.5*(1.13e-3 + -1.475e-4*T)*P)*P/(R*TK)))
    elif reaction == 'KB':
        dissociation_const_0dbar = np.exp((-8966.9 - 2890.53*S**0.5 - 77.942*S + 1.728*S**1.5 - 0.0996*S**2)/TK + 148.0248 + 137.1942*S**0.5 + 1.62142*S - (24.4344 + 25.085*S**0.5 + 0.2474*S)*np.log(TK) + 0.053105*S**0.5*TK)
        return(dissociation_const_0dbar * np.exp((-(-29.48 + 0.1622*T - 2.608e-3*T**2) + 0.5*(-2.84e-3)*P)*P/(R*TK)))
    elif reaction == 'KW':
        dissociation_const_0dbar = np.exp(148.9652 - 13847.26/TK - 23.6521*np.log(TK) + (118.67/TK - 5.977 + 1.0495*np.log(TK))*S**0.5 - 0.01615*S)
        return(dissociation_const_0dbar * np.exp((-(-25.60 + 0.2324*T - 3.6246e-3*T**2) + 0.5*(-5.13e-3 + 7.94e-5*T)*P)*P/(R*TK)))
    elif reaction == 'Kcalcite':
        dissociation_const_0dbar = 10**(-(171.9065 + 0.077993*TK - 2839.319/TK - 71.595*np.log10(TK) + (0.77712 - 0.0028426*TK - 178.34/TK)*S**0.5 + 0.07711*S - 0.0041249*S**1.5))
        return(dissociation_const_0dbar * np.exp((-(-48.76 + 0.5304*T) + 0.5*(-1.176e-2 + 3.692e-4*T)*P)*P/(R*TK)))
    elif reaction == 'Karagonite':
        dissociation_const_0dbar = 10**(-(171.945 + 0.077993*TK - 2903.293/TK - 71.595*np.log10(TK) + (0.068393 - 0.0017276*TK - 88.135/TK)*S**0.5 + 0.10018*S - 0.0059415*S**1.5))
        return(dissociation_const_0dbar * np.exp((-(-45.96 + 0.5304*T) + 0.5*(-1.176e-2 + 3.692e-4*T)*P)*P/(R*TK)))


def carbeq(S,T,P,DIC,Alk):
    """
    Calculates concentrations of carbonate species including fCO2 and pH from
    DIC and Alkalinity as a function of temperature, salinity, and pressure.
    Also calculates saturation state for calcite and aragonite.
    Considers only carbon, boron, and water ions in alkalinity, using
    equations from Appendix 5A.1 (b) of Emerson and Hamme (2022).
    This function requires simpler inputs than CO2SYS and is suitable for
    many educational applications. A comparison between results of this
    program and CO2SYS can be made by calculating the carbonate species for
    the conditions in Table 5.5 of E&H(22).  For surface waters they are
    identical, but for deeper waters they are slightly different because of
    the silicic acid and phosphate contributions

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    P : array_like
            pressure [dbar (decibars), use 0 for surface]
            Note that pressure in dbar is ~1-2% greater than depth in m.
    DIC : array_like
           dissolved inorganic carbon [micromol kg^-1]
    Alk : array_like
           total alkalinity [microeq kg^-1]

    Returns
    -------
    fCO2 : numpy array
           fugacity of CO2  [microatm]
    pH : numpy array
           -log10 of hydrogen ion concentration [total pH scale]
    CO2 : numpy array
           concentration of dissolved CO2 and H2CO3  [micromol kg^-1]
    HCO3 : numpy array
           concentration of dissolved HCO3  [micromol kg^-1]
    CO3 : numpy array
           concentration of dissolved CO3  [micromol kg^-1]
    omega_cal : numpy array
           degree of saturation for calcite (omega < 1 is undersaturated)
    omega_arag : numpy array
           degree of saturation for aragonite (omega < 1 is undersaturated)

    Usage
    --------
    >>> import eh22tools as eh
    >>> fCO2, pH, CO2, HCO3, CO3, omega_cal, omega_arg = eh.carbeq(S,T,P,DIC,Alk)
    if S, T, P, DIC, and Alk are not singular they must have same dimensions

    References
    ----------
    .. [1] Equilibrium constants and equations from: Dickson, A.G., Sabine, C.L. and Christian, J.R. (Eds.) 2007. Guide to
       best practices for ocean CO2 measurements. PICES Special Publication 3, 191 pp.

    .. [2] Calcite and aragonite solubility from Mucci "The solubility of calcite and aragonite in seawater at various
       salinities, temperatures, and one atmosphere total pressure" 1983. Am. J. Sci., 283.

    .. [3] Mimization routine and pressure corrections from Lewis, E., D.W.R Wallace, CO2SYS: Program developed for CO2
       system calculations

    """

    S = np.atleast_1d(S)
    T = np.atleast_1d(T)
    P = np.atleast_1d(P)
    DIC = np.atleast_1d(DIC)
    Alk = np.atleast_1d(Alk)

    # check S,T,P,DIC,Alk dimensions and verify they have the same shape or are singular
    if ((S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T))) or ((S.size>1) and (P.size>1) and (np.shape(S)!=np.shape(P))) or ((S.size>1) and (DIC.size>1) and (np.shape(S)!=np.shape(DIC))) or ((S.size>1) and (Alk.size>1) and (np.shape(S)!=np.shape(Alk))) or ((T.size>1) and (P.size>1) and (np.shape(T)!=np.shape(P))) or ((T.size>1) and (DIC.size>1) and (np.shape(T)!=np.shape(DIC))) or ((T.size>1) and (Alk.size>1) and (np.shape(T)!=np.shape(Alk))) or ((P.size>1) and (DIC.size>1) and (np.shape(P)!=np.shape(DIC))) or ((P.size>1) and (Alk.size>1) and (np.shape(P)!=np.shape(Alk))) or ((DIC.size>1) and (Alk.size>1) and (np.shape(DIC)!=np.shape(Alk))):
        raise TypeError('carbeq: S, T, P, DIC, & Alk must have same dimensions or be singular')

    # Do some basic unit conversions
    # convert DIC to mol/kg
    DIC = DIC * 0.000001
    # Convert Alkalinity to eq/kg
    Alk = Alk * 0.000001

    # Calculate ion concentrations and equilibrium constants
    # Calculate total borate (BT) from salinity (Uppström, L., Deep-Sea Research, 21,161-162, 1974)
    BT = 0.0004157 * S / 35        # [mol kg^-1]
    # Calculate total calcium from salinity (Riley, R.F., and M. Tongudai, Chem. Geol., 2, 263–269, 1967)
    Ca = 0.0102846 * S / 35        # [mol kg^-1]
    # Calculate Henry's Law coeff for CO2 (KH) from temp & sal (Weiss, R.F., Mar. Chem., 2, 203-215, 1974)
    KH = henryconst(S,T,'CO2')  # [mol kg^-1 atm^-1]
    # Calculate borate equil constant (KB) from temp & sal (Dickson, A.G., 1990, Deep-Sea Res., 37, 755-766)
    KB = dissconst(S,T,P,'KB')
    # Calculate carbonate equil constants (K1 & K2) from temp & sal (Lueker, T.J., A.G. Dickson, C.D. Keeling, 2000, Mar. Chem., 70, 105-119) these are on the total pH scale
    K1 = dissconst(S,T,P,'K1')
    K2 = dissconst(S,T,P,'K2')
    # Calculate water equil constant (KW) from temp & sal (Millero, F.J., 1995, Geochim. Cosmochim. Acta, 59, 661-677)
    KW = dissconst(S,T,P,'KW')
    # Calculate Ksp for calcite and argonite
    Ksp_calcite = dissconst(S,T,P,'Kcalcite')
    Ksp_aragonite = dissconst(S,T,P,'Karagonite')

    # Solve for H ion concentration using a minimization routine based on Newton's method following CO2SYS code
    pH = 8               # initial guess for minimization is pH = 8;
    pHTol = 0.0001       # final pH estimate must be within 0.0001 of true value
    deltapH = pHTol+1    # give an initially large deltapH to start the loop
    while np.any(np.abs(deltapH) > pHTol):
        H = 10**(-pH)
        HCO3 = DIC * K1 * H / (H**2 + K1 * H + K1 * K2)
        CO3 = DIC * K1 * K2 / (H**2 + K1 * H + K1 * K2)
        BOH4 = KB * BT / (H + KB)
        OH = KW / H
        AlkResid = Alk - HCO3 - 2*CO3 - BOH4 - OH + H
        # find Slope dTA/dpH; directly from CO2SYS code (this is not exact, but keeps all important terms);
        Slope = np.log(10)*(DIC*K1*H*(H*H + K1*K2 + 4*H*K2)/((H**2 + K1*H + K1*K2)**2) + BOH4*H/(KB + H) + OH + H)
        deltapH = AlkResid/Slope
        # to keep the jump from being too big
        while np.any(np.abs(deltapH) > 0.5):
            index = np.abs(deltapH) > 0.5
            deltapH[index] = deltapH[index]/2
        pH = pH + deltapH

    # calculate/convert units HCO3, CO3 and CO2, fCO2 and pH
    HCO3 = HCO3 * 1e6
    omega_cal = Ca * CO3 / Ksp_calcite
    omega_arag = Ca * CO3 / Ksp_aragonite
    CO3 = CO3 * 1e6
    CO2 = DIC/(1 + K1/H + K1*K2/(H*H)) * 1e6
    fCO2 = CO2 / KH

    return(fCO2,pH,CO2,HCO3,CO3,omega_cal,omega_arag)


def DICeq(S,T,Alk,fCO2):
    """
    Calculates DIC at surface in equilibrium with given Alk and fCO2
    at the given salinity and temperature of the water.  These constants are
    based on the "total" pH scale.

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    Alk : array_like
           total alkalinity [microeq kg^-1]
    fCO2 : array_like
           fugacity of CO2  [microatm]

    Returns
    -------
    DIC : array_like
           dissolved inorganic carbon [micromol kg^-1]

    Usage
    --------
    >>> import eh22tools_312 as eh
    >>> DIC = eh.DICeq(S,T,Alk,fCO2)
    if S, T, Alk, and fCO2 are not singular they must have same dimensions

    References
    ----------
    .. [1] Equilibrium constants and equations from: Dickson, A.G., Sabine, C.L. and Christian, J.R. (Eds.) 2007. Guide to
       best practices for ocean CO2 measurements. PICES Special Publication 3, 191 pp.

    .. [2] Calcite and aragonite solubility from Mucci "The solubility of calcite and aragonite in seawater at various
       salinities, temperatures, and one atmosphere total pressure" 1983. Am. J. Sci., 283.

    .. [3] Mimization routine and pressure corrections from Lewis, E., D.W.R Wallace, CO2SYS: Program developed for CO2
       system calculations

    """

    S = np.asanyarray(S)
    T = np.asanyarray(T)
    Alk = np.asanyarray(Alk)
    fCO2 = np.asanyarray(fCO2)

    # check S,T,Alk,fCO2 dimensions and verify they have the same shape or are singular
    if ((S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T))) or ((S.size>1) and (Alk.size>1) and (np.shape(S)!=np.shape(Alk))) or ((S.size>1) and (fCO2.size>1) and (np.shape(S)!=np.shape(fCO2))) or ((T.size>1) and (Alk.size>1) and (np.shape(T)!=np.shape(Alk))) or ((T.size>1) and (fCO2.size>1) and (np.shape(T)!=np.shape(fCO2))) or ((Alk.size>1) and (fCO2.size>1) and (np.shape(Alk)!=np.shape(fCO2))):
        raise TypeError('carbeq: S, T, Alk, & fCO2 must have same dimensions or be singular')

    # Do some basic unit conversions
    # Convert Alkalinity to eq/kg
    Alk = Alk * 0.000001
    # convert fCO2 to atm
    fCO2 = fCO2 * 0.000001

    # Calculate ion concentrations and equilibrium constants
    # Calculate total borate (BT) from salinity (Uppström, L., Deep-Sea Research, 21,161-162, 1974)
    BT = 0.0004157 * S / 35        # [mol kg^-1]
    # Calculate total calcium from salinity (Riley, R.F., and M. Tongudai, Chem. Geol., 2, 263–269, 1967)
    Ca = 0.0102846 * S / 35        # [mol kg^-1]
    # Calculate Henry's Law coeff for CO2 (KH) from temp & sal (Weiss, R.F., Mar. Chem., 2, 203-215, 1974)
    KH = henryconst(S,T,'CO2')  # [mol kg^-1 atm^-1]
    # Calculate borate equil constant (KB) from temp & sal (Dickson, A.G., 1990, Deep-Sea Res., 37, 755-766)
    KB = dissconst(S,T,0,'KB')
    # Calculate carbonate equil constants (K1 & K2) from temp & sal (Lueker, T.J., A.G. Dickson, C.D. Keeling, 2000, Mar. Chem., 70, 105-119) these are on the total pH scale
    K1 = dissconst(S,T,0,'K1')
    K2 = dissconst(S,T,0,'K2')
    # Calculate water equil constant (KW) from temp & sal (Millero, F.J., 1995, Geochim. Cosmochim. Acta, 59, 661-677)
    KW = dissconst(S,T,0,'KW')

    # Solve for H ion concentration using a minimization routine based on Newton's method following CO2SYS code
    pH = 8               # initial guess for minimization is pH = 8;
    pHTol = 0.0001       # final pH estimate must be within 0.0001 of true value
    deltapH = pHTol+1    # give an initially large deltapH to start the loop
    while np.any(np.abs(deltapH) > pHTol):
        H = 10**(-pH)
        HCO3 = KH * K1 * fCO2 / H
        CO3 = KH * K1 * K2 * fCO2 / (H**2)
        BOH4 = KB * BT / (H + KB)
        OH = KW / H
        AlkResid = Alk - HCO3 - 2*CO3 - BOH4 - OH + H
        # find Slope dTA/dpH; directly from CO2SYS code (this is not exact, but keeps all important terms);
        Slope = np.log(10) * (HCO3 + 4*CO3 + BOH4*H/(KB + H) + OH + H)
        deltapH = AlkResid/Slope
        # to keep the jump from being too big
        while np.any(np.abs(deltapH) > 0.5):
            index = np.abs(deltapH) > 0.5
            deltapH[index] = deltapH[index]/2
        pH = pH + deltapH

    # calculate/convert units HCO3, CO3 and CO2, fCO2 and pH
    DIC = (HCO3 + 2*CO3) * (H**2 + K1*H + K1*K2) / (K1 * (H + 2*K2));
    DIC = DIC * 1e6;

    return(DIC)

def iondiff(S,T,ion):
    """
    Calculates the molecular diffusion coefficient of an ion in seawater
    at the given salinity and temperature of the water and a hydrostatic
    pressure of 0 dbar (surface)

    Parameters
    ----------
    S : array_like
           practical salinity [(PSS-78 scale)]
    T : array_like
           temperature [degree C (ITS-90)]
    ion : string (single value only)
           ion for which to calculate diffusion coefficient
           possible values: 'H','K','Na','Ca','Mg','OH','Cl','SO4','HCO3'

    Returns
    -------
    ion_diffusion_coef : array_like
           molecular diffusion coefficient of the ion [cm^2 s^-1]

    Usage
    --------
    >>> import eh22tools as eh
    >>> ion_diffusion_coef = eh.iondiff(S,T,ion)
    if S and T are not singular they must have same dimensions
    ion must be a single string

    References
    ----------
    .. [1] Appendix E, Chemical Oceanography: Element Fluxes in the Sea
       Emerson, S.R., R.C. Hamme 2022, Cambridge University Press

    .. [2] Handbook of Chemistry and Physics (1992/93)
       Lide, D.E. (ed.), V.73. Cleveland:CRC Press.

    """
    S = np.asanyarray(S)
    T = np.asanyarray(T)

    # check S,T dimensions and verify they have the same shape or are singular
    if (S.size>1) and (T.size>1) and (np.shape(S)!=np.shape(T)):
        raise TypeError('iondiff: S & T must have same dimensions or be singular')

    # check that ion is one of the supported values
    if not (ion in {'H','K','Na','Ca','Mg','OH','Cl','SO4','HCO3'}):
        raise TypeError('iondiff: Expected input parameter ion to match one of these values:\n''H'',''K'',''Na'',''Ca'',''Mg'',''OH'',''Cl'',''SO4'',''HCO3''')

    # Set constants for calculation
    charge = 1
    if ion == 'H':
        lamb = 349.65e-4
    elif ion == 'K':
        lamb = 73.48e-4
    elif ion ==  'Na':
        lamb = 50.08e-4
    elif ion ==  'Ca':
        lamb = 59.47e-4
        charge = 2;
    elif ion ==  'Mg':
        lamb = 53.0e-4
        charge = 2
    elif ion ==  'OH':
        lamb = 198e-4
    elif ion ==  'Cl':
        lamb = 76.31e-4
    elif ion ==  'SO4':
        lamb = 80.0e-4
        charge = 2
    elif ion ==  'HCO3':
        lamb = 44.5e-4

    # Molecular diffusion coefficient of ions in pure water
    temp_K = T+273.15
    ion_diffusion_coef_0sal = 8.314462 * temp_K * lamb / charge / 96485.3329**2;

    # Correct molecular diffusion coefficient for salinity
    ion_diffusion_coef = ion_diffusion_coef_0sal * (1-0.049*S/35.5)

    # Convert from units of m^2 s^-1 to cm^2 s^-1
    return(ion_diffusion_coef*1e4)