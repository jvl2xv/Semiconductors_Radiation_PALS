""" This function returns the positron lifetime spectrum for a given set of
parameters which are the time bins ( t ) in ns , the time zero ( T0 ) in the
reference frame of t , the stardard deviations of the Gaussian components (
sigmaGaussian ) in ns , the shifts of the Gaussian components with one being zero
as reference ( shiftGaussian ) in ns , the normalized weights of the Gaussian
components ( weightGaussian ) , the decay rates of the exponential components in
inverse ns ( lamdaLifetime ) , the relative intensities of the lifetime
exponential terms ( relIntLifetime ) , the background ( B ) , and the experimental
maximum ( exprMax ) . I typically normalize the spectra ( so exprMax is 1) and B is
given with reference to the normalized spectra . I allow T0 to be fit for each
individual spectrum , as there is potential for some shift so the T0 from this
fit serves as a central starting point for that fitting . """

def f (t , T0 , sigmaGaussian , shiftGaussian , weightGaussian , lamdaLifetime , relIntLifetime , B , exprMax ) :
    vals = numpy . zeros ( len ( t ) )
    for lifetime_num in numpy . arange (0 , len ( lamdaLifetime ) ,1) :
        lamda = lamdaLifetime [ lifetime_num ]
        I = relIntLifetime [ lifetime_num ]
        A = I /(1.0/ lamda )
        for resolution_num in numpy . arange (0 , len ( sigmaGaussian ) ,1) :
            sigma = sigmaGaussian [ resolution_num ]
            shift = shiftGaussian [ resolution_num ]
            w = weightGaussian [ resolution_num ]
            u = t - shift - T0
            vals = vals + 0.5* A * w * numpy . exp ( - lamda * u + 0.5* lamda **2* sigma **2) *(1 - scipy.special.erf(( lamda * sigma **2 - u ) /( sqrt (2) * sigma ) ) )
            vals = vals/max(vals)
            vals = vals + B - vals[-1]
    return vals *(1.0/ max ( vals ) )




# This loop goes over potential Gaussian component parameters and returns the global minimum parameters to the objective function .

filename = '/Users/julielogan/Documents/mycode_python/PALS_Fitting/PALS-Ti/2022_05/2022-05-10_Si_pristine.dat'
# Fit out to here
max_index = 375
# Normalize the spectrum
exprMax = 1.0
# Read the spectrum and determine the time of maximum
counts = numpy . loadtxt ( filename , skiprows =10) [: ,0]
arg_max_counts = numpy . argmax ( savitzky_golay ( counts , 5 , 3) )
max_count = max ( counts )
# Get the background ( relative to the normalized value )
B = numpy . average ( counts [(max_index-10):max_index ]) / max ( counts )
# Get the applicable range of fitting ( wrt the experimental max ) and normalize
counts = counts [( arg_max_counts -30) : max_index ]
counts = counts / max ( counts )
# Get the time bins
time_axis_bins = numpy . arange (0 , len ( counts ) , 1)
ns_per_bin = numpy . loadtxt ( filename , skiprows =8 , max_rows =1) [0]
time_ns = time_axis_bins * ns_per_bin
# Loop over these potential lifetime parameters ( for silicon with a salt / positronium component of small fraction )
lt_lambda_bulk_vals = 1.0/ numpy . array ([0.218])
lt_lambda_2_vals = 1.0/ numpy . arange (0.32 ,3.0 , 0.2)
lt_I_2_vals = numpy . array ([0.005 , 0.01 , 0.015 , 0.02 , 0.025])

# Loop over these potential resolution function parameters
g_sigma_1_vals = numpy . arange (0.18 -0.03 , 0.18+0.03 , 0.01) /2.355
g_sigma_2_vals = numpy . arange (0.32 -0.03 , 0.32+0.03 , 0.01) /2.355
g_frac_1_vals = numpy . arange (0.2 , 0.45 , .05)
g_shift_2_vals = numpy . arange ( -0.05 ,0.0 ,0.01)
T0_vals = numpy . arange (0.69 -0.02 ,0.69+0.02 ,0.005) - 20* ns_per_bin
best_err = 1000000.0

for lt_lambda_bulk in lt_lambda_bulk_vals :
    for lt_lambda_2 in lt_lambda_2_vals :
        for lt_I_2 in lt_I_2_vals :
            lt_I_1 = (1.0 - lt_I_2 )
            lt_I = [ lt_I_1 , lt_I_2 ]
            lt_lambda_1 = ( lt_lambda_bulk - lt_I_2 * lt_lambda_2 ) / lt_I_1
            lt_lambda = [ lt_lambda_1 , lt_lambda_2 ]
            for g_frac_1 in g_frac_1_vals :
                g_fracs = [ g_frac_1 , (1.0 - g_frac_1 ) ]
                for g_sigma_1 in g_sigma_1_vals :
                    for g_sigma_2 in g_sigma_2_vals :
                        for g_shift_2 in g_shift_2_vals :
                            for T0 in T0_vals :
                                fit_vals = f ( time_ns , T0 , [ g_sigma_1 , g_sigma_2 ] , [0.0 , g_shift_2 ] , g_fracs , lt_lambda , lt_I , B ,exprMax )
                                err = numpy . sum ((( counts - fit_vals ) / numpy . sqrt (counts ) ) **2)
                                if err < best_err :
                                    best_err = err
                                    g_sigma_best = [ g_sigma_1 , g_sigma_2 ]
                                    g_weight_best = g_fracs
                                    g_shift_best = [0.0 , g_shift_2 ]
                                    T0_best = T0
                                    lt_lambda_best = [ lt_lambda_1 , lt_lambda_2 ]
                                    lt_I_best = [ lt_I_1 , lt_I_2 ]
    

# Fit the actual lifetime values

# Automatically uses g_sigma_best , g_shift_best , g_weight_best generated above

# If the first lifetime component drops to this level , it indicates that there is likely no bulk annihilation and that the spectrum should not be fit assuming bulk annihilation
lt1_lower_limit = 0.04
""" Iterate through all combinations of 2nd lifetime component inverse lifetimes (
inverse ns , lt_lambda_2_vals ) and intensities ( normalized fractional values ,
lt_I_2_vals ) , which the user defines and find the best fit to the normalized
experimental data ( counts ) . One inputs the bulk term decay rate ( inverse
lifetime ) in ns and the background for the normalized spectrum . This function
can be called with an iteration over potential T0 values as well . """ 


def iterate_2LT (counts , time_ns , B , lt_lambda_bulk , lt_lambda_2_vals , lt_I_2_vals):
    best_err = 10000
    for lt_lambda_2 in lt_lambda_2_vals :
        for lt_I_2 in lt_I_2_vals :
            lt_I_1 = (1.0 - lt_I_2 )
            if lt_I_1 > 0:
                lt_I = [ lt_I_1 , lt_I_2 ]
                lt_lambda_1 = ( lt_lambda_bulk - lt_I_2 * lt_lambda_2 ) / lt_I_1
                lt_lambda = [ lt_lambda_1 , lt_lambda_2 ]
                fit_vals = f ( time_ns , T0_best , g_sigma_best , g_shift_best , g_weight_best , lt_lambda , lt_I , B , exprMax )
                err = numpy . sum ((( counts - fit_vals ) / numpy . sqrt ( counts ) ) **2)
                if err < best_err :
                    best_err = err
                    lt_lambda_best = lt_lambda
                    lt_I_best = lt_I
                    return best_err , lt_lambda_best , lt_I_best
                
""" Iterate through all combinations of 2 nd lifetime component inverse lifetimes (
inverse ns , lt_lambda_2_vals ) and intensities ( normalized fractional values ,
lt_I_2_vals ) and 3 rd lifetime component inverse lifetimes ( inverse ns ,
lt_lambda_3_vals ) and intensities ( normalized fractional values , lt_I_3_vals ) ,
which the user defines and find the best fit to the normalized experimental
data ( counts ) . One inputs the bulk term decay rate ( inverse lifetime ) in ns and
the background for the normalized spectrum . This function can be called with
an iteration over potential T0 values as well . """ 

def iterate_3LT ( counts , time_ns , B , lt_lambda_bulk , lt_lambda_2_vals , lt_lambda_3_vals , lt_I_2_vals , lt_I_3_vals ) :
    best_err = 10000
    for lt_lambda_2 in lt_lambda_2_vals :
        for lt_lambda_3 in lt_lambda_3_vals :
            for lt_I_2 in lt_I_2_vals :
                for lt_I_3 in lt_I_3_vals :
                    lt_I_1 = (1.0 - lt_I_2 - lt_I_3 )
                    if lt_I_1 > 0:
                        lt_I = [ lt_I_1 , lt_I_2 , lt_I_3 ]
                        lt_lambda_1 = ( lt_lambda_bulk - lt_I_2 * lt_lambda_2 -
                        lt_I_3 * lt_lambda_3 ) / lt_I_1
                        if (1.0/ lt_lambda_1 > lt1_lower_limit ) :
                            lt_lambda = [ lt_lambda_1 , lt_lambda_2 , lt_lambda_3 ]
                            fit_vals = f ( time_ns , T0_best , g_sigma_best , g_shift_best , g_weight_best , lt_lambda , lt_I , B ,exprMax )
                            err = numpy.sum ((( counts - fit_vals ) / numpy . sqrt (counts ) ) **2)
                            if err < best_err :
                                best_err = err
                                lt_lambda_best = lt_lambda
                                lt_I_best = lt_I
    return best_err , lt_lambda_best , lt_I_best

""" Iterate through all combinations of 2 nd lifetime component inverse lifetimes (
inverse ns , lt_lambda_2_vals ) and intensities ( normalized fractional values ,
lt_I_2_vals ) ; 3 rd lifetime component inverse lifetimes ( inverse ns ,
lt_lambda_3_vals ) and intensities ( normalized fractional values , lt_I_3_vals ) ;
and 4 th lifetime component inverse lifetimes ( inverse ns , lt_lambda_4_vals ) and
intensities ( normalized fractional values , lt_I_4_vals ) which the user defines
and find the best fit to the normalized experimental data ( counts ) . One inputs
the bulk term decay rate ( inverse lifetime ) in ns and the background for the
normalized spectrum . This function can be called with an iteration over
potential T0 values as well . """ 

def iterate_4LT ( counts , time_ns , B , lt_lambda_bulk , lt_lambda_2_vals , lt_lambda_3_vals , lt_lambda_4_vals , lt_I_2_vals , lt_I_3_vals , lt_I_4_vals ) :
    best_err = 10000
    for lt_lambda_2 in lt_lambda_2_vals :
        for lt_lambda_3 in lt_lambda_3_vals :
            for lt_lambda_4 in lt_lambda_4_vals :
                for lt_I_2 in lt_I_2_vals :
                    for lt_I_3 in lt_I_3_vals :
                        for lt_I_4 in lt_I_4_vals :
                            lt_I_1 = (1.0 - lt_I_2 - lt_I_3 - lt_I_4 )
                            if lt_I_1 > 0:
                                lt_I = [ lt_I_1 , lt_I_2 , lt_I_3 , lt_I_4 ]
                                lt_lambda_1 = ( lt_lambda_bulk - lt_I_2 * lt_lambda_2 - lt_I_3 * lt_lambda_3 - lt_I_4 * lt_lambda_4 ) / lt_I_1
                                if (1.0/ lt_lambda_1 > lt1_lower_limit ) :
                                    lt_lambda = [ lt_lambda_1 , lt_lambda_2 ,
                                    lt_lambda_3 , lt_lambda_4 ]
                                    fit_vals = f ( time_ns , T0_best , g_sigma_best , g_shift_best , g_weight_best , lt_lambda , lt_I , B , exprMax )
                                    err = numpy . sum ((( counts - fit_vals ) / numpy.sqrt ( counts ) ) **2)
                                    if err < best_err :
                                        best_err = err
                                        lt_lambda_best = lt_lambda
                                        lt_I_best = lt_I
    return best_err , lt_lambda_best , lt_I_best

        