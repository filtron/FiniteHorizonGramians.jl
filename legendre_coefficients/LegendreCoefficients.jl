module LegendreCoefficients

using Symbolics, LinearAlgebra

"""
    compute_numerator(z, q::Integer)

Computes the numerator polynomial for the qth diagonal Pade approximation of the exponential function
"""
function compute_numerator(z, q::Integer)
    q = big(q)
    den = zero(z)
    for j = 0:q
        den = den + factorial(2q - j) // factorial(q - j) * z^j // factorial(j)
    end
    #den = den * factorial(q) // factorial(2q) to get the usual Pade numerator 
    return den
end

"""
    compute_denominator(z, q::Integer) 

Computes the denominator polynomial for the qth diagonal Pade approximation of the exponential function
"""
compute_denominator(z, q::Integer) = compute_numerator(-z, q::Integer)

"""
    compute_legendre_numerators(z, q::Integer)

Computes the numerators of the coefficients of the qth order Legendre expansion of the exponential function
in the variable z. This probably breaks for q < 3.  
"""
function compute_legendre_numerators(z, q::Integer)

    sqr_norms = ones(Int, q + 1) + 2 * (0:q)

    cs = zeros(typeof(z), q + 1)
    cs[end] = z^q // (2q + 1)
    cs[end-1] = 2 * (2q + 1) // z * cs[end]
    cs[end-2] = 2 * (2q - 1) // z * cs[end-1]

    for k = q-3:-1:0
        cs[k+1] = 2 * (2k + 3) / z * cs[k+2] + cs[k+3]
    end

    cs = cs .* sqr_norms
    cs = cs .|> expand .|> simplify
    den = compute_denominator(z, q)

    csum = sum([cs[k+1] * (-1)^k for k = 0:q])

    @assert csum - den == 0

    return cs
end

"""
    make_coefficient_table(z, q::Integer)

Computes the table of coefficients for the Legendre expansion 
"""
function compute_coefficient_table(z, q::Integer)
    leg_nums = compute_legendre_numerators(z, q)
    pade_den = compute_denominator(z, q)
    pade_num = compute_numerator(z, q)
    sqr_norms = ones(Int, q + 1) + 2 * (0:q)

    pade_den = poly2coeff(z, pade_den, q)
    pade_num = abs.(pade_den)

    leg_nums = poly2coeff.(z, leg_nums, q)
    leg_nums = mapreduce(permutedims, vcat, leg_nums)

    return pade_num, leg_nums, sqr_norms
end


"""
    poly2coeff(z, poly, deg)

Computes a vector of coefficients for symbolic polinomials in the variable z.
"""
function poly2coeff(z, poly, deg)
    dict, _ = semipolynomial_form(poly, [z], deg)
    c = zeros(typeof(deg), deg + 1)
    for n = 1:deg+1
        @inbounds c[n] = haskey(dict, z^(n - 1)) ? dict[z^(n-1)] : 0
    end
    return c
end


export compute_numerator,
    compute_denominator, compute_legendre_numerators, compute_coefficient_table


end
