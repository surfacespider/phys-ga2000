#!/usr/bin/env python
# coding: utf-8


import numpy as np

def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
       
       Taken directly from lecture notes
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))


value = np.float32(100.98763)


bitlist = get_bits(value)

sign = bitlist[0]
exponent = bitlist[1:9]
mantissa = bitlist[9:32]
template = """{value} decimal ->
    sign = {sign} 
    exponent = {exponent} 
    mantissa = {mantissa}"""
print(template.format(value=value, sign=sign, exponent=exponent, mantissa=mantissa))

print('calculating these exponent = 133 and mantissa = 4848043')
print('actual stored value 100.98763275146484375')
print('the difference is 2.75146484375 × 10^-6')

#my intial instinct was to convert it to 32 bit and then just subtract a higher accuracy representation
difference = value-100.98763
print('just printing the difference float 32 and whatever python saved my initial input as gives ',difference)
#these are very close




