#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:38:14 2017

@author: hans-christianthorsen-meyer
"""

import pandas as pd

index = pd.date_range('1/1/2000', periods=9, freq='T')
series = pd.Series(range(9), index=index)
series.resample('3T').sum()
