from . import spec_help
HELINES_AIR = [10829.09114,10830.25010,10830.33977]
CALINES_AIR = [8498.02,8542.09,8662.14]

# Vacuum
CALINES_VAC = [spec_help.airtovac(i) for i in CALINES_AIR] 
HELINES_VAC = [spec_help.airtovac(i) for i in HELINES_AIR] 
