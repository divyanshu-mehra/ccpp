import json
import numpy as np

# Example data containing a potential int64 value
data = {
    "Time": np.int64(0),
    "V1": -1.359807134,
    "V2": -0.072781173,
    "V3": 2.536346738,
    "V4": 1.378155224,
    "V5": -0.33832077,
    "V6": 0.462387778,
    "V7": 0.239598554,
    "V8": 0.098697901,
    "V9": 0.36378697,
    "V10": 0.090794172,
    "V11": -0.551599533,
    "V12": -0.617800856,
    "V13": -0.991389847,
    "V14": -0.311169354,
    "V15": 1.468176972,
    "V16": -0.470400525,
    "V17": 0.207971242,
    "V18": 0.02579058,
    "V19": 0.40399296,
    "V20": 0.251412098,
    "V21": -0.018306778,
    "V22": 0.277837576,
    "V23": -0.11047391,
    "V24": 0.066928075,
    "V25": 0.128539358,
    "V26": -0.189114844,
    "V27": 0.133558377,
    "V28": -0.021053053,
    "Amount": 149.62
}

# Convert all np.int64 values to int
data = {key: int(value) if isinstance(value, np.int64) else value for key, value in data.items()}

# Convert the dictionary to JSON
json_data = json.dumps(data, indent=2)
print(json_data)
