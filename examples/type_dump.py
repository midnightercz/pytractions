import sys
import json

import examples.tractors

if __name__ == "__main__":
    mod = __import__(f"examples.tractors", locals(), globals(), [sys.argv[1]], 0)
    tmod = getattr(mod, sys.argv[1])
    #print(tmod.to_json())
    print(json.dumps(tmod.to_json()))

