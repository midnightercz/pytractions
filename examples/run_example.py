import sys

import examples.tractors

if __name__ == "__main__":
    mod = __import__(f"examples.tractors", locals(), globals(), [sys.argv[1]], 0)
    tmod = getattr(mod, sys.argv[1])
    tmod.main()

