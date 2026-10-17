"""Dedicated YOLO training entry point using the validated dataset contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foodquality8.training import main

if __name__=='__main__':
    sys.argv.insert(1,'detector')
    main()
