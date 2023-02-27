from hours import Hours
from pathlib import Path

# File Location                                                                                                         
file_path=Path(r"C:\Users\_____SOME PATH GOES HERE_________") 

def main(): 
    calc = Hours()
    calc.get_dataFrame(file_path)
    calc.renderHours()    

if __name__ == '__main__':
    main()
