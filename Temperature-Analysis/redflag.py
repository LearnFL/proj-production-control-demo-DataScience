import pandas as pd

def redFlagLogger(*, ramp, tempDict, dataFrame, file_path):
    error_set = set() # For debugging, catches unique errors and exceptions.  
    red_flag_set =set() # To quickly visualize during which period a Red Flag was found and in which zone.
    periodList = ["010 days", "020 days", "030 days","040 days","050 days", "060 days", "070 days", "080 days"]
    index_for_period = 0
    for r in ramp:
            for i, j, k in zip(['TD', 'Top', 'Diss'],['td_max', 'top_max', 'diss_max'], ['td_min', 'top_min', 'diss_min']):
                t = 0
                for t in range(0, len(list(tempDict[j]))):
                    try:
                        r.loc[((r[i]>tempDict[j][t]) | (r[i]<tempDict[k][t])),"Flag"] = 1
                        if r.empty != True:
                            red_flag_set.add(f'{periodList[index_for_period]} period in {i}')  
                        dataFrame = pd.concat([dataFrame, r])
                        t+=1 
                    except Exception as e:
                        error_set.add(str(e))
                        continue
            del r
            index_for_period += 1        
    print(f'List of Exceptions:  {error_set}')    
    print(f'List of detected Red Flag: {sorted(red_flag_set, key=(lambda x: x[0:4]))}')          
    dataFrame.to_csv(file_path /'Flagged.csv') 
    del dataFrame
