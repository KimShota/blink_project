
import numpy as np
import pandas as pd 
import os 
import glob 

# helper functions
def parse_time_column(series):
    return series.str.replace('time', '', regex=False).astype(float) / 1000.0

def parse_blink_column(series):
    return series.str.extract(r'(\d+)').astype(int).squeeze() #put all the blink values into an one dimensional array

def parse_cellID_column(series):  
    return series.str.extract(r'(\d+)').astype(float).squeeze();

#main function 
def analyze_blink_file(filepath): 
    df = pd.read_csv(filepath)

    df['Timestamp'] = parse_time_column(df['time'])
    df['Leftblink'] = parse_blink_column(df['leftBlink'])
    df['Rightblink'] = parse_blink_column(df['rightBlink'])
    df['cellID'] = parse_cellID_column(df['cellID'])

    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    blinks = [] #empty list 
    blink_during = False
    blink_start = None

    minimum_blink_interval = 0.25
    last_blink_end = -np.inf

    for _, row in df.iterrows(): 
        blinking = row['Leftblink'] == 1 or row['Rightblink'] == 1

        if blinking and not blink_during: 
            potential_start = row['Timestamp']
            if potential_start - last_blink_end >= 0.25: 
                blink_start = potential_start
                blink_during = True
        elif not blinking and blink_during: 
            blink_end = row['Timestamp']
            duration = blink_end - blink_start
            if blink_end > blink_start and 0.08 < duration < 1.0: 
                blinks.append((blink_start, blink_end))
                last_blink_end = blink_end
            # else: 
            #     print(f'We have skipped this blink due to abnormal blink duration of {duration:.2f}s')
            blink_during = False
    
    #If the data frame ends while blinking
    if blink_during:
        blink_end = df.iloc[-1]['Timestamp']
        duration = blink_end - blink_start
        if blink_end > blink_start and 0.08 < duration < 1.0: 
            blinks.append((blink_start, blink_end))

    #Handle weird csv file 
    if len(blinks) > 75: 
        raise ValueError('Total blinks exceeded human capability. File being skipped')
    elif len(blinks) <= 2: 
        raise ValueError('Total blinks are too small that humans cannot achieve')
    
    blink_starts = [blink[0] for blink in blinks] #create a new list with the first item of blinks 
    blink_durations = [(end - start) for start, end in blinks] #create a new list of blink duration

    #inter-blink intervals = interval
    interval = [(blink_starts[i + 1] - blink_starts[i]) for i in range(len(blink_starts) - 1)]

    #count how many blinks there are in each stage
    blink_frameIDs = []

    for start, _ in blinks: 
        match_rows = df[df['Timestamp'].between(start - 0.01, start + 0.01)]
        if not match_rows.empty: 
            frame_str = match_rows['frameId'].iloc[0]
            frameID = int(frame_str.replace("frameId", ""))
            blink_frameIDs.append(frameID)
            
        else: 
            blink_frameIDs.append(-1)

    earlyStage = [1 for frameID in blink_frameIDs if frameID in [0, 1]]
    midStage = [1 for frameID in blink_frameIDs if frameID == 2]
    lateStage = [1 for frameID in blink_frameIDs if frameID in [3, 4]]

    #blink bursts
    burst_count = 0
    for i in range(len(blink_starts)): 
        curr = blink_starts[i]
        bursts = [start for start in blink_starts if curr <= start < curr + 3]
        if len(bursts) >= 3: 
            burst_count += 1
    
    #mismatch（左と右で差があるのか）
    mismatch = df[df['Leftblink'] != df['Rightblink']].shape[0]

    #blink near switch 
    switching = [start_time + 15, start_time + 30, start_time + 45, start_time + 60]
    blinkCounts_nearSwitch = sum(any(abs(blink - switch) <= 1 for switch in switching) for blink in blink_starts)
    blinkRatio_nearSwitch = blinkCounts_nearSwitch / len(blink_starts) if blink_starts else 0

    #Naturalness Score (higher = more natural blinking , lower = random blinking)
    burstRatio = burst_count / len(blink_starts) if blink_starts else 0
    naturalness_Score = blinkRatio_nearSwitch * (1 - burstRatio)

    #grid counts per blink 
    grid_Counts = []
    for start, end in blinks:  #for each pair 
        grid = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]['cellID'].dropna().unique()
        grid_Counts.append(grid.shape[0])

    # #scoring
    # count = 0
    # if earlyStage >= midStage and earlyStage >= lateStage: 
    #     count += 1; 
    # if 
    
    #output everything 
    return {
        "file name": os.path.basename(filepath), 
        "total_blinks": len(blinks), 
        "blink duration (mean)": np.mean(blink_durations) if blink_durations else 0, 
        "blink duration (max)": np.max(blink_durations) if blink_durations else 0, 
        "earlyStage_blinks": sum(earlyStage), 
        "midStage_blinks": sum(midStage), 
        "lateStage_blinks": sum(lateStage), 
        "inter-blink intervals (mean)": np.mean(interval) if interval else 0, 
        "inter-blink intervals (std)": np.std(interval) if interval else 0, 
        "burst counts": burst_count, 
        "mismatch Counts": mismatch, 
        "blink ratio near switch": blinkRatio_nearSwitch,
        "naturalness score": naturalness_Score, 
        "grid counts per blink (mean)": np.mean(grid_Counts) if grid_Counts else 0, 
        "Stress Score": df['score'].iloc[-1]
    }

def batch_analysis(input_folder, output_csvFile): 
    allfiles = glob.glob(os.path.join(input_folder, "VR*/**/*.csv"), recursive=True)
    results = []

    for file in allfiles: 
        try: 
            output = analyze_blink_file(file)
            results.append(output)
        except Exception as e: 
            print(f"There is an error with the file {file}: {e}")
        
    summaryData = pd.DataFrame(results)
    summaryData.to_csv(output_csvFile, index=False)
    print(f"Analysis has been completed. Here is the output csv file: {output_csvFile}")

#The program can only run when you run the script directly 
if __name__ == "__main__": 
    input_folder = "Health-O"
    output_csvFile = "summary.csv"
    batch_analysis(input_folder, output_csvFile)
