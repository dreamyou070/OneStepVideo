import os
import glob
def main() :

    # [1] make final csv file
    final_csv_file = r'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/final_score'
    os.makedirs(final_csv_file, exist_ok=True)
    final_csv_file = os.path.join(final_csv_file, 'final_score.csv')

    # [2] read previous csv files
    csv_base_folder = r'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4'
    csv_files = glob.glob(os.path.join(csv_base_folder, '*.csv'))
    csv_files.sort()
    total_csv_elem = []
    for i, csv_file in enumerate(csv_files) :
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        if i == 0 :
            header = lines[0].strip()
            header = header.split(',')

        contents = lines[-1].strip()
        contents = contents.split(',')
        #print(f'type of contents: {contents}')
        total_csv_elem.append(contents)
    # [e1,e2]
    import csv
    with open(final_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 헤더 작성
        writer.writerows(total_csv_elem)  # 데이터 작성




if __name__ == "__main__" :
    main()