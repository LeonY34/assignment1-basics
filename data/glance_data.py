import os



if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.dirname(__file__))
    # print(data_dir)    
    print(f"glancing files in {data_dir}. Available files:")
    # print(files)
    index = 0
    ff = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file[0] == '.': continue
            ff.append(file)
            index += 1
            print(f"{index}. {file}")
    while True:
        s = input(f"Select from 1 to {index}: ")
        try:
            d = int(s)
            break
        except:
            print("Invalid input!")
    
    lines = 50
    
    with open(os.path.join(data_dir, ff[d - 1]), 'r') as f:
        for i in range(lines):
            print(f.readline())