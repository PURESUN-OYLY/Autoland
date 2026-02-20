import os
import hashlib

def find_planes(pcd_input, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    planes = []

    # Find one plane, this function will be called recursively
    def find_plane(pcd_input):
        plane_model, inliers = pcd_input.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # remove plane
        plane_pcd = pcd_input.select_by_index(inliers)
        pcd_remaining = pcd_input.select_by_index(inliers, invert=True)

        return [plane_model, plane_pcd], pcd_remaining

    # Find all planes
    target_pcd = pcd_input
    while True:
        # there is no enough points for a plane
        if len(target_pcd.points) < 3:
            break
        plane, target_pcd = find_plane(target_pcd)

        # not found in current pcd
        if len(target_pcd.points) == 0:
            break
        planes.append(plane)

    return planes

def check_pcd_source_dir(dir):
    # check if path exists
    if not os.path.exists(dir):
        print("Path does not exist, please make sure your path is correct")
        return False
    
    # check if path is a directory
    if not os.path.isdir(dir):
        print("Path is not a directory, please make sure your path is correct")
        return False
    
    # count the number of pcd files and total size
    pcd_file_cnt = 0
    files = os.listdir(dir)
    total_size = 0
    pcd_files = []

    # count the number of pcd files
    for file in files:
        if file.endswith('.pcd'):
            pcd_file_cnt += 1
            pcd_files.append(file)
            total_size += os.path.getsize(os.getcwd() + '/points/' + file)

    # print the result if the path is valid
    print("Path is valid")
    print(f'Total {pcd_file_cnt} pcd files found, total size: {total_size / 1024 / 1024:.2f} MB')

    # return the number of pcd files if the path is valid
    return pcd_file_cnt, pcd_files

def get_target_pcd(dir, pcd_file_id):
    pcdf_cnt, pcd_files = check_pcd_source_dir(dir)
    if pcd_file_id >= pcdf_cnt:
        print("Invalid pcd file id, please make sure your pcd file id is correct")
        return False
    
    # print the target pcd file
    target_pcd_path = os.path.join(dir, pcd_files[pcd_file_id])
    print(f"Target pcd file: {pcd_files[pcd_file_id]}, size: {os.path.getsize(target_pcd_path) / 1024:.2f} KB")
    
    # return the target pcd file
    return target_pcd_path



# calculate the md5 of a file
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

if __name__ == "__main__":
    print("You can't run this script directly")