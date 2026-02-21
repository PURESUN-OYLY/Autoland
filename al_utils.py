import os
import hashlib
import numpy as np
import open3d as o3d
import tkinter as tk

def find_planes(pcd_input, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    """
    Find planes in point cloud\n
    :param pcd_input: the point cloud to be processed\n
    :param distance_threshold: the distance threshold for RANSAC\n
    :param ransac_n: the number of points to be sampled for RANSAC\n
    :param num_iterations: the number of iterations for RANSAC\n
    :return: a list of planes\n
    """
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

# utils

# Add color to point cloud
def add_color(pcd, color=[1, 0, 0]):
    """
    Add color to point cloud\n
    :param pcd: the point cloud to be added color\n
    :param color: the color to be added, default is red\n
    :return: the point cloud with added color\n
    """
    num_points = len(pcd.points)
    color_array = np.ones((num_points, 3)) * color
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    return pcd

# Show point cloud
def show_point_cloud(pcd, name = "Point Cloud"):
    """
    Show point cloud\n
    :param pcd: the point cloud to be shown\n
    :param name: the name of the window\n
    :return: None\n
    """
    o3d.visualization.draw_geometries([pcd], window_name=name)

# Change the color of point cloud
def change_color(pcd, ids=[], color=[0.5, 0.5, 0.5]):
    """
    Change the color of point cloud\n
    :param pcd: the point cloud to be changed\n
    :param ids: the ids of the points to be changed\n
    :param color: the color to be changed\n
    :return: the point cloud with changed color\n
    """
    colors = np.array(pcd.colors)
    for i in ids:
        colors[i] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# merge point clouds
def merge_pcds(pcd_list, merge_color = True, color_default = [0.5, 0.5, 0.5]):
    """
    Open3D PointCloud merge function\n
    :param pcd_list: the point clouds list to be merged\n
    :param merge_color: whether to merge color, if set to False, will drop all colors\n
    :param default_color: the color to be used if no color is found in the point cloud\n
    :Note: will be automatically merge normals, if all point clouds have normals\n
    :return: the merged point cloud
    """

    # check how many point clouds to be merged
    if len(pcd_list) < 2:
        print("Need at least 2 point clouds to be merged.")
        return

    merged_pcd = o3d.geometry.PointCloud()
    points = np.zeros((0, 3))
    colors = np.zeros((0, 3))
    normals = np.zeros((0, 3))

    # check if all point clouds have normals
    has_normals = True
    for i in range(len(pcd_list)):
        if not pcd_list[i].has_normals():
            has_normals = False
            break
    

    for i in range(len(pcd_list)):
        # get point cloud
        pcd = pcd_list[i]
        # merge point cloud
        points = np.vstack([points, np.asarray(pcd.points)])

        # merge color
        if merge_color:
            if pcd.has_colors():
                # add color if exist
                colors = np.vstack([colors, np.asarray(pcd.colors)])
            else:
                # add default color
                colors = np.vstack([colors, np.ones((len(pcd.points), 3)) * color_default])
        
        # merge normals
        if has_normals:
            normals = np.vstack([normals, np.asarray(pcd.normals)])
    
    # assign merged point cloud
    merged_pcd.points = o3d.utility.Vector3dVector(points)
    if merge_color:
        merged_pcd.colors = o3d.utility.Vector3dVector(colors)
    if has_normals:
        merged_pcd.normals = o3d.utility.Vector3dVector(normals)

    # print(f"Merged all {len(pcd_list)} point clouds.")

    return merged_pcd

# Add origin point
def add_origin(pcd, color=[1, 0, 0]):
    """
    Add origin point to point cloud\n
    :param pcd: the point cloud to be added origin\n
    :param color: the color to be added, default is red\n
    :return: the point cloud with added origin\n
    """    
    origin_point = np.array([[0.0, 0.0, 0.0]])
    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(origin_point)
    origin_pcd = add_color(origin_pcd, color=color)

    return merge_pcds([pcd, origin_pcd])


def check_pcd_source_dir(dir):
    """
    Check if the path is valid\n
    :param dir: the path to be checked\n
    :return: pcd_file_cnt, pcd_files\n
    """
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
    """
    Get the target pcd file\n
    :param dir: the path to the pcd files\n
    :param pcd_file_id: the id of the pcd file to be used\n
    :return: the path to the target pcd file\n
    """
    pcdf_cnt, pcd_files = check_pcd_source_dir(dir)
    if pcd_file_id >= pcdf_cnt:
        print("Invalid pcd file id, please make sure your pcd file id is correct")
        return False
    
    # print the target pcd file
    target_pcd_path = os.path.join(dir, pcd_files[pcd_file_id])
    print(f"Target pcd file: {pcd_files[pcd_file_id]}, size: {os.path.getsize(target_pcd_path) / 1024:.2f} KB")
    
    # return the target pcd file
    return target_pcd_path

# copy pcd file to used_pcd folder if it doesn't exist or is different file
def copy_used_pcd(used_pcd_path, pcd_file_path):
    print('Copy pcd file to used_pcd folder...')
    if not os.path.exists(os.path.join(os.getcwd(), used_pcd_path)):
        print('Create used_pcd folder')
        os.makedirs(os.path.join(os.getcwd(), used_pcd_path))

    # Check if the file already exists in the used_pcd folder
    if os.path.exists(os.path.join(os.getcwd(), used_pcd_path, os.path.basename(pcd_file_path))):
        print(f'File {os.path.basename(pcd_file_path)} already exists in used_pcd folder, Checking if it is the same file...')
        
        # check md5sum
        md5_src = calculate_md5(pcd_file_path)
        md5_dst = calculate_md5(os.path.join(os.getcwd(), used_pcd_path, os.path.basename(pcd_file_path)))
        if md5_src == md5_dst:  
            print(f'File {os.path.basename(pcd_file_path)} is the same file, no need to copy')
        else:
            print(f'File {os.path.basename(pcd_file_path)} is different file, copying...')
            os.remove(os.path.join(os.getcwd(), used_pcd_path, os.path.basename(pcd_file_path)))
            os.rename(pcd_file_path, os.path.join(os.getcwd(), used_pcd_path, os.path.basename(pcd_file_path)))
    
    else:
        print(f'File {os.path.basename(pcd_file_path)} does not exist in used_pcd folder, copying...')
        # open source file in binary mode
        with open(pcd_file_path, 'rb') as source:
            # open destination file in binary mode
            with open(os.path.join(os.getcwd(), used_pcd_path, os.path.basename(pcd_file_path)), 'wb') as destination:
                # copy the contents of the source file to the destination file
                destination.write(source.read())


# calculate the md5 of a file
def calculate_md5(file_path):
    """
    Calculate the md5 of a file\n
    :param file_path: the path to the file\n
    :return: the md5 of the file\n
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_screen_center():
    """
    Get the center of the screen\n
    :return: the center of the screen\n
    """
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w // 2, screen_h // 2

if __name__ == "__main__":
    print("You can't run this script directly")
