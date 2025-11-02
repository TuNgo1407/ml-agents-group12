import os

"""
Find the absolute path of a folder relative to the git repository root.
    
This function traverses up the directory tree from the current script's location
    until it finds repository root markers (.git folder or requirements.txt file).
    Once the repository root is found, it constructs and validates the path to the
    DIRECT CHILD of the root.
"""


def get_project_root():
    """Find and return the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find repository root by looking for .git folder and requirements.txt
    root_markers = ['.git', 'requirements.txt']
    
    search_dir = current_dir
    
    try:
        while True:
            for marker in root_markers:
                marker_path = os.path.join(search_dir, marker)
                if os.path.exists(marker_path):
                    return search_dir  # Return the root directory 
            
            parent_dir = os.path.dirname(search_dir)
            if parent_dir == search_dir:  # Reached filesystem root
                raise FileNotFoundError("Could not find repository root. Make sure you're running from within the git repository.")
            
            search_dir = parent_dir
            
    except Exception as e:
        print(f"Error when searching for project root: {e}")
        raise RuntimeError(f"Unexpected error while searching for project root: {str(e)}")
    

def get_absolute_path(folder: str):
    """Get absolute path of a folder within project root (your existing function)"""
    project_root = get_project_root()
    folder_path = os.path.join(project_root, folder)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder} folder not found at: {folder_path}")
    
    return folder_path
    