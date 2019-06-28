class lasif_comm(object):
    """
    Communication with Lasif
    """

    def __init__(self, info_dict: dict):
        self.lasif_root = info_dict["lasif_project"]
        self.comm = self.find_project_comm()
    
    def find_project_comm(self):
        """
        Get lasif communicator.
        """
        import pathlib
        from lasif.components.project import Project
        
        folder = pathlib.Path(self.lasif_root).absolute()
        max_folder_depth = 4
        
        for _ in range(max_folder_depth):
            if (folder / "lasif_config.toml").exists():
                return Project(folder).get_communicator()
            folder = folder.parent
        raise ValueError(f"Path {self.lasif_root} is not a LASIF project")
    

