class Path(object):
    @staticmethod
    def db_dir(database):
        if database == "uex_class2":
            root_dir = 'uex_video/uex_class2'
            return root_dir
        elif database == "uex_class3":
            root_dir = 'uex_video/uex_class3'
            return root_dir
        elif database == "uex_class4":
            root_dir = 'uex_video/uex_class4'
            return root_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'