class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'icdar':
            return '/path/to/ICDAR2015/icdar_deepLab/'
        elif dataset == 'synthText':
            return '/path/to/SynthText/synthText_deepLabV3Plus/'  # folder that contains dataset/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
    def exp_out_save(saveRoot,dataset):
        if saveRoot == 'tmp-network':
            if dataset == 'icdar':
                return '/path/to/deeplabV3Plus/icdar_models_softPHOC/run_tripleLoss'
            elif dataset == 'synthText':
                return '/path/to/deeplabV3Plus/synthText_models_softPHOC/run_tripleLoss'
 
