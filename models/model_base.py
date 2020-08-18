import torch
import torch.nn as nn
import datetime
import os
import glob2


class Model_base(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod  # Fixed
    def save(cls, model, path, optimizer, epoch,
             tr_loss=None, cv_loss=None, C=None):
        package = cls.serialize(model, optimizer, epoch,
                                tr_loss=tr_loss, cv_loss=cv_loss, C=C)
        torch.save(package, path)

    @classmethod  # Fixed
    def encode_model_identifier(cls,
                                metric_name,
                                metric_value):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")

        file_identifiers = [metric_name, str(metric_value)]
        model_identifier = "_".join(file_identifiers + [ts])

        return model_identifier

    @classmethod  # Fixed
    def decode_model_identifier(cls,
                                model_identifier):
        identifiers = model_identifier.split("_")
        ts = identifiers[-1].split('.pt')[0]
        [metric_name, metric_value] = identifiers[:-1]
        return metric_name, float(metric_value), ts

    @classmethod  # Fixed
    def get_best_checkpoint_path(cls, model_dir_path):
        best_paths = glob2.glob(model_dir_path + '/best_*')
        if best_paths:
            return best_paths[0]
        else:
            return None

    @classmethod  # Fixed
    def get_current_checkpoint_path(cls, model_dir_path):
        current_paths = glob2.glob(model_dir_path + '/current_*')
        if current_paths:
            return current_paths[0]
        else:
            return None

    @classmethod  # Fixed
    def save_if_best(cls, save_dir, model, optimizer, epoch, tr_loss, cv_loss, cv_loss_name, save_every=None, C=None):
        '''
        best model is determined by comparing the cv_loss
        :param save_dir:
        :param model:
        :param optimizer:
        :param epoch:
        :param tr_loss:
        :param cv_loss:
        :param cv_loss_name:
        :param C: a dict that saves the speaker embedding centroids
        '''
        # model saving path
        # model_dir_path = os.path.join(save_dir, cls.encode_dir_name(model))
        model_dir_path = save_dir

        if not os.path.exists(model_dir_path):
            print("Creating non-existing model states directory... {}"
                  "".format(model_dir_path))
            os.makedirs(model_dir_path)

        current_path = cls.get_current_checkpoint_path(model_dir_path)
        models_to_remove = []
        if current_path is not None:
            models_to_remove = [current_path]
        best_path = cls.get_best_checkpoint_path(model_dir_path)
        file_id = cls.encode_model_identifier(cv_loss_name, cv_loss)

        if best_path is not None:
            best_fileid = os.path.basename(best_path)
            _, best_metric_value, _ = cls.decode_model_identifier(
                best_fileid.split('best_')[-1])
        else:
            best_metric_value = -99999999

        if float(cv_loss) > float(best_metric_value):
            if best_path is not None:
                models_to_remove.append(best_path)
            save_path = os.path.join(model_dir_path, 'best_' + file_id + '.pt')
            cls.save(model, save_path, optimizer, epoch,
                     tr_loss=tr_loss, cv_loss=cv_loss, C=C)

        save_path = os.path.join(model_dir_path, 'current_' + file_id + '.pt')
        cls.save(model, save_path, optimizer, epoch,
                 tr_loss=tr_loss, cv_loss=cv_loss, C=C)

        if save_every:
            if epoch % save_every == 0:
                save_path = os.path.join(model_dir_path, 'temp{}_'.format(epoch) + file_id + '.pt')
                cls.save(model, save_path, optimizer, epoch,
                         tr_loss=tr_loss, cv_loss=cv_loss, C=C)

        try:
            for model_path in models_to_remove:
                os.remove(model_path)
        except:
            print("Warning: Error in removing {} ...".format(current_path))

    @staticmethod  # Fixed
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None, C=None):
        package = {'model_args': model.model_args}
        package['state_dict'] = model.state_dict()
        package['optim_dict'] = optimizer.state_dict()
        package['epoch'] = epoch
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        if C is not None:
            package['C'] = C
        return package

    @classmethod  # Customize the model_dir_name, which is used to save the model
    def encode_dir_name(cls, model):
        model_dir_name = 'spectra_reg_{}_L_{}_N_{}_WN_{}'.format(
            model.cae_reg, model.L, model.N, model.weighted_norm)
        return model_dir_name

    @classmethod  # Customize the dir_id
    def load_model(cls, models_dir, model_state='best', arg_changes=None, model=None):
        dir_path = models_dir

        path = ''
        try:
            path = glob2.glob(dir_path + '/{}_*'.format(model_state))[0]
        except IndexError:
            print('No {} model in {}'.format(model_state, dir_path))

        print('\nLoad the pre-trained model: {}\n'.format(path))
        package = torch.load(path, map_location=lambda storage, loc: storage)

        if 'model_args' in package.keys():
            saved_args = package['model_args']
        else:
            saved_args = package
        if arg_changes is not None:
            for item in arg_changes.items():
                if item[0] in saved_args.keys():
                    if item[1] != saved_args[item[0]]:
                        print('Update args to: {}'.format(item))
                else:
                    print('Add new arg item: {}'.format(item))
            saved_args.update(arg_changes)

        if model is None:
            model = cls(model_args=saved_args)
        model.load_state_dict(package['state_dict'], strict=False)
        return model

    @classmethod  # Customize the dir_id
    def load_optimizer(cls, opt=None, models_dir=None, model_state='best'):
        dir_path = models_dir

        path = ''
        try:
            path = glob2.glob(dir_path + '/{}_*'.format(model_state))[0]
        except IndexError:
            print('No {} model in {}'.format(model_state, dir_path))

        package = torch.load(path, map_location=lambda storage, loc: storage)
        if opt is not None:
            opt.load_state_dict(package['optim_dict'])
        epoch = package['epoch']
        if 'C' in package.keys():
            C = package['C']
        else:
            C = None
        return opt, epoch, C