import os
import torch

"""
Read label data.
format: label_id label_name
"""
def read_label_data(label_path):	
	id2label, label2id = {}, {}
	with open(label_path, 'r') as f:
		for i, line in enumerate(f):
			line = line.strip().split(' ')
			id2label[int(line[0])] = line[1]
			label2id[line[1]] = int(line[0])

	return id2label, label2id


def load_pretrained_model(model, pretrained_path):
    """
    Load pretrained weight.
    the pretrained_path only includes state_dict()
    """
    if os.path.isfile(pretrained_path):
        print ('Loading weight file from {}'.format(pretrained_path))
        weight_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])  # remove module
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print ('Pretrained weight loaded completed.')
    else:
        print ('Can not find weight file!')

def load_checkpoint_model(model, checkpoint_path):
    """
    This is for resume training.
    the checkpoint_path includes:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        },
    only need 'state_dict'
    NOTE: train and resume train must have same number of GPU, since the name 'module'
    """
    if os.path.isfile(checkpoint_path):
        print ('Loading checkpoint from {}'.format(checkpoint_path))
        pretrained_checkpoint = torch.load(checkpoint_path)
        pretrained_dict = pretrained_checkpoint['state_dict']
        model_dict = model.state_dict()

        # only use 1 gpu         
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if 'module' in k:
                name = k[7:]    # remove 'module.'
            new_pretrained_dict[name] = v
        pretrained_dict = new_pretrained_dict
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) == 0:
            print('   checkpoint model and current model do not match!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Checkpoint loaded completed.")
    else:
        print ('Can not fiind the checkpoint!')