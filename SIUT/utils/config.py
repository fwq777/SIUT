from pprint import pprint

class Config:
    # data
    rootpath = './data/'
    testpath = './data/test/seismic2.npy'
    num_workers = 1
    image_size = 128

    # param for optimizer
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3    # 1e-3
    lr_step = 45*56
    # mode1, use L1+Crossentropy(L1+Texure); mode2, use MSEloss(Recon+Texure)
    LAMBDA_DICT = {
        'L1': 10.0, 'Recon': 150.0, 'Texure': 1.0}
    mode = 'mode1'

    model_name = 'SIUT'
    # training
    epoch = 150
    train_batch_size = 14
    test_batch_size = 24
    # visualization
    env = model_name  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter

    test_num = 10000
    # model
    test_only = False
    load_F = './checkpoints/netInp_04051616_28.31363.pth'
    savepath = "./result/"
    load_G = './checkpoints/netG_02081503_0.17432.pth'


    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
