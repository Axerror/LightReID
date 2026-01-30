from importlib import import_module #动态模块加载工具

import torch
import torch.nn as nn
import os.path as osp
from collections import OrderedDict

# 这是核心工厂函数，main.py 调用它来获取模型对象
def make_model(args, ckpt):

    ckpt.write_log('[INFO] Building {} model...'.format(args.model))

    # 确定计算设备：如果 args.cpu 为 True 则用 CPU，否则用 GPU (cuda)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # nGPU = args.nGPU

    # 导入模块
    module = import_module('model.' + args.model.lower())
    #实例化它，传入 args 参数，最后移至对应设备
    model = getattr(module, args.model)(args).to(device)

    # 如果有多个 GPU 且没开启 CPU 模式，使用 nn.DataParallel 进行多卡并行加速
    if not args.cpu and args.nGPU > 1:
        model = nn.DataParallel(model, range(args.nGPU))

    return model

# class Model(nn.Module):

#     def __init__(self, args, ckpt):
#         super(Model, self).__init__()
#         ckpt.write_log('[INFO] Making {} model...'.format(args.model))

#         if args.drop_block:
#             ckpt.write_log('[INFO] Using batch drop block with h_ratio {} and w_ratio {}.'.format(args.h_ratio, args.w_ratio))

#         self.device = torch.device('cpu' if args.cpu else 'cuda')
#         self.nGPU = args.nGPU

#         module = import_module('model.' + args.model.lower())
#         # self.model = module.make_model(args).to(self.device)
#         self.model = getattr(module, args.model)(args).to(self.device)

#         if not args.cpu and args.nGPU > 1:
#             self.model = nn.DataParallel(self.model, range(args.nGPU))

#     def forward(self, x):
#         return self.model(x)

#     def get_model(self):
#         if self.nGPU == 1:
#             return self.model
#         else:
#             return self.model.module

#     def save(self, apath, epoch, is_best=False):
#         target = self.get_model()
#         torch.save(
#             target.state_dict(),
#             os.path.join(apath, 'model', 'model_latest.pt')
#         )
#         if is_best:
#             torch.save(
#                 target.state_dict(),
#                 os.path.join(apath, 'model', 'model_best.pt')
#             )


#         if self.save_models:
#             torch.save(
#                 target.state_dict(),
#                 os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
#             )
'''
    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # if resume == -1:
        #     print('Loading model from last checkpoint')
        #     self.get_model().load_state_dict(
        #         torch.load(
        #             os.path.join(apath, 'model', 'model_latest.pt'),
        #             **kwargs
        #         ),
        #         strict=False
        #     )
        # elif resume == 0:
        #     if pre_train != '':
        #         print('Loading model from {}'.format(pre_train))
        #         self.get_model().load_state_dict(
        #             torch.load(pre_train, **kwargs),
        #             strict=False
        #         )
        # modified on 01.02.1010
        # if resume == 0:
        #     if pre_train != '':
        #         print('Loading model from {}'.format(pre_train))
        #         self.get_model().load_state_dict(
        #             torch.load(pre_train, **kwargs),
        #             strict=False
        #         )
        #     else:
        #         print('Loading model from last checkpoint')
        #         self.get_model().load_state_dict(
        #             torch.load(
        #                 os.path.join(apath, 'model', 'model_latest.pt'),
        #                 **kwargs
        #             ),
        #             strict=False
        #         )
        # else:
        #     self.get_model().load_state_dict(
        #         torch.load(
        #             os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
        #             **kwargs
        #         ),
        #         strict=False
        #     )
        # modified on 01.02.1010
        if pre_train != '':
            print('Loading model from {}'.format(pre_train))
            if pre_train.split('.')[-1][:3] == 'tar':
                print('load checkpointerrrrrrr')
                # checkpoint = self.load_checkpoint(pre_train)
                # self.get_model().load_state_dict(checkpoint['state_dict'])
                self.load_pretrained_weights(self.get_model(), pre_train)
            else:

                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            print('Loading model from last checkpoint')
            # print(apath)
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                # strict=False
            )
'''
