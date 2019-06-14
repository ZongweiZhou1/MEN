import cv2
import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.correlation_package.correlation import Correlation

sns.set()


x = cv2.imread('/data/zwzhou/Data/MOT16/train/MOT16-02/img1/000001.jpg')[425:556, 1377:1508, ::-1]/255.
y = cv2.imread('/data/zwzhou/Data/MOT16/train/MOT16-02/img1/000002.jpg')[405:536, 1357:1488, ::-1]/255.
plt.imsave('images/x.png', x)
plt.imsave('images/y.png', y)
# plt.imshow(y, cmap=plt.cm.gray)
plt.imsave('images/y_crop.png', y[25:106, 25:106])
x = torch.from_numpy(x[:, :, 0].astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
y0 = torch.from_numpy(y[:, :, 0].astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
y1 = torch.from_numpy(y[25:106, 25:106, 0].astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

# conv2d
out_conv2 = F.conv2d(x, y1, padding=40)
print(out_conv2.size())
zy = out_conv2.squeeze().cpu().numpy()
# print(zy)
ax = plt.figure(figsize=(12, 12))
sns.heatmap(zy/zy.max(), annot=False, cmap='jet', vmin=0.0, vmax=1.0)
plt.savefig("images/2d_conv")


# conv_corr
corr = Correlation(pad_size=65, kernel_size=1, max_displacement=65, stride2=1, stride1=1)
z = corr(y0, x)
out_corr = F.avg_pool2d(z[:,:,25:106, 25:106], (81, 81))
print(out_corr.size())
out_d = 2 * 65 + 1
z = out_corr.squeeze().cpu().numpy().reshape(out_d,out_d)
ax = plt.figure(figsize=(12, 12))
sns.heatmap(z/z.max(), annot=False, cmap='jet', vmin=0.0, vmax=1.0)
plt.savefig("images/corr_conv")


# import torch
# import numpy as np
# from model.correlation_package.correlation import Correlation
# x = np.zeros((5, 5)).astype(np.float32)
# x[1:4, 1:4] = np.arange(9).reshape(3, 3).astype(np.float32)
# x = torch.from_numpy(x.reshape(1, 1, 5, 5)).cuda()
# y = torch.from_numpy(np.arange(25).reshape(1, 1, 5, 5).astype(np.float32)).cuda()
# corr = Correlation(pad_size=2, kernel_size=1, max_displacement=2, stride1=1, stride2=1)
# z = corr(x, y)
# print('x:')
# print(x[0,0])
# print('y:')
# print(y[0,0])
# print('z[0,0]:')
# print(z[0,0])


