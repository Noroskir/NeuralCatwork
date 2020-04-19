import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""Calculate cross correlation efficiently"""


# working
# def cross_correlate(a, f):
#     s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
#     # print("shape")
#     # print(s)
#     strd = np.lib.stride_tricks.as_strided
#     subM = strd(a, shape=s, strides=a.strides * 2)
#     # print("Sub matrix")
#     # print(subM)
#     return np.einsum('ijm,ijmkln->kl', f, subM)


# A = np.array([[[5, 5], [0, 0], [0, 0]], [
#              [0, 0], [5, 4], [0, 0]], [[0, 0], [0, 0], [5, 3]]])
# f = np.array([[[1, 1], [2, 0]], [[0, 1], [0, 0]]])

# print(A.shape)
# print(f.shape)
# res = cross_correlate(A, f)
# print(res.shape)
# print(res)


# def cross_correlate(a, f):
#     s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
#     print("shape")
#     print(s)
#     strd = np.lib.stride_tricks.as_strided
#     subM = strd(a, shape=s, strides=a.strides * 2)
#     print("subM")
#     print(subM)
#     return np.einsum('ijmy,zijmklny->zkl', f, subM)


# A = np.array([[[[5, 5], [0, 0], [0, 0]], [
#              [0, 0], [5, 4], [0, 0]], [[0, 0], [0, 0], [5, 3]]]])
# f = np.array([[[[1], [1]], [[2], [0]]], [[[0], [1]], [[0], [0]]]])

# print(A.shape)
# print(f.shape)
# res = cross_correlate(A, f)
# print(res.shape)
# print(res)

# works for A
# def cross_correlate_c(A, F):
#     # cross correlate image of shape (c, y, x) with filter (n_c, c, f, f)
#     s = tuple(np.subtract(A.shape, F.shape[1:]) + 1) + F.shape[1:]
#     print('shape')
#     print(s)
#     strd = np.lib.stride_tricks.as_strided
#     subM = strd(A, shape=s, strides=A.strides*2)
#     print("sub matrix")
#     print(subM)
#     print(subM.shape)
#     return np.einsum('fijc,mklijc->fkl', F, subM)


# def cross_correlate_full(A, F):
#     """Cross correlate image volume A with filter F.
#     Args:
#          A (np.array): array with shape (m, c, y, x)
#          F (np.array): array with shape (n_c, c, f, f)
#     Returns:
#         np.array: correlated array with shape (m, n_c, c, y)
#     """
#     # cross correlate images of shape (m, c, y, x) with filter (n_c, c, f, f)
#     s = tuple(np.subtract(A.shape, (1,)+F.shape[1:]) + 1) + (1,)+F.shape[1:]
#     print('shape')
#     print(s)
#     strd = np.lib.stride_tricks.as_strided
#     subM = strd(A, shape=s, strides=A.strides*2)
#     print("sub matrix")
#     print(subM)
#     print(subM.shape)
#     return np.einsum('fijc,nmklzijc->nfkl', F, subM)


def cross_correlate(A, F, s=1):
    """Cross correlate image volume A with filter F.
    Args:
        A (np.array): array with shape (m, c, y, x)
        F (np.array): array with shape (n_c, c, f, f)
        s (int): stride
    Returns:
        np.array: correlated array with shape (m, n_c, c, y)
    """
    shape = tuple(np.subtract(
        A.shape, (1,)+F.shape[1:]) + 1) + (1,)+F.shape[1:]
    s1 = (shape[2]+s-1)//s
    s2 = (shape[3]+s-1)//s
    shape = (shape[0], shape[1], s1, s2, *shape[4:])
    # print('shape')
    # print(shape)
    strides = A.strides*2
    # print(strides)
    strides = (*strides[:-2], s*strides[-2], s*strides[-1])
    strd = np.lib.stride_tricks.as_strided
    subM = strd(A, shape=shape, strides=strides)
    # print("sub matrix")
    # print(subM)
    # print(subM.shape)
    return np.einsum('fijc,nmklzijc->nfkl', F, subM)


X = np.array([[[[5, 0, 0, 1], [0, 5, 0, 1], [0, 0, 5, 1], [0, 0, 0, 1]],
               [[5, 0, 0, 1], [0, 4, 0, 1], [0, 0, 3, 1], [0, 0, 0, 1]]]])

A = np.array([[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
              [[5, 0, 0], [0, 4, 0], [0, 0, 3]]])
B = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              [[2, 0, 0], [0, 2, 0], [0, 0, 2]]])
C = np.array([A, B])
f = np.array([[[[1, 0], [1, 0]],
               [[1, 2], [0, 0]]],
              [[[1, 2], [0, 0]],
               [[1, 0], [1, 0]]]])

# cross_correlate_stride(X, f, 1)

# print(A.shape)
# print(f.shape)
# print(A.strides)
# print(f.strides)

# print('### Res')
# res = cross_correlate(C, f, s=1)
# print(res)
# print(res.shape)


# def load_image(infilename):
#     img = Image.open(infilename).convert('L')
#     img.load()
#     data = np.asarray(img)
#     return data


# im = load_image("data/test.jpg")
# print(im.shape)
# plt.imshow(im)
# plt.show()

# im = np.array([np.array([im])])
# f = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]],
#               [[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]])

# res = cross_correlate(im, f)
# print(res.shape)
# plt.imshow(res[0, 0])
# plt.show()

# plt.imshow(res[0, 1])
# plt.show()

def conv2d(x, f):
    s = f.shape + tuple(np.subtract(x.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(x, shape=s, strides=x.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


# works
# A = np.array([[5, 0, 0], [0, 5, 0], [1, 0, 4]])
# f = np.array([[0, 0], [0, 1]])


# def conv_back2d(x, W, da):
#     dW = np.zeros(f.shape)
#     dx = np.zeros(x.shape)
#     s = W.shape + tuple(np.subtract(x.shape, W.shape) + 1)
#     strd = np.lib.stride_tricks.as_strided
#     sub_x = strd(x, shape=s, strides=x.strides * 2)
#     pad = 1
#     da_pad = np.pad(da, ((pad, pad), (pad, pad)),
#                     mode='constant', constant_values=(0, 0))
#     s = (W.shape[0], W.shape[1], x.shape[0], x.shape[1])
#     sub_da = strd(da_pad, shape=s, strides=da_pad.strides*2)
#     dW = np.einsum('ij,mnij->ij', da, sub_x)
#     db = np.einsum('ij->', da)
#     dx = np.einsum('ij,ijmn->mn', np.rot90(W, 2), sub_da)
#     print(dW)
#     print(db)
#     print(dx)


# res = conv2d(A, f)
# print(res)

# conv_back2d(A, f, res)

def cross_correlate_c(A, F):
    # cross correlate image of shape (c, y, x) with filter (n_c, c, f, f)
    s = tuple(np.subtract(A.shape, F.shape[1:]) + 1) + F.shape[1:]
    strd = np.lib.stride_tricks.as_strided
    subM = strd(A, shape=s, strides=A.strides*2)
    return np.einsum('fijc,mklijc->fkl', F, subM)


A = np.array([[[5, 0, 0], [0, 5, 0], [1, 0, 4]],
              [[4, 0, 0], [0, 5, 0], [1, 0, 4]]])
f = np.array([[[[1, 0], [0, 0]],
               [[0, 0], [0, 0]]]])


def conv_back2d(x, W, da):
    # backward for image shape (c, x, y) with filter (n_c, c, f, f)
    dW = np.zeros(f.shape)
    dx = np.zeros(x.shape)
    s = tuple(np.subtract(x.shape, W.shape[1:]) + 1) + W.shape[1:]
    strd = np.lib.stride_tricks.as_strided
    sub_x = strd(x, shape=s, strides=x.strides * 2)
    print(sub_x.shape)
    print(da.shape)
    dW = np.einsum('fij,flijmn->fij', da, sub_x)
    db = np.einsum('fij->', da)
    print(dW)
    print(db)

    pad = 1
    da_pad = np.pad(da, ((0, 0), (pad, pad), (pad, pad)),
                    mode='constant', constant_values=(0, 0))
    s = (W.shape[0], W.shape[1], x.shape[0], x.shape[1])
    sub_da = strd(da_pad, shape=s, strides=da_pad.strides*2)
    dx = np.einsum('ij,ijmn->mn', np.rot90(W, 2), sub_da)
    print(dx)


res = cross_correlate_c(A, f)
print(res)

conv_back2d(A, f, res)
