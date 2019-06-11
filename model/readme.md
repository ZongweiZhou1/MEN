Instructions of some defined layers.

### correlation
corr = Correlation(pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1)

```
    C(i, j, k) = <A(i, j),  B(i+p, j+q)>
```
where `p, q \in [-d+stride2*0, -d+stride2*1, -d+stride2*2, ..., d]`, `d` is max_displacement, 
`k = (int(2d/stride2)+1) * p + q`, so the channels of output `C` is `(int(2d/stride2)+1)^2`.

When `kernel_size!=1`, 3 for example, 

```
    C(i, j, k) = \sum_{u=-1, v=1}^{1,1} <A(i-u, j-v),  B(i+p-u, j+q-v)>
```


 - pad_size:            The `pad_size` for the tensor `B`, usually `pad_size` equal to `max_displacement`. 
 - kernel_size:         The 'kernel_size' used in each correlation operation.
 - max_displacement:    The `max_displacement` for each pixel in tensor `A`.
 - stride1:             The `stride` of `A` used in correlation.
 - stride2:             The `stride` of `B` used in correlation.
 - corr_multiply:       The 'multiply' used on the correlation result.
 
 `NOTE:`  The maxplacement is related to the coordinate in `A`, not the center of `B`. For example, if `A\in R^{3, 3}, B\in R^{5,5}`,
 then `corr(A, B)`, operated form top-left, not the center.
 `NOTE:` The input `A, B` are all `THCUDAtensor` of `Variable`
 
 Now, some examples are given:
 ```python
x = Variable(torch.from_numpy(np.arange(9).reshape(1, 1, 3, 3).astype(np.float32))).cuda()
y = Variable(torch.from_numpy(np.arange(25).reshape(1, 1, 5, 5).astype(np.float32))).cuda()
corr = Correlation(pad_size=2, kernel_size=1, max_displacement=2, stride1=1, stride2=1)
z = corr(x, y)
print(z[0,0])
```
The output is
```
Variable containing:
 0  0  0
 0  0  0
 0  0  0
[torch.cuda.FloatTensor of size 3x3 (GPU 0)]
```

```python
x = np.zeros((5, 5)).astype(np.float32)
x[1:4, 1:4] = np.arange(9).reshape(3, 3).astype(np.float32)
x = Variable(torch.from_numpy(x.reshape(1, 1, 5, 5))).cuda()
y = Variable(torch.from_numpy(np.arange(25).reshape(1, 1, 5, 5).astype(np.float32))).cuda()
corr = Correlation(pad_size=2, kernel_size=1, max_displacement=2, stride1=1, stride2=1)
z = corr(x, y)
print('x:')
print(x[0,0])
print('y:')
print(y[0,0])
print('z[0,0]:')
print(z[0,0])
```
the output is 
```sh
x:
Variable containing:
 0  0  0  0  0
 0  0  1  2  0
 0  3  4  5  0
 0  6  7  8  0
 0  0  0  0  0
[torch.cuda.FloatTensor of size 5x5 (GPU 0)]

y:
Variable containing:
  0   1   2   3   4
  5   6   7   8   9
 10  11  12  13  14
 15  16  17  18  19
 20  21  22  23  24
[torch.cuda.FloatTensor of size 5x5 (GPU 0)]

z[0,0]:
Variable containing:
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   5   0
  0   0  35  48   0
  0   0   0   0   0
[torch.cuda.FloatTensor of size 5x5 (GPU 0)]
``` 
the `(i=2, j=2)` in `x`  aligned to `(i=0, j=0)` in 'y'.