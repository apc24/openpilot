# calcDistanceToLink.md
ルート上の各リンクと現在地との距離を計算し、一番近いリンクを求める。

# 線分と点の距離の計算

$\bm{p_1}$と$\bm{p_2}$を結ぶ線分と$\bm{p_0}$との距離を考える。
ここで
$$
\bm{p_i}:=(x_i,y_i,z_i...)
$$
と定義する。

$\bm{p_1}$が原点になるよう座標変換をして以下のように定義する
$$
\bm{p_r}:=\bm{p_2}-\bm{p_1} \\
\bm{p}:=\bm{p_0}-\bm{p_1}
$$
$\bm{p_r}$を$x$軸方向の単位ベクトルに変換する行列$M_r$を考える。
まず、$x$軸方向に回転する回転行列を$\bm{\Theta_r}$と定義する。
$$
\bm{\Theta_r}:=\begin{pmatrix}
\bm{p_r}^T/|\bm{p_r}| \\ \bm{\Theta_m}
\end{pmatrix} \\
|\bm{p_r}|:=\sqrt{x_r^2+y_r^2+z_r^2 +...}
$$
さらに単位ベクトル化するために$|\bm{p_r}|$で正規化すると、変換行列$M_r$は以下のように定義できる。
$$
\bm{M_r}:=\begin{pmatrix}
\bm{p_r}^T/(\bm{p_r}^T\bm{p_r}) \\ \bm{M_m} 
\end{pmatrix} \\
\bm{M_m}:=\bm{\Theta_m}/|\bm{p_r}|
$$
2次元の場合は以下の通り
$$
\bm{M_r}:=\frac{1}{x_r^2+y_r^2}
\begin{pmatrix}
x_r, y_r \\ -y_r, x_r
\end{pmatrix} \\
$$
$\bm{p}$を変換すると
$$
\bm{M_r}\bm{p}=\begin{pmatrix}
\bm{p_r}^T\bm{p}/(\bm{p_r^Tp_r}) \\
\bm{M_m}\bm{p}
\end{pmatrix}
$$
2次元の場合は
$$
\bm{M_r}\bm{p}:=\frac{1}{x_r^2+y_r^2}\begin{pmatrix}
x_rx+y_ry \\ -y_rx+ x_ry
\end{pmatrix} \\
$$
ここで、$x$軸の値kが0～1の場合、線分と点の距離$l$はx軸との距離になる。負の場合は$\bm{p_1}$との距離、1以上の場合は$\bm{p_2}$との距離となる
$$
l=\begin{cases}
|\bm{p_0-p_1}| & k\leq 0 \\
|\bm{\Theta_mp}| & 0 \lt k \lt 1 \\
|\bm{p_0-p_2}| & 1 \leq k 
\end{cases} \\ 
k:=\frac{1}{\bm{p_r^Tp_r}}\bm{p_r}^T\bm{p}
$$
2次元の場合は
$$
l=\begin{cases}
\sqrt{x^2+y^2} & k \leq 0 \\
\frac{|-y_rx+x_ry|}{\sqrt{x_r^2+y_r^2}} & 0 \lt k \lt 1 \\
\sqrt{(x_0-x_2)^2+(y_0-y_2)^2} & 1 \leq k 
\end{cases} \\
k:=\frac{x_rx+y_ry}{x_r^2+y_r^2}
$$
また、最近傍点$p_n$の座標はx軸上の点を逆回転すれば求められるので
$$
\bm{p_n}=\bm{\Theta_r^T}\begin{pmatrix}
x_p \\
\bm{0}
\end{pmatrix}\\
x_p:=\bm{p_r}^T\bm{p}/|p_r|
$$
展開すると
$$
\bm{p_n}=\begin{pmatrix}
\bm{p_r}^/|\bm{p_r}| & \bm{\Theta_m}^T
\end{pmatrix} 
\begin{pmatrix}
\bm{p_r}^T\bm{p}/|p_r| \\
\bm{0}
\end{pmatrix}
=k\bm{p_r}\\
$$
まとめると
$$
\bm{p_n}=
\begin{cases}
0 & k \leq 0 \\
k\bm{p_r} & 0 \lt k \lt 1 \\
\bm{p_r} & 1 \leq k 

\end{cases} 
$$
$k$の値を$0 \leq k \leq1$に制限すれば
$$
\bm{p_n}=k\bm{p_r} 
$$
とまとめられる。
よって、lは以下のように表現できる。
$$
l = |\bm{p_n} - \bm{p}|
$$


# リンク列に対する計算
$$
P_i := \begin{pmatrix}
\bm{p_1},\bm{p_2},\bm{p_3},...
\end{pmatrix} \\
P_r := \begin{pmatrix}
\bm{p_2}-\bm{p_1},\bm{p_3}-\bm{p_2},\bm{p_4}-\bm{p_3},...
\end{pmatrix} \\
P:=\begin{pmatrix}
\bm{p_0}-\bm{p_1},\bm{p_0}-\bm{p_2},\bm{p_0}-\bm{p_3},...
\end{pmatrix}\\
\bm{k}=diag(P_r^T P)/diag(P_r^TP_r)\\
$$
$k$の値を0～1に制限した値を$k_{lim}$とすると
$$P_n=P_r diag(\bm{k_{lim})}\\
\bm{l}^2 =diag((P_n - P)^T(P_n - P))
$$
もっとも$l^2$が小さいリングが最近傍リンクとなる
pythonコードだと
```python
import numpy as np
Pi = np.array([[x1,y1],[x2,y2],[x3,y3]])
p0 = np.array([[x0,y0]])
Pr = Pi[1:,:]-Pi[:-1,:]
P  = np.ones([len(Pr)])@p0-Pi[:-1,:]
klist = sum((Pr*P).T)/sum((Pr*Pr).T)
klim = np.clip(klist,0,1)
Pn = np.diag(klim) @ Pr
PnP = Pn - P
l2 = sum((PnP*PnP).T)
idx = np.argmin(l2)
pout = Pn[idx,:]+Pi[idx,:]
#idx = np.argmin(klist)
#k = klist[idx]
```
ここで、簡単化のため行と列を入れ替えているので注意
