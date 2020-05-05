import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import numpy as np

# png
matplotlib.use('Agg')

"Уменьшение количества цветов изображения"


def recreate_image(codebook, labels, w, h, d):
    # Recreate the (compressed) image from the code book & labels
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def plot(image, fname):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(fname)


# Загрузите картинку parrots.jpg.
# Преобразуйте изображение, приведя все значения в интервал от 0 до
image = img_as_float(imread('parrots.jpg'))
# Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности в
# пространстве RGB.
h, w, d = image.shape
X = np.reshape(image.flatten(), (-1, 3))


def calculate():
    from sklearn.metrics import mean_squared_error
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for k in range(2, 22):
        cluster = KMeans(k, init='k-means++', random_state=241)
        cluster.fit(X)
        reduced_image = recreate_image(cluster.cluster_centers_, cluster.labels_, h, w, d)
        mse = np.mean((image - reduced_image) ** 2)
        # Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику нужно реализовать
        # самостоятельно
        psnr = 10 * np.log10(1.0 / mse)
        plot(reduced_image, "plots/plot%d.png" % (k))
        print("k: %d, mse: %.2f psnr: %.2f" % (k, mse, psnr))
        # Найдите минимальное количество кластеров, при котором значение PSNR выше 20
        if psnr > 20:
            return k


min_k = calculate()
with open('q1.txt', 'w') as output:
    output.write('%d' % (min_k))
