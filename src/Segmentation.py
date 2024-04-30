import numpy as np
import cv2


class Segmentation:

    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)

        self.modes_means = {'beginner': np.array([139.489, 99.937, 76.708]),
                            'intermediate1': np.array([73.192, 51.308, 64.041]),
                            'intermediate2': np.array([104.587,  91.975, 100.2])
                            }

        self.thresholds = {'beginner': {1: np.array([[68, 78],
                                                     [107, 125],
                                                     [146, 170]]),
                                        2: np.array([[22, 32],
                                                     [36, 42],
                                                     [37, 51]]),
                                        3: np.array([[0, 5],
                                                     [70, 110],
                                                     [110, 182]]),
                                        4: np.array([[56, 69],
                                                     [47, 68],
                                                     [40, 49]]),
                                        5: np.array([[17, 30],
                                                     [24, 34],
                                                     [110, 129]])},

                           'intermediate1': {1: np.array([[190, 255],
                                                          [180, 255],
                                                          [140, 255]]),
                                             2: np.array([[20, 85],
                                                          [20, 85],
                                                          [20, 85]]),
                                             3: np.array([[0, 30],
                                                          [110, 151],
                                                          [140, 230]]),
                                             4: np.array([[130, 165],
                                                          [100, 125],
                                                          [60, 100]]),
                                             5: np.array([[10, 45],
                                                          [0, 30],
                                                          [120, 160]])},

                           'intermediate2': {1: np.array([[252, 255],
                                                          [252, 255],
                                                          [252, 255]]),
                                             2: np.array([[105, 150],
                                                          [155, 205],
                                                          [125, 170]]),
                                             3: np.array([[70, 136],
                                                          [218, 255],
                                                          [245, 255]]),
                                             4: np.array([[235, 255],
                                                          [202, 246],
                                                          [137, 205]]),
                                             5: np.array([[46, 103],
                                                          [60, 138],
                                                          [215, 255]])}
                           }

    def count_circles(self, image, mode, kind):
        # get mask
        im = image.copy()
        values = self.thresholds[mode][kind]

        mask = ((im[..., 0] >= values[0, 0]) & (im[..., 0] <= values[0, 1]) &
                (im[..., 1] >= values[1, 0]) & (im[..., 1] <= values[1, 1]) &
                ((im[..., 2] >= values[2, 0]) & (im[..., 2] <= values[2, 1])))

        im[~mask] = 0

        # process mask
        if kind == 2:
            im = cv2.dilate(im, np.ones((15, 15), dtype=np.uint8))
        else:
            match mode:
                case 'beginner':
                    im = cv2.dilate(im, np.ones((3, 3), dtype=np.uint8))
                case 'intermediate2':
                    im = cv2.dilate(im, np.ones((2, 2), dtype=np.uint8))
                case 'intermediate1':
                    im = im
                case _:
                    raise ValueError('Invalid mode')

        kernel_size = 3
        if mode == 'beginner':
            kernel_size = 5

        im = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im[im > 0] = 255

        # count circles
        num_labels, _ = cv2.connectedComponents(im)

        return (num_labels - 1)

    def get_edges(self, image, mode):
        thresholds = [50, 200]
        if mode != 'beginner':
            thresholds = [140, 200]

        edges = cv2.Canny(image, thresholds[0], thresholds[1])
        edges[edges > 0] = 255
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

        return edges

    def fill_contours(self, contours, bgr_image, eroded=True):
        filled_cntrs = np.zeros_like(bgr_image)
        cv2.drawContours(filled_cntrs, contours, -1, (0, 255, 0), thickness=-1)
        if eroded:
            filled_cntrs = cv2.erode(
                filled_cntrs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        filled_cntrs = cv2.cvtColor(filled_cntrs, cv2.COLOR_RGB2GRAY)
        filled_cntrs[filled_cntrs > 0] = 255
        return filled_cntrs

    def get_center(self, contour):
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        return cx, cy

    def get_segmentation(self, bgr_image, mode):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = 0
        hsv[..., 1] = 0

        contours, _ = cv2.findContours(self.get_edges(
            hsv, mode), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = self.fill_contours(contours, bgr_image)
        segmentation_contours, _ = cv2.findContours(
            self.get_edges(segmentation, 'beginner'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triominoes = []
        centers = []

        for contour in segmentation_contours:
            filled = self.fill_contours([contour], bgr_image, eroded=False)

            if not np.all(filled == 0):
                triominoes.append(filled)
                centers.append(self.get_center(contour))

        return triominoes, (segmentation > 0), centers

    def define_mode(self, image):
        mode = 'beginner'
        im_means = image.mean(axis=(0, 1))
        for key, means in self.modes_means.items():
            if np.all((means - 15 <= im_means) & (im_means <= means + 15)):
                mode = key
                break

        return mode

    def process(self, image=None):
        if image is None:
            image = self.image

        mode = self.define_mode(image)
        triominoes, mask, centers = self.get_segmentation(image, mode)

        result_string = ''

        result_string += f'Число треугольников: {len(centers)}\n'

        for index, triomino in enumerate(triominoes):
            result_string += ', '.join(map(str, centers[index])) + '; '

            edge_ind = 0
            edges_values = [0, 0, 0]

            im = image.copy()
            im[triomino == 0] = 0

            for num in range(5):
                kind = num + 1

                count = self.count_circles(im, mode, kind)

                if count == 0:
                    continue

                if edge_ind > 2:
                    break

                border = kind if kind != 2 else 1

                while count >= border:
                    edges_values[edge_ind] = kind
                    if kind == 2:
                        count -= 1
                    else:
                        count -= kind
                    edge_ind += 1
                    if edge_ind > 2:
                        break

            result_string += ', '.join(map(str, edges_values)) + '\n'

        image[~mask] = 0
        return result_string, image
