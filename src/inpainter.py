import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import laplace
from skimage.color import rgb2gray, rgb2lab


class Inpainter():

    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        """
        Parameters:
            image: the image to be inpainted
            mask: the mask of the region to be inpainted, 255 for target region, 0 for reserved region
            patch_size: the size of the patches, must be an odd number
            plot_progress: whether to plot each generated image
        """
        # validate input
        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask must have the same size')
        if patch_size % 2 == 0:
            raise ValueError('Patch size must be an odd number')

        # initialize attributes
        self.working_image = image.astype('uint8')
        self.initial_mask = mask.astype('uint8') // 255  # convert to 0-1
        self.working_mask = mask.astype('uint8') // 255  # convert to 0-1
        self.patch_size = patch_size
        self.plot_progress = plot_progress
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.confidence = (1 - self.working_mask).astype(float)
        self.data = np.zeros([self.height, self.width])

        # Non initialized attributes
        self.front = None
        self.front_points = None
        self.priority = None

    def inpaint(self):
        """
        Inpaint the image
        """
        start_time = time.time()

        # iterate until the target region is filled
        while not self.__finished():

            self.__find_front()

            if self.plot_progress:
                self.__plot_image()

            self.__calculate_priority()
            target_point = self.__find_highest_priority_point()

            find_start_time = time.time()
            source_patch = self.__find_source_patch(target_point)
            print('Time to find best-match patch: %f seconds' %
                  (time.time() - find_start_time))

            self.__update_image(target_point, source_patch)

        print('Took {} seconds to complete'.format(time.time() - start_time))

        return self.working_image

    def __find_front(self):
        """
        Find the front of the target region and the points on it
        """
        # use laplacian to find the edge of the target region
        # set the edge to 1, and the rest to 0
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        self.front_points = np.argwhere(self.front == 1)

    def __calculate_priority(self):
        """
        Find the priority of each pixel in the front
        """
        new_confidence = self.__calculate_confidence()
        self.__update_data()

        # calculate the priority of each pixel in the front
        # the priority is the multiply of the confidence and the data
        self.priority = new_confidence * self.data * self.front

    def __calculate_confidence(self):
        """
        Calculate the confidence of each pixel in the front

        Return:
            the confidence with the update of each pixel in the front
        """
        # temporary confidence used in calculating priority
        new_confidence = np.copy(self.confidence)

        # get the position of the pixels in the front
        front_positions = np.argwhere(self.front == 1)

        for point in front_positions:

            # get the patch around the point
            patch = self.__get_patch(point)

            # calculate the confidence of the point
            new_confidence[point[0],
                           point[1]] = self.__get_patch_confidence(patch)

        return new_confidence

    def __update_data(self):
        """
        Update the data term of each pixel in the front
        """
        normal = self.__calculate_normal_matrix()
        gradient = self.__calculate_gradient_matrix()

        normal_gradient = normal * gradient
        self.data = np.sqrt(normal_gradient[:, :, 0]**2 +
                            normal_gradient[:, :, 1]**2) + 0.001

    def __calculate_normal_matrix(self):
        """
        Calculate the normal matrix of the image

        Return:
            the normal matrix of the image
        """

        # calculate gradient on working mask
        normal = np.array(np.gradient(self.working_mask.astype(float)))
        x_normal = normal[0]
        y_normal = normal[1]
        normal = np.dstack((x_normal, y_normal))

        # normalize the normal
        norm = self.__2d_to_3d(np.sqrt(x_normal**2 + y_normal**2), 2)
        norm[norm == 0] = 1
        unit_normal = normal / norm

        return unit_normal

    def __calculate_gradient_matrix(self):
        """
        Calculate the gradient matrix of the image

        Return:
            the gradient matrix of the image
        """
        # generate the gray image
        gray_image = rgb2gray(self.working_image)
        gray_image[self.working_mask == 1] = None

        # calculate gradient on gray image
        gradient = np.nan_to_num(np.array(np.gradient(gray_image)))

        # calculate the gradient magnitude
        gradient_val = gradient[0]**2 + gradient[1]**2

        max_gradient = np.zeros([self.height, self.width, 2])

        # find the maximum gradient in each patch of the front points
        for point in self.front_points:
            patch = self.__get_patch(point)
            patch_x_gradient = self.__get_patch_data(gradient[0], patch)
            patch_y_gradient = self.__get_patch_data(gradient[1], patch)
            patch_gradient_val = self.__get_patch_data(gradient_val, patch)

            # find the maximum gradient position in the patch
            patch_max_pos = np.unravel_index(patch_gradient_val.argmax(),
                                             patch_gradient_val.shape)

            # set to the orthogonal direction of the max gradient
            max_gradient[point[0], point[1],
                         0] = patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1],
                         1] = -patch_x_gradient[patch_max_pos]

        return max_gradient

    def __find_highest_priority_point(self):
        """
        Find the highest priority point in the front

        Return:
            the position of the highest priority point
        """
        max_priority = 0
        max_point = None

        # find the highest priority point in the front points
        for point in self.front_points:
            if max_point is None or self.priority[point[0],
                                                  point[1]] > max_priority:
                max_priority = self.priority[point[0], point[1]]
                max_point = point

        return max_point

    def __find_source_patch(self, target_pixel):
        """
        Find the source patch best matching the target patch

        Parameters:
            target_pixel: the position of the target pixel
        
        Return:
            best matching source patch
        """
        target_patch = self.__get_patch(target_pixel)
        patch_height, patch_width = self.__get_patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        # use lab color space to calculate the difference
        lab_image = rgb2lab(self.working_image)

        # iterate through the image to find the best matching source patch
        for x in range(self.height - patch_height + 1):
            for y in range(self.width - patch_width + 1):

                # construct the source patch
                source_patch = [[x, x + patch_height - 1],
                                [y, y + patch_width - 1]]

                # check if the source patch are not totally in the initial source region
                if self.__get_patch_data(self.initial_mask,
                                         source_patch).sum() != 0:
                    continue

                # calculate the difference between the target patch and the source patch
                difference = self.__calculate_patch_difference(
                    lab_image, target_patch, source_patch)

                # update the best matching source patch
                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference

        return best_match

    def __calculate_patch_difference(self, lab_image, target_patch,
                                     source_patch):
        """
        Calculate the difference between the target patch and the source patch

        Parameters:
            lab_image: the image in lab color space
            target_patch: the target patch
            source_patch: the source patch
        
        Return:
            the difference between the target patch and the source patch
        """
        # get the mask of the target patch
        mask = 1 - self.__get_patch_data(self.working_mask, target_patch)
        mask_3d = self.__2d_to_3d(mask)

        # calculate the squared distance between the target patch and the source patch filtered by the mask
        target_data = self.__get_patch_data(lab_image, target_patch) * mask_3d
        source_data = self.__get_patch_data(lab_image, source_patch) * mask_3d
        squared_distance = ((target_data - source_data)**2).sum()

        # calculate the euclidean distance between the target patch and the source patch
        # euclidean_distance = np.sqrt(
        #     (target_patch[0][0] - source_patch[0][0])**2 +
        #     (target_patch[1][0] - source_patch[1][0])**2)
        # return squared_distance + euclidean_distance

        return squared_distance

    def __update_image(self, target_pixel, source_patch):
        """
        Update the image, mask and confidence with the source patch

        Parameters:
            target_pixel: the target pixel
            source_patch: the source patch
        """
        target_patch = self.__get_patch(target_pixel)

        # relative position of the unfilled pixel in the target patch
        pixels_positions = np.argwhere(
            self.__get_patch_data(self.working_mask, target_patch) == 1)

        # bias of the pixels in the target patch
        x_bias, y_bias = target_patch[0][0], target_patch[1][0]

        # confidence of the target pixel
        patch_confidence = self.__get_patch_confidence(target_patch)

        # update the confidence of the pixels in the unfilled target patch
        for point in pixels_positions:
            self.confidence[point[0] + x_bias,
                            point[1] + y_bias] = patch_confidence

        # turn the mask of the target patch to 3d
        mask = self.__get_patch_data(self.working_mask, target_patch)
        rgb_mask = self.__2d_to_3d(mask)

        # fetch the data of the source patch and the target patch
        source_data = self.__get_patch_data(self.working_image, source_patch)
        target_data = self.__get_patch_data(self.working_image, target_patch)

        # reserve the already-filled pixels and fill the rest pixels using the source patch
        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)

        # update the image and the mask
        self.__copy_to_patch(self.working_image, target_patch, new_data)
        self.__copy_to_patch(self.working_mask, target_patch, 0)

    def __get_patch(self, point):
        """
        Get the patch around the point

        Parameters:
            point: the point (x, y) to get the patch around
        
        Return:
            the patch, a list of two lists, each list contains the begin and end of the patch
        """
        half_patch_size = (self.patch_size - 1) // 2
        x = point[0]
        y = point[1]
        x_begin = max(0, x - half_patch_size)
        x_end = min(self.height - 1, x + half_patch_size)
        y_begin = max(0, y - half_patch_size)
        y_end = min(self.width - 1, y + half_patch_size)

        return [[x_begin, x_end], [y_begin, y_end]]

    def __get_patch_area(self, patch):
        """
        Get the area of the patch

        Parameters:
            patch: the patch to get the area of
        
        Return:
            the area of the patch
        """
        return (patch[0][1] - patch[0][0] + 1) * (patch[1][1] - patch[1][0] +
                                                  1)

    def __get_patch_shape(self, patch):
        """
        Get the shape of the patch

        Parameters:
            patch: the patch to get the shape of
        
        Return:
            the shape of the patch
        """
        return (patch[0][1] - patch[0][0] + 1, patch[1][1] - patch[1][0] + 1)

    def __get_patch_data(self, matrix, patch):
        """
        Get the data of the patch in the matrix

        Parameters:
            matrix: the matrix to get the data from
            patch: the patch to get the data of
        
        Return:
            the data of the patch in the matrix
        """
        return matrix[patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1]

    def __get_patch_confidence(self, patch):
        """
        Get the confidence of the patch

        Parameters:
            patch: the patch to get the confidence

        Return:
            the confidence of the patch center pixel
        """
        return np.sum(self.__get_patch_data(
            self.confidence, patch)) / (self.__get_patch_area(patch))

    def __copy_to_patch(self, dest_matrix, dest_patch, data):
        """
        Copy the data to the patch in the destination matrix

        Parameters:
            dest_matrix: the destination matrix
            dest_patch: the destination patch
            data: the data to copy
        """
        dest_matrix[dest_patch[0][0]:dest_patch[0][1] + 1,
                    dest_patch[1][0]:dest_patch[1][1] + 1] = data

    def __finished(self):
        """
        Check if the inpainting is finished

        Return:
            True if the inpainting is finished, False otherwise
        """
        #calculate the number of pixels in the target region
        remaining = np.sum(self.working_mask)

        #calculate the total number of pixels
        total = self.height * self.width

        print('Remaining pixels: {} / {}'.format(remaining, total))
        return remaining == 0

    def __plot_image(self):
        """
        Plot the current image
        """
        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        inverse_mask = self.__2d_to_3d(inverse_mask)
        image = self.working_image * inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        white_region = self.__2d_to_3d(white_region)
        image += white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)

    def __2d_to_3d(self, matrix, channel_num=3):
        """
        Convert a 2D matrix to a 3D matrix with the same value in each channel

        Parameters:
            matrix: the 2D matrix to be converted

        Return: 
            the 3D matrix
        """
        return np.stack([matrix] * channel_num, axis=2)
