"""
DSC 20 Mid-Quarter Project
Name(s): Charlie Sun, Aaron Feng
PID(s):  A17370387, XXXXXXXXXXXX
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    A template for image objects in RGB color spaces. 
    """

    def __init__(self, pixels):
        """
        A constructor for an RGBImage instance that creates needed instance 
        variables.

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if not isinstance(pixels, list): 
            raise TypeError()
        elif len(pixels) < 1:
            raise TypeError()
        elif any(not isinstance(row, list) for row in pixels):
            raise TypeError()
        elif any(len(row) < 1 for row in pixels):
            raise TypeError()
        elif any(len(row) != len(pixels[0]) for row in pixels):
            raise TypeError()
        elif any(not isinstance(pixel, list) for row in pixels for pixel in \
            row):
            raise TypeError()
        elif any(len(pixel) != 3 for row in pixels for pixel in row):
            raise TypeError()
        elif any(not isinstance(value, int) for row in pixels for pixel in \
            row for value in pixel):
            raise TypeError()
        elif any(value < 0 or value > 255 for row in pixels for pixel in row \
            for value in pixel):
            raise TypeError()
        
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        A getter method that returns the size of the image, where size is 
        defined as a tuple of (number of rows, number of columns).

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        A getter method that returns a DEEP COPY of the pixels matrix.

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[value for value in pixel] for pixel in row] for row \
            in self.pixels]

    def copy(self):
        """
        A method that returns a COPY of the RGBImage instance.

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        A getter method that returns the color of the pixel at position 
        (row, col).

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        elif row > self.size()[0] - 1 or col > self.size()[1] - 1:
            raise ValueError()
        elif row < 0 or col < 0:
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        A setter method that updates the color of the pixel at position 
        (row, col) to the new_color. 

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        elif row > self.size()[0] - 1 or col > self.size()[1] - 1:
            raise ValueError()
        elif row < 0 or col < 0:
            raise ValueError()
        elif not isinstance(new_color, tuple): 
            raise TypeError()
        elif len(new_color) != 3: 
            raise TypeError()
        elif any(not isinstance(value, int) for value in new_color):
            raise TypeError()
        elif any(value > 255 for value in new_color):
            raise ValueError()
        
        for i in range(3): 
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
        
        
# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Implements several image processing methods.
    """

    def __init__(self):
        """
        A constructor that initializes an ImageProcessingTemplate instance 
        and necessary instance variables.

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        A getter method that returns the current total incurred cost.

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        A method that returns the negative of the given image. 

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        pixels = image.get_pixels()
        new_pixels = [[[255 - value for value in pixel] for pixel in row] \
            for row in pixels]
        return RGBImage(new_pixels)

    def grayscale(self, image):
        """
        Converts the given image to grayscale. 

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        pixels = image.get_pixels()
        new_pixels = [[[(pixel[0] + pixel[1] + pixel[2]) // 3, (pixel[0] + \
            pixel[1] + pixel[2]) // 3, (pixel[0] + pixel[1] + pixel[2]) // 3] \
            for pixel in row] for row in pixels]
        return RGBImage(new_pixels)

    def rotate_180(self, image):
        """
        Rotates the image 180 degrees. 

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        pixels = image.get_pixels()
        rows = image.size()[0]
        cols = image.size()[1]
        
        new_pixels = [[pixels[rows - i - 1][cols - j - 1] for j in range(cols)\
            ] for i in range(rows)]
        return RGBImage(new_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    A monetized version of the template class. Each method will add a 
    certain amount to the cost each time it is called.
    """

    def __init__(self):
        """
        A constructor that initializes a StandardImageProcessing instance and 
        necessary instance variables.

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.coupon = 0
        self.coupon_for_reverse = False
        self.reverse = False


    def negate(self, image):
        """
        Same as negate in the template, except it should add +5 to the cost 
        each time it is used.

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupon > 0:
            self.coupon -= 1
        else: 
            self.cost += 5
            
        return ImageProcessingTemplate.negate(self, image)
        

    def grayscale(self, image):
        """
        Same as grayscale in the template, except it should add +6 to the 
        cost each time it is used.

        """
        if self.coupon > 0:
            self.coupon -= 1
        else: 
            self.cost += 6
        
        return ImageProcessingTemplate.grayscale(self, image)
    
    
    def redeem_coupon(self, amount):
        """
        Makes the next amount image processing method calls free, as in 
        they add 0 to the cost.

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount, int):
            raise TypeError()
        elif amount <= 0:
            raise ValueError()
        
        self.coupon += amount
            
        
    def rotate_180(self, image):
        """
        Same as rotate_180 in the template, except it should add +10 to the
        cost each time it is used. When rotate_180 undoes itself the method 
        will be free, and the cost of the previous rotate_180 will be refunded. 

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img_proc.reverse
        True
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        >>> img_proc.reverse
        False
        """
        if self.reverse:                        # if rotating back (refund)
            # check if they used a coupon the first time
            if not self.coupon_for_reverse:
                self.cost -= 10
            # check if they have a coupon
            if self.coupon > 0:
                self.coupon -= 1
            # reset instance variables
            self.reverse = False
            self.coupon_for_reverse = False
        else:                                   # if rotating for first time
            # if have coupon, no need to pay
            if self.coupon > 0:
                self.coupon -= 1
                self.coupon_for_reverse = True
            # pay if don't have coupon
            else: 
                self.cost += 10
            self.reverse = True

        return ImageProcessingTemplate.rotate_180(self, image)


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    This version of the app should cost 50 dollars up front, but every method
    call after that will be free. Additionally, to draw in premium users it
    will have 2 new methods: chroma_key and sticker.
    """

    def __init__(self):
        """
        A constructor that initializes a PremiumImageProcessing instance 
        and necessary instance variables.

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50


    def chroma_key(self, chroma_image, background_image, color):
        """
        A method that performs a chroma key on the chroma_image by changing 
        all pixels with the specified color in the chroma_image to the pixels
        at the same places in the background_image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        if not isinstance(chroma_image, RGBImage):
            raise TypeError()
        elif not isinstance(background_image, RGBImage):
            raise TypeError()
        elif chroma_image.size() != background_image.size():
            raise ValueError()
        
        chroma = chroma_image.copy()
        for i in range(chroma.size()[0]):
            for j in range(chroma.size()[1]):
                if chroma.pixels[i][j] == list(color):
                    chroma.pixels[i][j] = background_image.pixels[i][j]

        return chroma


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        A method that creates a new image where the top left corner of the
        sticker is placed at position x_pos, y_pos on the given background 
        image.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage):
            raise TypeError()
        elif not isinstance(background_image, RGBImage):
            raise TypeError()
        elif sticker_image.size()[0] >= background_image.size()[0]:
            raise ValueError()
        elif sticker_image.size()[1] >= background_image.size()[1]:
            raise ValueError()
        elif not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        elif sticker_image.size()[0] + y_pos > background_image.size()[0]:
            raise ValueError()
        elif sticker_image.size()[1] + x_pos > background_image.size()[1]:
            raise ValueError()
        
        sticker_width = sticker_image.size()[1]
        sticker_height = sticker_image.size()[0]
        
        new_image = background_image.copy()
        for i in range(new_image.size()[0]):
            for j in range(new_image.size()[1]):
                if i >= y_pos and i < y_pos + sticker_width and j >= x_pos and j < x_pos + sticker_height:
                    # print(i - y_pos, j - x_pos)
                    new_image.pixels[i][j] = sticker_image.pixels[i - y_pos][j - x_pos]
                    
        return new_image
                
                    
# helper function for the final doctest
def create_random_pixels(low, high, nrows, ncols):
        """
        Create a random pixels matrix with dimensions of
        3 (channels) x `nrows` x `ncols`, and fill in integer
        values between `low` and `high` (both exclusive).
        """
        return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    A K-nearest Neighbors (KNN) classifier for the RGB images. Given an image,
    this algorithm will predict the label by finding the most popular labels 
    in a collection (with size k) of nearest training data. 
    """

    def __init__(self, n_neighbors):
        """
        A constructor that initializes a ImageKNNClassifier instance and 
        necessary instance variables.
        """
        self.n_neighbors = n_neighbors


    def fit(self, data):
        """
        Fit the classifier by storing all training data in the classifier
        instance. 
        """
        if len(data) <= self.n_neighbors:
            raise ValueError()
        if hasattr(self, "data"):
            raise ValueError()
        
        self.data = data
        

    @staticmethod
    def distance(image1, image2):
        """
        A method to calculate the Euclidean distance between RGB image 
        image1 and image2.
        """
        if not isinstance(image1, RGBImage):
            raise TypeError()
        elif not isinstance(image2, RGBImage):
            raise TypeError()
        elif image1.size()[0] != image2.size()[0]:
            raise ValueError()
        elif image1.size()[1] != image2.size()[1]:
            raise ValueError()
        
        diff_sum = sum([(image1.pixels[i][j][k] - image2.pixels[i][j][k]) ** 2 for i in range(image1.size()[0]) for j in range(image1.size()[1]) for k in range(2)])
        
        return diff_sum ** (1/2)


    @staticmethod
    def vote(candidates):
        """
        Find the most popular label from a list of candidates (nearest 
        neighbors) labels. If there is a tie when determining the majority 
        return any of them.
        """
        return max(set(candidates), key = candidates.count)
        

    def predict(self, image):
        """
        Predict the label of the given image using the KNN classification 
        algorithm.

        # make random training data (type: List[Tuple[RGBImage, str]])
        >>> train = []

        # create training images with low intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        ...     for _ in range(20)
        ... )

        # create training images with high intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        ...     for _ in range(20)
        ... )

        # initialize and fit the classifier
        >>> knn = ImageKNNClassifier(5)
        >>> knn.fit(train)

        # should be "low"
        >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
        low

        # can be either "low" or "high" randomly
        # >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
        # This will randomly be either low or high

        # should be "high"
        >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
        high
        """
        if not hasattr(self, "data"):
            raise ValueError()
        
        distances = sorted([(ImageKNNClassifier.distance(data_point[0], image), data_point[1]) for data_point in self.data])
        labels = [candidate[1] for candidate in distances[:self.n_neighbors]]
        return ImageKNNClassifier.vote(labels)
    
    
